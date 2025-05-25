import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dll.configs.training_config import TrainingConfig

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, learnable=False, num_classes=3, device=None):
        super().__init__()
        self.learnable = learnable
        self.num_classes = num_classes
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32, device=self.device))
            if alpha is not None:
                alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)
                if alpha_tensor.numel() == 1:
                    alpha_tensor = alpha_tensor.expand(num_classes)
                self.alpha = nn.Parameter(alpha_tensor)
            else:
                self.alpha = None
        else:
            self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32, device=self.device))
            if alpha is not None:
                alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=self.device)
                if alpha_tensor.numel() == 1:
                    alpha_tensor = alpha_tensor.expand(num_classes)
                self.register_buffer('alpha', alpha_tensor)
            else:
                self.alpha = None

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        gamma = self.gamma.clamp(min=0)
        focal_weight = (1 - pt) ** gamma

        if self.alpha is not None:
            clamped_alpha = self.alpha.clamp(0, 1)
            alpha_t = clamped_alpha[targets]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss

        return loss.mean()

    def extra_repr(self) -> str:
        gamma_val = self.gamma.item() if isinstance(self.gamma, torch.Tensor) else self.gamma
        if self.alpha is not None:
            alpha_str = f", alpha={self.alpha.detach().cpu().numpy()}"
        else:
            alpha_str = ""
        return f"gamma={gamma_val:.4f}{alpha_str}, learnable={self.learnable}"

class HeatmapLoss(nn.Module):
    def __init__(self, use_target_weight: bool = True):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred_heatmaps: torch.Tensor,
                gt_heatmaps: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_heatmaps: (B, K, H, W)
            gt_heatmaps: (B, K, H, W)
            target_weight: (B, K) or None
        Returns:
            Scalar loss
        """
        if pred_heatmaps.shape != gt_heatmaps.shape:
            raise ValueError(f"Shape mismatch: {pred_heatmaps.shape} vs {gt_heatmaps.shape}")

        loss = self.criterion(pred_heatmaps, gt_heatmaps)

        if self.use_target_weight:
            if target_weight is None:
                raise ValueError("target_weight is required when use_target_weight is True")
            weight = target_weight.view(loss.shape[0], loss.shape[1], 1, 1)
            loss = loss * weight

        return loss.mean()

class KeypointLoss(nn.Module):
    def __init__(self, num_keypoints: int, config: TrainingConfig, device: None):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.config = config
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.heatmap_criterion = HeatmapLoss(use_target_weight=True).to(self.device)
        self.keypoint_weights = self._init_keypoint_weights(num_keypoints).to(self.device)

        # Add visibility_criterion for compatibility with Trainer
        self.visibility_criterion = FocalLoss(
            gamma=config.loss.focal_gamma,
            alpha=config.loss.focal_alpha,
            learnable=config.loss.learnable_focal_params,
            num_classes=3,  # 0: not visible, 1: occluded, 2: visible
            device=self.device
        ).to(self.device)

    def forward(self, predictions : dict, targets : dict):
        """
        Args:
            predictions: Dictionary containing:
                - heatmaps: (B, K, H, W) Predicted heatmaps
                - visibilities: (B, K, 3) or (B, P, K, 3) Predicted visibility probabilities
            targets: Dictionary containing:
                - heatmaps: (B, K, H, W) Ground truth heatmaps
                - visibility: (B, K) or (B, P, K) Visibility flags (0: not visible, 1: occluded, 2: visible)
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        pred_heatmaps = predictions['heatmaps']
        gt_heatmaps = targets['heatmaps']

        if pred_heatmaps.shape != gt_heatmaps.shape:
            raise ValueError(f"Shape mismatch: predictions {pred_heatmaps.shape} vs targets {gt_heatmaps.shape}")

        batch_size = pred_heatmaps.size(0)

        # Compute heatmap loss
        if 'visibility' in targets:
            visibility = targets['visibility']
            if visibility.dim() == 3:  # [B, P, K]
                visibility = visibility.max(dim=1)[0]  # shape: (B, K)
            # Convert 3-class visibility to binary weights (visible if class 1 or 2)
            binary_visibility = (visibility > 0).float()
            weights = (self.keypoint_weights.view(1, -1) * binary_visibility)   # shape (B, K)
        else:
            weights = self.keypoint_weights.view(1, -1).expand(batch_size, -1)

        # Use simple MSE loss to avoid memory issues temporarily
        heatmap_loss = torch.nn.functional.mse_loss(pred_heatmaps, gt_heatmaps)

        # Temporarily disable visibility loss to debug memory issue
        visibility_loss = torch.tensor(0.0, device=pred_heatmaps.device)

        # Temporarily disable coordinate loss to debug memory issue
        coordinate_loss = torch.tensor(0.0, device=pred_heatmaps.device)

        # Combine losses with weights
        total_loss = (heatmap_loss +
                     0.1 * visibility_loss +
                     0.05 * coordinate_loss)  # Weight coordinate loss

        # Return both total loss and loss components
        loss_dict = {
            'heatmap_loss': heatmap_loss.item(),
            'visibility_loss': visibility_loss.item(),
            'coordinate_loss': coordinate_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def _init_keypoint_weights(self, num_keypoints):
        """Initialize keypoint importance weights with specific emphasis on key joints"""
        weights = torch.ones(num_keypoints)
        # Weights for specific keypoints based on their importance
        special_weights = {
            0: 1.5,  # nose - important for face orientation
            1: 1.5,  # neck - central body reference
            8: 1.2,  # left hip - important for pose
            9: 1.2,  # right hip - important for pose
            15: 1.3, # left ankle - important for stance
            16: 1.3  # right ankle - important for stance
        }
        for idx, weight in special_weights.items():
            if idx < num_keypoints:
                weights[idx] = weight
        return weights

import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dll.configs.training_config import TrainingConfig
from dll.utils.device_manager import get_device_manager

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

        # Use device manager if available, otherwise fallback to provided device
        try:
            device_manager = get_device_manager()
            self.device = device_manager.device
        except RuntimeError:
            self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Use improved weighted heatmap loss with config parameters
        # Handle both object and dict config formats
        weighted_loss_enabled = False
        weighted_loss_config = None

        if hasattr(config.loss, 'weighted_loss'):
            weighted_loss_config = config.loss.weighted_loss
            # Check if it's an object with .enabled attribute
            if hasattr(weighted_loss_config, 'enabled'):
                weighted_loss_enabled = weighted_loss_config.enabled
            # Check if it's a dict with 'enabled' key
            elif isinstance(weighted_loss_config, dict) and 'enabled' in weighted_loss_config:
                weighted_loss_enabled = weighted_loss_config['enabled']

        if weighted_loss_enabled and weighted_loss_config:
            # Extract parameters based on config type
            if hasattr(weighted_loss_config, 'keypoint_weight'):
                # Object format
                keypoint_weight = weighted_loss_config.keypoint_weight
                background_weight = weighted_loss_config.background_weight
                threshold = weighted_loss_config.threshold
            else:
                # Dict format
                keypoint_weight = weighted_loss_config.get('keypoint_weight', 15.0)
                background_weight = weighted_loss_config.get('background_weight', 1.0)
                threshold = weighted_loss_config.get('threshold', 0.1)

            # Import WeightedHeatmapLoss if not already imported
            try:
                from dll.losses.keypoint_loss import WeightedHeatmapLoss
            except ImportError:
                # Define WeightedHeatmapLoss locally if import fails
                class WeightedHeatmapLoss(nn.Module):
                    def __init__(self, use_target_weight: bool = True, keypoint_weight: float = 10.0,
                                 background_weight: float = 1.0, threshold: float = 0.1):
                        super().__init__()
                        self.use_target_weight = use_target_weight
                        self.keypoint_weight = keypoint_weight
                        self.background_weight = background_weight
                        self.threshold = threshold
                        self.criterion = nn.MSELoss(reduction='none')

                    def forward(self, pred_heatmaps: torch.Tensor,
                                gt_heatmaps: torch.Tensor,
                                target_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
                        if pred_heatmaps.shape != gt_heatmaps.shape:
                            raise ValueError(f"Shape mismatch: {pred_heatmaps.shape} vs {gt_heatmaps.shape}")

                        # Compute MSE loss
                        loss = self.criterion(pred_heatmaps, gt_heatmaps)

                        # Create weight map: higher weight for keypoint regions
                        keypoint_mask = (gt_heatmaps > self.threshold).float()
                        background_mask = (gt_heatmaps <= self.threshold).float()

                        weight_map = (keypoint_mask * self.keypoint_weight +
                                     background_mask * self.background_weight)

                        # Apply weight map
                        loss = loss * weight_map

                        # Apply target weight if provided
                        if self.use_target_weight and target_weight is not None:
                            target_weight_expanded = target_weight.view(loss.shape[0], loss.shape[1], 1, 1)
                            loss = loss * target_weight_expanded

                        return loss.mean()

            self.heatmap_criterion = WeightedHeatmapLoss(
                use_target_weight=True,
                keypoint_weight=keypoint_weight,
                background_weight=background_weight,
                threshold=threshold
            ).to(self.device)
        else:
            # Fallback to regular heatmap loss
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

        # Add coordinate loss criterion
        self.coordinate_criterion = nn.SmoothL1Loss(reduction='none')

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

        # Compute heatmap loss
        heatmap_loss = self.heatmap_criterion(pred_heatmaps, gt_heatmaps, weights)

        # Compute visibility loss if visibility predictions are available
        visibility_loss = torch.tensor(0.0, device=self.device)
        if 'visibilities' in predictions and 'visibility' in targets:
            try:
                pred_vis = predictions['visibilities']
                target_vis = targets['visibility']

                # Ensure tensors are on the same device
                pred_vis = pred_vis.to(self.device)
                target_vis = target_vis.to(self.device)

                # Handle different dimensions
                if pred_vis.dim() == 4:  # [B, P, K, 3]
                    pred_vis = pred_vis.view(-1, pred_vis.size(-1))  # [B*P*K, 3]
                elif pred_vis.dim() == 3:  # [B, K, 3] or [B, P, K]
                    if pred_vis.size(-1) == 3:
                        pred_vis = pred_vis.view(-1, 3)  # [B*K, 3]
                    else:  # [B, P, K]
                        pred_vis = pred_vis.view(-1)  # [B*P*K]

                if target_vis.dim() == 3:  # [B, P, K]
                    target_vis = target_vis.view(-1)  # [B*P*K]
                elif target_vis.dim() == 2:  # [B, K]
                    target_vis = target_vis.view(-1)  # [B*K]

                # Ensure target_vis is long type for cross entropy
                target_vis = target_vis.long().clamp(0, 2)

                if pred_vis.size(0) == target_vis.size(0):
                    visibility_loss = self.visibility_criterion(pred_vis, target_vis)

            except Exception as e:
                print(f"Warning: Could not compute visibility loss: {e}")
                visibility_loss = torch.tensor(0.0, device=self.device)

        # Compute coordinate loss if coordinate predictions are available
        coordinate_loss = torch.tensor(0.0, device=self.device)
        if 'coordinates' in predictions and 'keypoints' in targets:
            try:
                pred_coords = predictions['coordinates']
                target_coords = targets['keypoints']

                # Ensure tensors are on the same device
                pred_coords = pred_coords.to(self.device)
                target_coords = target_coords.to(self.device)

                # Handle different dimensions and extract x,y coordinates
                if target_coords.dim() == 4:  # [B, P, K, 2]
                    target_coords = target_coords.view(-1, 2)  # [B*P*K, 2]
                elif target_coords.dim() == 3:  # [B, K, 2]
                    target_coords = target_coords.view(-1, 2)  # [B*K, 2]

                if pred_coords.dim() == 4:  # [B, P, K, 2]
                    pred_coords = pred_coords.view(-1, 2)  # [B*P*K, 2]
                elif pred_coords.dim() == 3:  # [B, K, 2]
                    pred_coords = pred_coords.view(-1, 2)  # [B*K, 2]

                if pred_coords.size(0) == target_coords.size(0):
                    # Only compute loss for visible keypoints
                    if 'visibility' in targets:
                        vis_mask = targets['visibility'].to(self.device)
                        if vis_mask.dim() == 3:  # [B, P, K]
                            vis_mask = vis_mask.view(-1)  # [B*P*K]
                        elif vis_mask.dim() == 2:  # [B, K]
                            vis_mask = vis_mask.view(-1)  # [B*K]

                        # Only compute loss for visible keypoints (visibility > 0)
                        visible_mask = (vis_mask > 0).float()
                        if visible_mask.sum() > 0:
                            coord_loss = self.coordinate_criterion(pred_coords, target_coords)
                            coord_loss = coord_loss.mean(dim=1)  # Average over x,y
                            coordinate_loss = (coord_loss * visible_mask).sum() / visible_mask.sum()
                    else:
                        coordinate_loss = self.coordinate_criterion(pred_coords, target_coords).mean()

            except Exception as e:
                print(f"Warning: Could not compute coordinate loss: {e}")
                coordinate_loss = torch.tensor(0.0, device=self.device)

        # Combine losses with weights from config
        total_loss = (
            self.config.lambda_keypoint * heatmap_loss +
            self.config.lambda_visibility * visibility_loss +
            0.05 * coordinate_loss  # Weight coordinate loss
        )

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

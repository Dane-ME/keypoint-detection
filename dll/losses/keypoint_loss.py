import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dll.configs.training_config import TrainingConfig

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, learnable=False, num_classes=3):
        super().__init__()
        self.learnable = learnable
        self.num_classes = num_classes
        
        # Khởi tạo gamma
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
            
            # Khởi tạo alpha là tensor với số phần tử bằng số lớp
            if alpha is not None:
                if isinstance(alpha, (float, int)):
                    # Nếu alpha là số vô hướng, tạo tensor với giá trị đồng nhất
                    self.alpha = nn.Parameter(torch.ones(num_classes) * alpha)
                else:
                    # Nếu alpha đã là list/tensor, chuyển thành Parameter
                    self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
            else:
                self.alpha = None
        else:
            self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))
            if alpha is not None:
                if isinstance(alpha, (float, int)):
                    self.register_buffer('alpha', torch.ones(num_classes) * alpha)
                else:
                    self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            else:
                self.alpha = None

    def forward(self, inputs, targets):
        # Đảm bảo targets là Long tensor
        targets = targets.long()
        
        # Tính cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Tính focal loss với gamma có thể học được
        pt = torch.exp(-ce_loss)
        
        # Áp dụng gamma
        focal_weight = (1 - pt) ** self.gamma.clamp(min=0)  # Đảm bảo gamma không âm
        
        # Áp dụng alpha nếu có
        if self.alpha is not None:
            # Đảm bảo alpha nằm trong khoảng [0, 1]
            clamped_alpha = self.alpha.clamp(0, 1)
            # Lấy alpha tương ứng với targets
            alpha_t = clamped_alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
        
    def extra_repr(self) -> str:
        """Hiển thị thông tin bổ sung khi print model"""
        gamma_info = f"gamma={self.gamma.item():.4f}"
        alpha_info = f", alpha={self.alpha.item():.4f}" if self.alpha is not None else ""
        learnable_info = f", learnable={self.learnable}"
        return f"{gamma_info}{alpha_info}{learnable_info}"

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
            pred_heatmaps: (B, K, H, W) Predicted heatmaps
            gt_heatmaps: (B, K, H, W) Ground truth heatmaps
            target_weight: (B, K) Weights for each keypoint
        Returns:
            loss: Scalar loss value
        """
        batch_size = pred_heatmaps.size(0)
        num_keypoints = pred_heatmaps.size(1)
        
        # Calculate MSE loss for each point
        heatmaps_loss = self.criterion(pred_heatmaps, gt_heatmaps)
        
        if self.use_target_weight:
            if target_weight is None:
                raise ValueError("target_weight is required when use_target_weight is True")
            # Apply keypoint weights: (B, K, 1, 1)
            weight = target_weight.view(batch_size, num_keypoints, 1, 1)
            heatmaps_loss = heatmaps_loss * weight
            
        return heatmaps_loss.mean()

class KeypointLoss(nn.Module):
    def __init__(self, num_keypoints: int, config: TrainingConfig):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.heatmap_criterion = HeatmapLoss(use_target_weight=True).to(self.device)
        self.keypoint_weights = self._init_keypoint_weights(num_keypoints).to(self.device)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dictionary containing:
                - heatmaps: (B, K, H, W) Predicted heatmaps
            targets: Dictionary containing:
                - heatmaps: (B, K, H, W) Ground truth heatmaps
                - visibility: (B, K) Visibility flags for each keypoint (0: invisible, 1: visible)
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        # Validate input shapes
        pred_heatmaps = predictions['heatmaps']
        gt_heatmaps = targets['heatmaps']
        
        if pred_heatmaps.shape != gt_heatmaps.shape:
            raise ValueError(f"Shape mismatch: predictions {pred_heatmaps.shape} vs targets {gt_heatmaps.shape}")
            
        batch_size = pred_heatmaps.size(0)
        num_keypoints = pred_heatmaps.size(1)
        
        # Apply visibility mask if provided
        if 'visibility' in targets:
            visibility = targets['visibility'].to(self.device)
            # Nếu visibility có thêm dimension (như (B, P, K)), giảm theo person:
            if visibility.dim() == 3:
                visibility = visibility.max(dim=1)[0]  # shape: (B, K)
            # Ép self.keypoint_weights (shape (K,)) thành (1, K) và nhân với visibility (B, K)
            weights = (self.keypoint_weights.view(1, -1) * visibility)   # shape (B, K)
        else:
            weights = self.keypoint_weights.view(1, -1).expand(batch_size, -1)
            
        # Calculate heatmap loss; trong HeatmapLoss forward, chúng ta sẽ view weights thành (B, K, 1, 1)
        heatmap_loss = self.heatmap_criterion(
            pred_heatmaps,
            gt_heatmaps,
            weights  # weights shape (B, K)
        )
        
        # Return both total loss and loss components
        loss_dict = {
            'heatmap_loss': heatmap_loss.item(),
            'total_loss': heatmap_loss.item()
        }
        
        return heatmap_loss, loss_dict

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

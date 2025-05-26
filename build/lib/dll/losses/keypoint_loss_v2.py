import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dll.configs.training_config import TrainingConfig

class HeatmapFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps [B, K, H, W]
            target: Target heatmaps [B, K, H, W]
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()

class CoordinateLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, visibility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted coordinates [B, K, 2]
            target: Target coordinates [B, K, 2]
            visibility: Visibility mask [B, K]
        Returns:
            L1 loss value
        """
        loss = F.l1_loss(pred, target, reduction='none')
        if visibility is not None:
            loss = loss * visibility.unsqueeze(-1)
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class VisibilityLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted visibility probabilities [B, K, 3]
            target: Target visibility classes [B, K]
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = focal_weight * alpha_weight
            
        loss = focal_weight * ce_loss
        return loss.mean()

class KeypointLoss(nn.Module):
    def __init__(self, num_keypoints: int, config: TrainingConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.config = config
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize loss components
        self.heatmap_loss = HeatmapFocalLoss(
            alpha=config.loss.focal_alpha or 0.25,
            gamma=config.loss.focal_gamma
        ).to(self.device)
        
        self.coordinate_loss = CoordinateLoss().to(self.device)
        
        self.visibility_loss = VisibilityLoss(
            gamma=config.loss.focal_gamma,
            alpha=config.loss.focal_alpha
        ).to(self.device)
        
        # Initialize keypoint weights
        self.keypoint_weights = self._init_keypoint_weights(num_keypoints).to(self.device)
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Dictionary containing:
                - heatmaps: [B, K, H, W] Predicted heatmaps
                - coordinates: [B, K, 2] Predicted coordinates
                - visibilities: [B, K, 3] Predicted visibility probabilities
            targets: Dictionary containing:
                - heatmaps: [B, K, H, W] Target heatmaps
                - coordinates: [B, K, 2] Target coordinates
                - visibility: [B, K] Target visibility classes
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        # Heatmap loss
        heatmap_loss = self.heatmap_loss(predictions['heatmaps'], targets['heatmaps'])
        
        # Coordinate loss with visibility weighting
        coord_loss = self.coordinate_loss(
            predictions['coordinates'],
            targets['coordinates'],
            targets['visibility'] > 0
        )
        
        # Visibility loss
        vis_loss = self.visibility_loss(predictions['visibilities'], targets['visibility'])
        
        # Combine losses with weights from config
        total_loss = (
            self.config.loss.keypoint_loss_weight * heatmap_loss +
            self.config.loss.coordinate_loss_weight * coord_loss +
            self.config.loss.visibility_loss_weight * vis_loss
        )
        
        # Add L2 regularization if specified
        if self.config.loss.l2_lambda > 0:
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            total_loss = total_loss + self.config.loss.l2_lambda * l2_reg
        
        loss_dict = {
            'heatmap_loss': heatmap_loss.item(),
            'coordinate_loss': coord_loss.item(),
            'visibility_loss': vis_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
        
    def _init_keypoint_weights(self, num_keypoints: int) -> torch.Tensor:
        """Initialize keypoint importance weights"""
        weights = torch.ones(num_keypoints)
        # Weights for specific keypoints based on their importance
        special_weights = {
            0: 1.5,  # nose
            1: 1.5,  # neck
            8: 1.2,  # left hip
            9: 1.2,  # right hip
            15: 1.3, # left ankle
            16: 1.3  # right ankle
        }
        for idx, weight in special_weights.items():
            if idx < num_keypoints:
                weights[idx] = weight
        return weights 
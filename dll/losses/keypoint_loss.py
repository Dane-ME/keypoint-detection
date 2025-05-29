"""
Improved Keypoint Loss Architecture - Phase 1.1 Implementation

This module addresses the critical loss scaling issue causing stagnant PCK metrics
by implementing dynamic loss balancing and spatial-aware coordinate loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LossMetrics:
    """Container for loss component metrics"""
    total_loss: float
    heatmap_loss: float
    coordinate_loss: float
    visibility_loss: float
    loss_weights: Dict[str, float]


class DynamicLossBalancer:
    """
    Dynamic loss balancing to prevent component dominance and ensure
    all loss components contribute meaningfully to learning.
    """

    def __init__(self,
                 initial_weights: Dict[str, float] = None,
                 adaptation_rate: float = 0.1,
                 min_weight: float = 0.1,
                 max_weight: float = 10.0):
        """
        Initialize dynamic loss balancer.

        Args:
            initial_weights: Initial weights for loss components
            adaptation_rate: Rate of weight adaptation (0.0 to 1.0)
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize weights
        self.weights = initial_weights or {
            'heatmap': 1.0,
            'coordinate': 1.0,
            'visibility': 1.0
        }

        # Track loss history for adaptation
        self.loss_history = {key: [] for key in self.weights.keys()}
        self.update_count = 0

    def update_weights(self, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        Update loss weights based on component magnitudes.

        The goal is to balance loss components so they contribute roughly equally
        to the total gradient magnitude.
        """
        self.update_count += 1

        # Store current losses
        for key, value in loss_components.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
                # Keep only recent history
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]

        # Adapt weights every 10 updates
        if self.update_count % 10 == 0:
            self._adapt_weights()

        return self.weights.copy()

    def _adapt_weights(self):
        """Adapt weights based on loss component statistics"""
        if len(self.loss_history['heatmap']) < 10:
            return

        # Calculate recent average losses
        recent_losses = {}
        for key in self.weights.keys():
            if self.loss_history[key]:
                recent_losses[key] = sum(self.loss_history[key][-10:]) / 10
            else:
                recent_losses[key] = 1.0

        # Find the component with median loss magnitude
        loss_values = list(recent_losses.values())
        loss_values.sort()
        target_magnitude = loss_values[len(loss_values) // 2]

        # Adjust weights to balance components toward target magnitude
        for key in self.weights.keys():
            if recent_losses[key] > 0:
                # If loss is too large, reduce weight; if too small, increase weight
                ratio = target_magnitude / recent_losses[key]
                adjustment = (ratio - 1.0) * self.adaptation_rate

                new_weight = self.weights[key] * (1.0 + adjustment)
                self.weights[key] = max(self.min_weight, min(self.max_weight, new_weight))

        logger.debug(f"Adapted loss weights: {self.weights}")


class SpatialCoordinateLoss(nn.Module):
    """
    Spatial-aware coordinate loss that properly weights pixel-space errors.

    This addresses the issue where coordinate loss was too small compared to
    other components, preventing effective keypoint localization learning.
    """

    def __init__(self,
                 loss_type: str = 'smooth_l1',
                 pixel_weight_scale: float = 100.0,
                 distance_threshold: float = 5.0):
        """
        Initialize spatial coordinate loss.

        Args:
            loss_type: Type of loss ('smooth_l1', 'mse', 'huber')
            pixel_weight_scale: Scale factor for pixel-space errors
            distance_threshold: Threshold for distance-based weighting
        """
        super().__init__()
        self.loss_type = loss_type
        self.pixel_weight_scale = pixel_weight_scale
        self.distance_threshold = distance_threshold

        if loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self,
                pred_coords: torch.Tensor,
                gt_coords: torch.Tensor,
                visibility: torch.Tensor,
                image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Compute spatial coordinate loss.

        Args:
            pred_coords: Predicted coordinates (B, N, K, 2)
            gt_coords: Ground truth coordinates (B, N, K, 2)
            visibility: Visibility mask (B, N, K)
            image_size: Image size for normalization

        Returns:
            Scalar loss value
        """
        # Only compute loss for visible keypoints
        visible_mask = (visibility > 0).float()

        if visible_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_coords.device, requires_grad=True)

        # Compute coordinate differences
        coord_diff = pred_coords - gt_coords

        # Apply loss function
        if self.loss_type == 'smooth_l1':
            loss = self.criterion(pred_coords, gt_coords)
        else:
            loss = self.criterion(pred_coords, gt_coords)

        # Sum over coordinate dimensions (x, y)
        loss = loss.sum(dim=-1)  # (B, N, K)

        # Apply visibility mask
        loss = loss * visible_mask

        # Distance-based weighting: give more weight to larger errors
        distances = torch.norm(coord_diff, dim=-1)  # (B, N, K)
        distance_weights = torch.clamp(distances / self.distance_threshold, min=1.0, max=3.0)
        loss = loss * distance_weights

        # Scale to pixel space magnitude
        loss = loss * self.pixel_weight_scale

        # Average over valid keypoints
        valid_count = visible_mask.sum()
        return loss.sum() / (valid_count + 1e-8)


class AdaptiveHeatmapLoss(nn.Module):
    """
    Adaptive heatmap loss with region-specific weighting and improved
    focus on keypoint regions vs background.
    """

    def __init__(self,
                 keypoint_weight: float = 50.0,
                 background_weight: float = 1.0,
                 adaptive_threshold: bool = True,
                 focal_alpha: float = 2.0):
        """
        Initialize adaptive heatmap loss.

        Args:
            keypoint_weight: Weight for keypoint regions
            background_weight: Weight for background regions
            adaptive_threshold: Whether to use adaptive thresholding
            focal_alpha: Focal loss parameter for hard example mining
        """
        super().__init__()
        self.keypoint_weight = keypoint_weight
        self.background_weight = background_weight
        self.adaptive_threshold = adaptive_threshold
        self.focal_alpha = focal_alpha

    def _compute_adaptive_threshold(self, gt_heatmaps: torch.Tensor) -> torch.Tensor:
        """Compute adaptive threshold based on heatmap statistics"""
        if not self.adaptive_threshold:
            return torch.tensor(0.1, device=gt_heatmaps.device)

        # Use percentile-based threshold
        flattened = gt_heatmaps.flatten()
        threshold = torch.quantile(flattened, 0.9)
        return torch.clamp(threshold, min=0.05, max=0.3)

    def forward(self,
                pred_heatmaps: torch.Tensor,
                gt_heatmaps: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive heatmap loss.

        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W)
            gt_heatmaps: Ground truth heatmaps (B, K, H, W)
            target_weight: Target weights (B, K)

        Returns:
            Scalar loss value
        """
        # Compute adaptive threshold
        threshold = self._compute_adaptive_threshold(gt_heatmaps)

        # Create region masks
        keypoint_mask = (gt_heatmaps > threshold).float()
        background_mask = (gt_heatmaps <= threshold).float()

        # Compute base MSE loss
        mse_loss = F.mse_loss(pred_heatmaps, gt_heatmaps, reduction='none')

        # Apply region-specific weights
        weighted_loss = (mse_loss * keypoint_mask * self.keypoint_weight +
                        mse_loss * background_mask * self.background_weight)

        # Apply focal weighting for hard examples
        if self.focal_alpha > 0:
            pt = torch.exp(-mse_loss)
            focal_weight = (1 - pt) ** self.focal_alpha
            weighted_loss = weighted_loss * focal_weight

        # Apply target weights if provided
        if target_weight is not None:
            target_weight_expanded = target_weight.view(
                weighted_loss.shape[0], weighted_loss.shape[1], 1, 1
            )
            weighted_loss = weighted_loss * target_weight_expanded

        return weighted_loss.mean()


class ImprovedKeypointLoss(nn.Module):
    """
    Improved keypoint loss with dynamic balancing and spatial awareness.

    This addresses the critical issue of stagnant PCK metrics by ensuring
    all loss components contribute meaningfully to learning.
    """

    def __init__(self,
                 num_keypoints: int,
                 config,
                 device: Optional[torch.device] = None):
        """Initialize improved keypoint loss"""
        super().__init__()

        self.num_keypoints = num_keypoints
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize loss components
        self.heatmap_loss = AdaptiveHeatmapLoss()
        self.coordinate_loss = SpatialCoordinateLoss()
        self.visibility_loss = nn.CrossEntropyLoss()

        # Initialize dynamic loss balancer
        initial_weights = {
            'heatmap': getattr(config, 'lambda_keypoint', 1.0),
            'coordinate': 5.0,  # Higher initial weight for coordinate loss
            'visibility': getattr(config, 'lambda_visibility', 1.0)
        }
        self.loss_balancer = DynamicLossBalancer(initial_weights=initial_weights)

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute improved keypoint loss with dynamic balancing.

        Args:
            predictions: Model outputs containing predictions
            targets: Batch data containing ground truth

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Combined weighted loss value
            - loss_dict: Dictionary containing individual loss components
        """
        # Extract predictions and ground truth
        pred_heatmaps = predictions.get('heatmaps')
        pred_coords = predictions.get('keypoints')
        pred_visibility = predictions.get('visibility', predictions.get('visibilities'))

        gt_heatmaps = targets.get('heatmaps')
        gt_coords = targets.get('keypoints')
        gt_visibility = targets.get('visibilities', targets.get('visibility'))
        target_weight = targets.get('target_weight')

        # Compute individual loss components
        loss_components = {}

        # Heatmap loss
        if pred_heatmaps is not None and gt_heatmaps is not None:
            loss_components['heatmap'] = self.heatmap_loss(pred_heatmaps, gt_heatmaps, target_weight)
        else:
            loss_components['heatmap'] = torch.tensor(0.0, device=self.device)

        # Coordinate loss
        if pred_coords is not None and gt_coords is not None and gt_visibility is not None:
            loss_components['coordinate'] = self.coordinate_loss(
                pred_coords, gt_coords, gt_visibility
            )
        else:
            loss_components['coordinate'] = torch.tensor(0.0, device=self.device)

        # Visibility loss
        if pred_visibility is not None and gt_visibility is not None:
            # Reshape for cross entropy
            pred_vis_flat = pred_visibility.view(-1, pred_visibility.size(-1))
            gt_vis_flat = gt_visibility.view(-1).long()
            loss_components['visibility'] = self.visibility_loss(pred_vis_flat, gt_vis_flat)
        else:
            loss_components['visibility'] = torch.tensor(0.0, device=self.device)

        # Get current loss values for balancing
        current_losses = {k: v.item() for k, v in loss_components.items()}

        # Update dynamic weights
        weights = self.loss_balancer.update_weights(current_losses)

        # Compute weighted total loss
        total_loss = sum(weights[k] * loss_components[k] for k in loss_components.keys())

        # Create loss dictionary for monitoring (matching original interface)
        loss_dict = {
            'keypoint_loss': loss_components['heatmap'].item(),  # For backward compatibility
            'heatmap_loss': loss_components['heatmap'].item(),
            'visibility_loss': loss_components['visibility'].item(),
            'coordinate_loss': loss_components['coordinate'].item(),
            'total_loss': total_loss.item(),
            'loss_weights': weights,
            'loss_metrics': LossMetrics(
                total_loss=total_loss.item(),
                heatmap_loss=loss_components['heatmap'].item(),
                coordinate_loss=loss_components['coordinate'].item(),
                visibility_loss=loss_components['visibility'].item(),
                loss_weights=weights
            )
        }

        return total_loss, loss_dict


# Alias for backward compatibility
KeypointLoss = ImprovedKeypointLoss

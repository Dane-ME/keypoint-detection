from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from dll.configs.training_config import TrainingConfig
from dll.utils.device_manager import get_device_manager


@dataclass
class LossWeights:
    """Configuration for loss component weights"""
    heatmap: float = 1.0
    visibility: float = 1.0
    coordinate: float = 1.0

    def __post_init__(self):
        """Validate weights are non-negative"""
        for field_name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{field_name} weight must be non-negative, got {value}")


@dataclass
class LossScaling:
    """Configuration for loss scaling factors"""
    coordinate_scale: float = 10.0
    visibility_scale: float = 2.0

    def __post_init__(self):
        """Validate scaling factors are positive"""
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{field_name} scale must be positive, got {value}")


class LossConfigurationError(Exception):
    """Raised when loss configuration is invalid"""
    pass


class LossComputationError(Exception):
    """Raised when loss computation fails"""
    pass


class ConfigurationHandler:
    """Handles different configuration formats for loss components"""

    @staticmethod
    def extract_weighted_loss_config(config: TrainingConfig) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Extract weighted loss configuration from training config.

        Args:
            config: Training configuration object

        Returns:
            Tuple of (enabled, config_dict)

        Raises:
            LossConfigurationError: If configuration is invalid
        """
        try:
            if not hasattr(config.loss, 'weighted_loss'):
                return False, None

            weighted_loss_config = config.loss.weighted_loss

            # Handle object format
            if hasattr(weighted_loss_config, 'enabled'):
                enabled = weighted_loss_config.enabled
                if enabled:
                    return True, {
                        'keypoint_weight': getattr(weighted_loss_config, 'keypoint_weight', 15.0),
                        'background_weight': getattr(weighted_loss_config, 'background_weight', 1.0),
                        'threshold': getattr(weighted_loss_config, 'threshold', 0.1)
                    }
                return False, None

            # Handle dict format
            elif isinstance(weighted_loss_config, dict):
                enabled = weighted_loss_config.get('enabled', False)
                if enabled:
                    return True, {
                        'keypoint_weight': weighted_loss_config.get('keypoint_weight', 15.0),
                        'background_weight': weighted_loss_config.get('background_weight', 1.0),
                        'threshold': weighted_loss_config.get('threshold', 0.1)
                    }
                return False, None

            return False, None

        except Exception as e:
            raise LossConfigurationError(f"Failed to extract weighted loss config: {e}")

    @staticmethod
    def extract_focal_loss_config(config: TrainingConfig) -> Dict[str, Any]:
        """Extract focal loss configuration"""
        try:
            return {
                'gamma': getattr(config.loss, 'focal_gamma', 2.0),
                'alpha': getattr(config.loss, 'focal_alpha', None),
                'learnable': getattr(config.loss, 'learnable_focal_params', False),
                'num_classes': 3
            }
        except Exception as e:
            raise LossConfigurationError(f"Failed to extract focal loss config: {e}")


class DeviceManager:
    """Centralized device management for loss components"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = self._resolve_device(device)

    def _resolve_device(self, device: Optional[torch.device]) -> torch.device:
        """Resolve device with fallback strategy"""
        if device is not None:
            return device

        try:
            device_manager = get_device_manager()
            return device_manager.device
        except (RuntimeError, AssertionError):
            # Handle both RuntimeError and CUDA AssertionError
            return torch.device('cpu')

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to managed device"""
        return tensor.to(self.device)

    def ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device"""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

class FocalLoss(nn.Module):
    """
    Improved Focal Loss implementation with better parameter handling and validation.

    Focal Loss addresses class imbalance by down-weighting easy examples and focusing
    on hard examples during training.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None,
                 learnable: bool = False, num_classes: int = 3,
                 device: Optional[torch.device] = None):
        """
        Initialize Focal Loss.

        Args:
            gamma: Focusing parameter (higher values focus more on hard examples)
            alpha: Class balancing parameter
            learnable: Whether gamma and alpha are learnable parameters
            num_classes: Number of classes
            device: Device to place tensors on
        """
        super().__init__()
        self.learnable = learnable
        self.num_classes = num_classes
        self.device_manager = DeviceManager(device)

        # Validate parameters
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self._initialize_parameters(gamma, alpha)

    def _initialize_parameters(self, gamma: float, alpha: Optional[float]):
        """Initialize gamma and alpha parameters"""
        # Create tensors on CPU first, then move to device to avoid CUDA issues
        gamma_tensor = torch.tensor(gamma, dtype=torch.float32)
        gamma_tensor = self.device_manager.to_device(gamma_tensor)

        if self.learnable:
            self.gamma = nn.Parameter(gamma_tensor)
        else:
            self.register_buffer('gamma', gamma_tensor)

        if alpha is not None:
            alpha_tensor = self._prepare_alpha_tensor(alpha)
            if self.learnable:
                self.alpha = nn.Parameter(alpha_tensor)
            else:
                self.register_buffer('alpha', alpha_tensor)
        else:
            self.alpha = None

    def _prepare_alpha_tensor(self, alpha: float) -> torch.Tensor:
        """Prepare alpha tensor with proper shape"""
        # Create tensor on CPU first, then move to device
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        alpha_tensor = self.device_manager.to_device(alpha_tensor)
        if alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.expand(self.num_classes)
        return alpha_tensor

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Focal loss value

        Raises:
            LossComputationError: If computation fails
        """
        try:
            # Ensure inputs are on correct device and validate shapes
            inputs = self.device_manager.ensure_device(inputs)
            targets = self.device_manager.ensure_device(targets).long()

            if inputs.size(0) != targets.size(0):
                raise ValueError(f"Batch size mismatch: inputs {inputs.size(0)} vs targets {targets.size(0)}")

            # Compute cross entropy loss
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)

            # Apply focal weight
            gamma_clamped = self.gamma.clamp(min=0)
            focal_weight = (1 - pt) ** gamma_clamped

            # Apply class balancing if alpha is provided
            if self.alpha is not None:
                alpha_clamped = self.alpha.clamp(0, 1)
                alpha_t = alpha_clamped[targets]
                focal_weight = focal_weight * alpha_t

            loss = focal_weight * ce_loss
            return loss.mean()

        except Exception as e:
            raise LossComputationError(f"Failed to compute focal loss: {e}")

    def extra_repr(self) -> str:
        """String representation of the module"""
        gamma_val = self.gamma.item() if isinstance(self.gamma, torch.Tensor) else self.gamma
        alpha_str = ""
        if self.alpha is not None:
            alpha_vals = self.alpha.detach().cpu().numpy()
            alpha_str = f", alpha={alpha_vals}"
        return f"gamma={gamma_val:.4f}{alpha_str}, learnable={self.learnable}, num_classes={self.num_classes}"

class WeightedHeatmapLoss(nn.Module):
    """
    Improved weighted heatmap loss for keypoint detection.

    This loss applies different weights to keypoint regions vs background regions,
    helping the model focus more on accurate keypoint localization.
    """

    def __init__(self, use_target_weight: bool = True, keypoint_weight: float = 10.0,
                 background_weight: float = 1.0, threshold: float = 0.1,
                 device: Optional[torch.device] = None):
        """
        Initialize weighted heatmap loss.

        Args:
            use_target_weight: Whether to apply target-specific weights
            keypoint_weight: Weight for keypoint regions (higher values focus more on keypoints)
            background_weight: Weight for background regions
            threshold: Threshold to distinguish keypoint vs background regions
            device: Device to place tensors on
        """
        super().__init__()

        # Validate parameters
        if keypoint_weight < 0 or background_weight < 0:
            raise ValueError("Weights must be non-negative")
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        self.use_target_weight = use_target_weight
        self.keypoint_weight = keypoint_weight
        self.background_weight = background_weight
        self.threshold = threshold
        self.device_manager = DeviceManager(device)
        self.criterion = nn.MSELoss(reduction='none')

    def _create_weight_map(self, gt_heatmaps: torch.Tensor) -> torch.Tensor:
        """Create weight map based on keypoint vs background regions"""
        keypoint_mask = (gt_heatmaps > self.threshold).float()
        background_mask = (gt_heatmaps <= self.threshold).float()

        weight_map = (keypoint_mask * self.keypoint_weight +
                     background_mask * self.background_weight)
        return weight_map

    def _apply_target_weights(self, loss: torch.Tensor,
                            target_weight: torch.Tensor) -> torch.Tensor:
        """Apply target-specific weights to loss"""
        # Expand target weight to match loss dimensions
        target_weight_expanded = target_weight.view(
            loss.shape[0], loss.shape[1], 1, 1
        )
        return loss * target_weight_expanded

    def forward(self, pred_heatmaps: torch.Tensor,
                gt_heatmaps: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted heatmap loss.

        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W)
            gt_heatmaps: Ground truth heatmaps (B, K, H, W)
            target_weight: Target weights (B, K) or None

        Returns:
            Scalar loss value

        Raises:
            LossComputationError: If computation fails
        """
        try:
            # Ensure tensors are on correct device
            pred_heatmaps = self.device_manager.ensure_device(pred_heatmaps)
            gt_heatmaps = self.device_manager.ensure_device(gt_heatmaps)

            # Validate shapes
            if pred_heatmaps.shape != gt_heatmaps.shape:
                raise ValueError(f"Shape mismatch: {pred_heatmaps.shape} vs {gt_heatmaps.shape}")

            # Compute base MSE loss
            loss = self.criterion(pred_heatmaps, gt_heatmaps)

            # Apply spatial weighting (keypoint vs background)
            weight_map = self._create_weight_map(gt_heatmaps)
            loss = loss * weight_map

            # Apply target weights if provided
            if self.use_target_weight and target_weight is not None:
                target_weight = self.device_manager.ensure_device(target_weight)
                loss = self._apply_target_weights(loss, target_weight)

            return loss.mean()

        except Exception as e:
            raise LossComputationError(f"Failed to compute weighted heatmap loss: {e}")

    def extra_repr(self) -> str:
        """String representation of the module"""
        return (f"keypoint_weight={self.keypoint_weight}, "
                f"background_weight={self.background_weight}, "
                f"threshold={self.threshold}, "
                f"use_target_weight={self.use_target_weight}")

class HeatmapLoss(nn.Module):
    """
    Legacy heatmap loss for backward compatibility.

    Simple MSE loss between predicted and ground truth heatmaps with optional target weighting.
    """

    def __init__(self, use_target_weight: bool = True, device: Optional[torch.device] = None):
        """
        Initialize heatmap loss.

        Args:
            use_target_weight: Whether to apply target-specific weights
            device: Device to place tensors on
        """
        super().__init__()
        self.use_target_weight = use_target_weight
        self.device_manager = DeviceManager(device)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred_heatmaps: torch.Tensor,
                gt_heatmaps: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute heatmap loss.

        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W)
            gt_heatmaps: Ground truth heatmaps (B, K, H, W)
            target_weight: Target weights (B, K) or None

        Returns:
            Scalar loss value

        Raises:
            LossComputationError: If computation fails
        """
        try:
            # Ensure tensors are on correct device
            pred_heatmaps = self.device_manager.ensure_device(pred_heatmaps)
            gt_heatmaps = self.device_manager.ensure_device(gt_heatmaps)

            # Validate shapes
            if pred_heatmaps.shape != gt_heatmaps.shape:
                raise ValueError(f"Shape mismatch: {pred_heatmaps.shape} vs {gt_heatmaps.shape}")

            loss = self.criterion(pred_heatmaps, gt_heatmaps)

            if self.use_target_weight:
                if target_weight is None:
                    raise ValueError("target_weight is required when use_target_weight is True")
                target_weight = self.device_manager.ensure_device(target_weight)
                weight = target_weight.view(loss.shape[0], loss.shape[1], 1, 1)
                loss = loss * weight

            return loss.mean()

        except Exception as e:
            raise LossComputationError(f"Failed to compute heatmap loss: {e}")

    def extra_repr(self) -> str:
        """String representation of the module"""
        return f"use_target_weight={self.use_target_weight}"


class LossComponentFactory:
    """Factory for creating loss components with proper configuration"""

    @staticmethod
    def create_heatmap_loss(config: TrainingConfig, device: Optional[torch.device] = None) -> nn.Module:
        """Create appropriate heatmap loss based on configuration"""
        try:
            weighted_enabled, weighted_config = ConfigurationHandler.extract_weighted_loss_config(config)

            if weighted_enabled and weighted_config:
                return WeightedHeatmapLoss(
                    use_target_weight=True,
                    keypoint_weight=weighted_config['keypoint_weight'],
                    background_weight=weighted_config['background_weight'],
                    threshold=weighted_config['threshold'],
                    device=device
                )
            else:
                return HeatmapLoss(use_target_weight=True, device=device)

        except Exception as e:
            raise LossConfigurationError(f"Failed to create heatmap loss: {e}")

    @staticmethod
    def create_focal_loss(config: TrainingConfig, device: Optional[torch.device] = None) -> FocalLoss:
        """Create focal loss based on configuration"""
        try:
            focal_config = ConfigurationHandler.extract_focal_loss_config(config)
            return FocalLoss(
                gamma=focal_config['gamma'],
                alpha=focal_config['alpha'],
                learnable=focal_config['learnable'],
                num_classes=focal_config['num_classes'],
                device=device
            )
        except Exception as e:
            raise LossConfigurationError(f"Failed to create focal loss: {e}")

class KeypointLoss(nn.Module):
    """
    Comprehensive keypoint detection loss combining multiple loss components.

    This loss combines:
    - Heatmap loss (MSE or weighted MSE)
    - Visibility classification loss (Focal Loss)
    - Coordinate regression loss (Smooth L1)
    """

    def __init__(self, num_keypoints: int, config: TrainingConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize keypoint loss.

        Args:
            num_keypoints: Number of keypoints to detect
            config: Training configuration
            device: Device to place tensors on

        Raises:
            LossConfigurationError: If configuration is invalid
        """
        super().__init__()

        # Validate inputs
        if num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")
        if config is None:
            raise ValueError("config cannot be None")

        self.num_keypoints = num_keypoints
        self.config = config
        self.device_manager = DeviceManager(device)

        # Initialize loss weights and scaling
        self.loss_weights = LossWeights(
            heatmap=getattr(config, 'lambda_keypoint', 15.0),
            visibility=getattr(config, 'lambda_visibility', 5.0),
            coordinate=1.0  # Will be scaled in forward pass
        )

        self.loss_scaling = LossScaling()

        # Initialize loss components using factory
        self._initialize_loss_components()

        # Initialize keypoint importance weights
        self.keypoint_weights = self._init_keypoint_weights(num_keypoints)
        self.keypoint_weights = self.device_manager.to_device(self.keypoint_weights)

    def _initialize_loss_components(self):
        """Initialize all loss components"""
        try:
            # Create heatmap loss (weighted or regular based on config)
            self.heatmap_criterion = LossComponentFactory.create_heatmap_loss(
                self.config, self.device_manager.device
            )

            # Create visibility loss (focal loss)
            self.visibility_criterion = LossComponentFactory.create_focal_loss(
                self.config, self.device_manager.device
            )

            # Create coordinate loss
            self.coordinate_criterion = nn.SmoothL1Loss(reduction='none')

        except Exception as e:
            raise LossConfigurationError(f"Failed to initialize loss components: {e}")

    def _validate_inputs(self, predictions: Dict[str, torch.Tensor],
                        targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and extract basic inputs"""
        if 'heatmaps' not in predictions:
            raise ValueError("predictions must contain 'heatmaps'")
        if 'heatmaps' not in targets:
            raise ValueError("targets must contain 'heatmaps'")

        pred_heatmaps = predictions['heatmaps']
        gt_heatmaps = targets['heatmaps']

        if pred_heatmaps.shape != gt_heatmaps.shape:
            raise ValueError(f"Shape mismatch: predictions {pred_heatmaps.shape} vs targets {gt_heatmaps.shape}")

        return pred_heatmaps, gt_heatmaps

    def _compute_target_weights(self, targets: Dict[str, torch.Tensor],
                               batch_size: int) -> torch.Tensor:
        """Compute target weights for heatmap loss"""
        if 'visibility' in targets:
            visibility = self.device_manager.ensure_device(targets['visibility'])

            # Handle multi-person case: [B, P, K] -> [B, K]
            if visibility.dim() == 3:
                visibility = visibility.max(dim=1)[0]

            # Convert 3-class visibility to binary weights (visible if class 1 or 2)
            binary_visibility = (visibility > 0).float()
            weights = self.keypoint_weights.view(1, -1) * binary_visibility
        else:
            # Use default keypoint weights for all samples
            weights = self.keypoint_weights.view(1, -1).expand(batch_size, -1)

        return weights.clone()

    def _compute_heatmap_loss(self, pred_heatmaps: torch.Tensor,
                             gt_heatmaps: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
        """Compute heatmap loss component"""
        try:
            return self.heatmap_criterion(pred_heatmaps, gt_heatmaps, weights)
        except Exception as e:
            raise LossComputationError(f"Failed to compute heatmap loss: {e}")

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined keypoint detection loss.

        Args:
            predictions: Dictionary containing:
                - heatmaps: (B, K, H, W) Predicted heatmaps
                - visibilities: (B, K, 3) or (B, P, K, 3) Predicted visibility probabilities (optional)
                - coordinates: (B, K, 2) or (B, P, K, 2) Predicted coordinates (optional)
            targets: Dictionary containing:
                - heatmaps: (B, K, H, W) Ground truth heatmaps
                - visibility: (B, K) or (B, P, K) Visibility flags (optional)
                - keypoints: (B, K, 2) or (B, P, K, 2) Ground truth coordinates (optional)

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Combined weighted loss value
            - loss_dict: Dictionary containing individual loss components

        Raises:
            LossComputationError: If loss computation fails
        """
        try:
            # Validate inputs and extract basic tensors
            pred_heatmaps, gt_heatmaps = self._validate_inputs(predictions, targets)
            batch_size = pred_heatmaps.size(0)

            # Compute target weights for heatmap loss
            weights = self._compute_target_weights(targets, batch_size)

            # Compute heatmap loss
            heatmap_loss = self._compute_heatmap_loss(pred_heatmaps, gt_heatmaps, weights)

            # Compute visibility loss
            visibility_loss = self._compute_visibility_loss(predictions, targets)

            # Compute coordinate loss
            coordinate_loss = self._compute_coordinate_loss(predictions, targets)

            # Combine losses with proper scaling
            total_loss = self._combine_losses(heatmap_loss, visibility_loss, coordinate_loss)

            # Create loss dictionary for monitoring
            loss_dict = self._create_loss_dict(heatmap_loss, visibility_loss, coordinate_loss, total_loss)

            return total_loss, loss_dict

        except Exception as e:
            raise LossComputationError(f"Failed to compute keypoint loss: {e}")

    def _compute_visibility_loss(self, predictions: Dict[str, torch.Tensor],
                                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute visibility classification loss"""
        if 'visibilities' not in predictions or 'visibility' not in targets:
            return torch.tensor(0.0, device=self.device_manager.device)

        try:
            pred_vis = self.device_manager.ensure_device(predictions['visibilities'])
            target_vis = self.device_manager.ensure_device(targets['visibility'])

            # Reshape tensors to handle multi-person scenarios
            pred_vis = self._reshape_visibility_predictions(pred_vis)
            target_vis = self._reshape_visibility_targets(target_vis)

            # Ensure target_vis is long type and within valid range
            target_vis = target_vis.long().clamp(0, 2)

            if pred_vis.size(0) == target_vis.size(0):
                return self.visibility_criterion(pred_vis, target_vis)
            else:
                return torch.tensor(0.0, device=self.device_manager.device)

        except Exception as e:
            # Log warning but don't fail the entire loss computation
            print(f"Warning: Could not compute visibility loss: {e}")
            return torch.tensor(0.0, device=self.device_manager.device)

    def _reshape_visibility_predictions(self, pred_vis: torch.Tensor) -> torch.Tensor:
        """Reshape visibility predictions to [N, 3] format"""
        if pred_vis.dim() == 4:  # [B, P, K, 3]
            return pred_vis.view(-1, pred_vis.size(-1))
        elif pred_vis.dim() == 3:  # [B, K, 3] or [B, P, K]
            if pred_vis.size(-1) == 3:
                return pred_vis.view(-1, 3)
            else:  # [B, P, K]
                return pred_vis.view(-1)
        return pred_vis

    def _reshape_visibility_targets(self, target_vis: torch.Tensor) -> torch.Tensor:
        """Reshape visibility targets to [N] format"""
        if target_vis.dim() >= 2:
            return target_vis.view(-1)
        return target_vis

    def _compute_coordinate_loss(self, predictions: Dict[str, torch.Tensor],
                                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute coordinate regression loss"""
        if 'coordinates' not in predictions or 'keypoints' not in targets:
            return torch.tensor(0.0, device=self.device_manager.device)

        try:
            pred_coords = self.device_manager.ensure_device(predictions['coordinates'])
            target_coords = self.device_manager.ensure_device(targets['keypoints'])

            # Reshape tensors to handle multi-person scenarios
            pred_coords = self._reshape_coordinates(pred_coords)
            target_coords = self._reshape_coordinates(target_coords)

            if pred_coords.size(0) != target_coords.size(0):
                return torch.tensor(0.0, device=self.device_manager.device)

            # Compute loss only for visible keypoints
            if 'visibility' in targets:
                vis_mask = self._get_visibility_mask(targets)
                visible_mask = (vis_mask > 0).float()

                if visible_mask.sum() > 0:
                    coord_loss = self.coordinate_criterion(pred_coords, target_coords)
                    coord_loss = coord_loss.mean(dim=1)  # Average over x,y coordinates
                    return (coord_loss * visible_mask).sum() / visible_mask.sum()
                else:
                    return torch.tensor(0.0, device=self.device_manager.device)
            else:
                return self.coordinate_criterion(pred_coords, target_coords).mean()

        except Exception as e:
            print(f"Warning: Could not compute coordinate loss: {e}")
            return torch.tensor(0.0, device=self.device_manager.device)

    def _reshape_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """Reshape coordinate tensors to [N, 2] format"""
        if coords.dim() == 4:  # [B, P, K, 2]
            return coords.view(-1, 2)
        elif coords.dim() == 3:  # [B, K, 2]
            return coords.view(-1, 2)
        return coords

    def _get_visibility_mask(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get visibility mask for coordinate loss computation"""
        vis_mask = self.device_manager.ensure_device(targets['visibility'])
        if vis_mask.dim() >= 2:
            return vis_mask.view(-1)
        return vis_mask

    def _combine_losses(self, heatmap_loss: torch.Tensor,
                       visibility_loss: torch.Tensor,
                       coordinate_loss: torch.Tensor) -> torch.Tensor:
        """Combine individual loss components with proper weighting and scaling"""
        # Apply scaling factors
        coordinate_loss_scaled = coordinate_loss * self.loss_scaling.coordinate_scale
        visibility_loss_scaled = visibility_loss * self.loss_scaling.visibility_scale

        # Combine with configured weights
        total_loss = (
            self.loss_weights.heatmap * heatmap_loss +
            self.loss_weights.visibility * visibility_loss_scaled +
            self.loss_weights.coordinate * coordinate_loss_scaled
        )

        return total_loss

    def _create_loss_dict(self, heatmap_loss: torch.Tensor,
                         visibility_loss: torch.Tensor,
                         coordinate_loss: torch.Tensor,
                         total_loss: torch.Tensor) -> Dict[str, float]:
        """Create dictionary of loss components for monitoring"""
        # Apply scaling for reporting
        coordinate_loss_scaled = coordinate_loss * self.loss_scaling.coordinate_scale
        visibility_loss_scaled = visibility_loss * self.loss_scaling.visibility_scale

        return {
            'keypoint_loss': heatmap_loss.detach().item(),
            'heatmap_loss': heatmap_loss.detach().item(),  # Backward compatibility
            'visibility_loss': visibility_loss.detach().item(),
            'visibility_loss_scaled': visibility_loss_scaled.detach().item(),
            'coordinate_loss': coordinate_loss.detach().item(),
            'coordinate_loss_scaled': coordinate_loss_scaled.detach().item(),
            'total_loss': total_loss.detach().item()
        }

    def _init_keypoint_weights(self, num_keypoints: int) -> torch.Tensor:
        """
        Initialize keypoint importance weights with specific emphasis on key joints.

        Args:
            num_keypoints: Number of keypoints

        Returns:
            Tensor of keypoint weights
        """
        if num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")

        weights = torch.ones(num_keypoints, dtype=torch.float32)

        # Define importance weights for specific keypoints (COCO format)
        # Higher weights for more important/reliable keypoints
        keypoint_importance = {
            0: 1.5,   # nose - important for face orientation
            1: 1.5,   # neck - central body reference
            2: 1.3,   # right eye
            3: 1.3,   # left eye
            4: 1.2,   # right ear
            5: 1.2,   # left ear
            6: 1.4,   # right shoulder
            7: 1.4,   # left shoulder
            8: 1.3,   # right elbow
            9: 1.3,   # left elbow
            10: 1.2,  # right wrist
            11: 1.2,  # left wrist
            12: 1.4,  # right hip - important for pose
            13: 1.4,  # left hip - important for pose
            14: 1.3,  # right knee
            15: 1.3,  # left knee
            16: 1.3,  # right ankle - important for stance
            17: 1.3   # left ankle - important for stance
        }

        # Apply weights only for existing keypoints
        for idx, weight in keypoint_importance.items():
            if idx < num_keypoints:
                weights[idx] = weight

        return weights

    def extra_repr(self) -> str:
        """String representation of the module"""
        return (f"num_keypoints={self.num_keypoints}, "
                f"heatmap_weight={self.loss_weights.heatmap}, "
                f"visibility_weight={self.loss_weights.visibility}, "
                f"coordinate_weight={self.loss_weights.coordinate}")


# Backward compatibility aliases
KeypointLossV1 = KeypointLoss  # For any code that might reference the old version

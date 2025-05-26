"""
Training configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .base_config import BaseConfig, DeviceConfig

@dataclass
class OptimizerConfig(BaseConfig):
    name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9  # for SGD
    beta1: float = 0.9     # for Adam
    beta2: float = 0.999   # for Adam

    def validate(self):
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.momentum <= 1, "momentum must be between 0 and 1"
        assert 0 <= self.beta1 <= 1, "beta1 must be between 0 and 1"
        assert 0 <= self.beta2 <= 1, "beta2 must be between 0 and 1"

@dataclass
class AugmentationConfig(BaseConfig):
    enabled: bool = True
    prob: float = 0.5
    flip: Dict[str, bool] = field(default_factory=lambda: {
        "enabled": True,
        "horizontal": True
    })
    rotate: Dict[str, Union[bool, float]] = field(default_factory=lambda: {
        "enabled": True,
        "max_angle": 30.0
    })
    scale: Dict[str, Union[bool, List[float]]] = field(default_factory=lambda: {
        "enabled": True,
        "range": [0.8, 1.2]
    })

    def validate(self):
        assert 0 <= self.prob <= 1, "prob must be between 0 and 1"

@dataclass
class LossConfig(BaseConfig):
    keypoint_loss_weight: float = 1.0
    visibility_loss_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    learnable_focal_params: bool = False
    label_smoothing: float = 0.05

    def validate(self):
        assert self.keypoint_loss_weight >= 0, "keypoint_loss_weight must be non-negative"
        assert self.visibility_loss_weight >= 0, "visibility_loss_weight must be non-negative"
        assert self.focal_gamma >= 0, "focal_gamma must be non-negative"
        if self.focal_alpha is not None:
            assert 0 <= self.focal_alpha <= 1, "focal_alpha must be between 0 and 1"
        assert 0 <= self.label_smoothing <= 1, "label_smoothing must be between 0 and 1"

@dataclass
class TrainingConfig(BaseConfig):
    num_epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    checkpoint_interval: int = 5
    validation_interval: int = 1

    # Learning rate scheduler parameters
    lr_factor: float = 0.1  # Factor to reduce learning rate
    patience: int = 3  # Number of epochs with no improvement after which learning rate will be reduced
    min_lr: float = 1e-6  # Minimum learning rate

    # Loss weights
    lambda_keypoint: float = 15.0
    lambda_visibility: float = 5.0
    l2_lambda: float = 0.0003

    # Metrics
    default_validation_threshold: float = 0.5
    pck_thresholds: List[float] = field(default_factory=lambda: [0.002, 0.05, 0.2])

    def validate(self):
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.lr_factor > 0 and self.lr_factor < 1, "lr_factor must be between 0 and 1"
        assert self.patience >= 0, "patience must be non-negative"
        assert self.min_lr > 0, "min_lr must be positive"
        assert all(0 <= x <= 1 for x in self.pck_thresholds), "pck_thresholds must be between 0 and 1"
        assert all(self.pck_thresholds[i] < self.pck_thresholds[i+1] for i in range(len(self.pck_thresholds)-1)), "pck_thresholds must be strictly increasing"
        self.optimizer.validate()
        self.augmentation.validate()
        self.loss.validate()

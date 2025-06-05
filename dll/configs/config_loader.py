"""
Configuration loader for YAML configs
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field

from .base_config import DeviceConfig
from .model_config import ModelConfig, BackboneConfig, HeatmapHeadConfig, KeypointHeadConfig, PersonDetectionConfig
from .training_config import TrainingConfig, OptimizerConfig, AugmentationConfig, LossConfig, WeightedLossConfig, LRSchedulerConfig

@dataclass
class PathsConfig:
    default_config: str = ""
    data_dir: str = ""
    output_dir: str = ""

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Parse paths
    paths_data = yaml_data.get('paths', {})
    paths = PathsConfig(
        default_config=paths_data.get('default_config', ''),
        data_dir=paths_data.get('data_dir', ''),
        output_dir=paths_data.get('output_dir', '')
    )

    # Parse device config
    device_data = yaml_data.get('device', {})
    device = DeviceConfig(
        type=device_data.get('type', 'auto'),
        force_cpu=device_data.get('force_cpu', False),
        mixed_precision=device_data.get('mixed_precision', True),
        pin_memory=device_data.get('pin_memory', True)
    )

    # Parse model config
    model_data = yaml_data.get('model', {})

    # Backbone config
    backbone_data = model_data.get('backbone', {})
    backbone = BackboneConfig(
        width_mult=backbone_data.get('width_mult', 1.0),
        in_channels=backbone_data.get('in_channels', 3),
        out_channels=backbone_data.get('out_channels', 128),
        input_size=backbone_data.get('input_size', 224),
        convert_to_grayscale=backbone_data.get('convert_to_grayscale', True)
    )

    # Person head config
    person_head_data = model_data.get('person_head', {})
    person_head = PersonDetectionConfig(
        in_channels=person_head_data.get('in_channels', 128),
        num_classes=person_head_data.get('num_classes', 1),
        conf_threshold=person_head_data.get('conf_threshold', 0.3),
        nms_iou_threshold=person_head_data.get('nms_iou_threshold', 0.3),
        anchor_sizes=person_head_data.get('anchor_sizes', [32, 64, 128])
    )

    # Keypoint head config
    keypoint_head_data = model_data.get('keypoint_head', {})
    keypoint_head = KeypointHeadConfig(
        in_channels=keypoint_head_data.get('in_channels', 128),
        num_keypoints=keypoint_head_data.get('num_keypoints', 17),
        height=keypoint_head_data.get('height', 56),
        width=keypoint_head_data.get('width', 56),
        fine_branch_channels=keypoint_head_data.get('fine_branch_channels', 64),
        regression_channels=keypoint_head_data.get('regression_channels', 32),
        visibility_channels=keypoint_head_data.get('visibility_channels', 32),
        dropout_rate=keypoint_head_data.get('dropout_rate', 0.2)
    )

    # Heatmap head config
    heatmap_head_data = model_data.get('heatmap_head', {})
    heatmap_head = HeatmapHeadConfig(
        in_channels=heatmap_head_data.get('in_channels', 64),
        hidden_channels=heatmap_head_data.get('hidden_channels', 64),
        num_keypoints=heatmap_head_data.get('num_keypoints', 17),
        heatmap_size=tuple(heatmap_head_data.get('heatmap_size', [56, 56])),
        dropout_rate=heatmap_head_data.get('dropout_rate', 0.1),
        use_attention=heatmap_head_data.get('use_attention', True),
        num_deconv_layers=heatmap_head_data.get('num_deconv_layers', 2),
        deconv_kernel_sizes=heatmap_head_data.get('deconv_kernel_sizes', [4, 4]),
        deconv_channels=heatmap_head_data.get('deconv_channels', [256, 256])
    )

    model = ModelConfig(
        backbone=backbone,
        person_head=person_head,
        keypoint_head=keypoint_head,
        heatmap_head=heatmap_head,
        num_keypoints=keypoint_head_data.get('num_keypoints', 17)
    )

    # Parse training config
    training_data = yaml_data.get('training', {})

    # Optimizer config
    optimizer_data = training_data.get('optimizer', {})
    optimizer = OptimizerConfig(
        name=optimizer_data.get('name', 'adam'),
        learning_rate=optimizer_data.get('learning_rate', 0.001),
        weight_decay=optimizer_data.get('weight_decay', 0.0001),
        momentum=optimizer_data.get('momentum', 0.9),
        beta1=optimizer_data.get('beta1', 0.9),
        beta2=optimizer_data.get('beta2', 0.999)
    )

    # Augmentation config
    augmentation_data = training_data.get('augmentation', {})
    augmentation = AugmentationConfig(
        enabled=augmentation_data.get('enabled', True),
        prob=augmentation_data.get('prob', 0.5),
        flip=augmentation_data.get('flip', {
            "enabled": True,
            "horizontal": True
        }),
        rotate=augmentation_data.get('rotate', {
            "enabled": True,
            "max_angle": 30.0
        }),
        scale=augmentation_data.get('scale', {
            "enabled": True,
            "range": [0.8, 1.2]
        })
    )

    # Loss config
    loss_data = training_data.get('loss', {})

    # Weighted loss config
    weighted_loss_data = loss_data.get('weighted_loss', {})
    weighted_loss = WeightedLossConfig(
        enabled=weighted_loss_data.get('enabled', True),
        keypoint_weight=weighted_loss_data.get('keypoint_weight', 15.0),
        background_weight=weighted_loss_data.get('background_weight', 1.0),
        threshold=weighted_loss_data.get('threshold', 0.1)
    )

    loss = LossConfig(
        keypoint_loss_weight=loss_data.get('keypoint_loss_weight', 1.0),
        visibility_loss_weight=loss_data.get('visibility_loss_weight', 1.0),
        focal_gamma=loss_data.get('focal_gamma', 2.0),
        focal_alpha=loss_data.get('focal_alpha'),
        learnable_focal_params=loss_data.get('learnable_focal_params', False),
        label_smoothing=loss_data.get('label_smoothing', 0.05),
        weighted_loss=weighted_loss
    )

    # LR Scheduler config
    lr_scheduler_data = training_data.get('lr_scheduler', {})
    lr_scheduler = LRSchedulerConfig(
        factor=lr_scheduler_data.get('factor', 0.1),
        patience=lr_scheduler_data.get('patience', 3),
        min_lr=lr_scheduler_data.get('min_lr', 1e-6),
        mode=lr_scheduler_data.get('mode', 'min'),
        threshold=lr_scheduler_data.get('threshold', 0.0001),
        metric=lr_scheduler_data.get('metric', 'loss')
    )

    training = TrainingConfig(
        num_epochs=training_data.get('num_epochs', 50),
        batch_size=training_data.get('batch_size', 32),
        num_workers=training_data.get('num_workers', 4),
        optimizer=optimizer,
        augmentation=augmentation,
        loss=loss,
        lr_scheduler=lr_scheduler,
        device=device,
        checkpoint_interval=training_data.get('checkpoint_interval', 5),
        validation_interval=training_data.get('validation_interval', 1),
        lr_factor=training_data.get('lr_factor', 0.1),
        patience=training_data.get('patience', 3),
        min_lr=training_data.get('min_lr', 1e-6),
        lambda_keypoint=training_data.get('lambda_keypoint', 15.0),
        lambda_visibility=training_data.get('lambda_visibility', 5.0),
        l2_lambda=training_data.get('l2_lambda', 0.0003),
        default_validation_threshold=training_data.get('default_validation_threshold', 0.5),
        pck_thresholds=training_data.get('pck_thresholds', [0.002, 0.05, 0.2])
    )

    return Config(
        paths=paths,
        device=device,
        model=model,
        training=training
    )

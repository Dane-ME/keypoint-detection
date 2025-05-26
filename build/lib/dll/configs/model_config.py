"""
Model configuration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml
from .base_config import BaseConfig

@dataclass
class BackboneConfig(BaseConfig):
    width_mult: float = 1.0
    in_channels: int = 3  # Default to RGB
    out_channels: int = 128
    input_size: int = 224  # Default input size
    convert_to_grayscale: bool = False  # Option to convert RGB to grayscale
    
    def validate(self):
        assert self.width_mult > 0, "width_mult must be positive"
        assert self.in_channels in [1, 3], "in_channels must be 1 (grayscale) or 3 (RGB)"
        assert self.out_channels > 0, "out_channels must be positive"
        assert self.input_size > 0 and self.input_size % 32 == 0, "input_size must be positive and divisible by 32"

@dataclass
class PersonDetectionConfig(BaseConfig):
    in_channels: int = 128
    num_classes: int = 1
    conf_threshold: float = 0.3
    nms_iou_threshold: float = 0.3
    anchor_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    
    def validate(self):
        assert 0 <= self.conf_threshold <= 1, "conf_threshold must be between 0 and 1"
        assert 0 <= self.nms_iou_threshold <= 1, "nms_iou_threshold must be between 0 and 1"

@dataclass
class KeypointHeadConfig(BaseConfig):
    in_channels: int = 128
    num_keypoints: int = 17
    height: int = 32
    width: int = 32
    fine_branch_channels: int = 64
    regression_channels: int = 32
    visibility_channels: int = 32
    dropout_rate: float = 0.2
    
    def validate(self):
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.num_keypoints > 0, "num_keypoints must be positive"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be between 0 and 1"

@dataclass
class HeatmapHeadConfig(BaseConfig):
    in_channels: int = 64
    hidden_channels: int = 64
    num_keypoints: int = 17
    heatmap_size: Tuple[int, int] = (224, 224)
    dropout_rate: float = 0.1
    use_attention: bool = True
    num_deconv_layers: int = 2
    deconv_kernel_sizes: Tuple[int, ...] = (4, 4)
    deconv_channels: Tuple[int, ...] = (256, 256)
    
    def validate(self):
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.hidden_channels > 0, "hidden_channels must be positive"
        assert self.num_keypoints > 0, "num_keypoints must be positive"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be between 0 and 1"
        assert len(self.deconv_kernel_sizes) == self.num_deconv_layers, "Number of deconv kernel sizes must match num_deconv_layers"
        assert len(self.deconv_channels) == self.num_deconv_layers, "Number of deconv channels must match num_deconv_layers"

@dataclass
class ModelConfig(BaseConfig):
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    person_head: PersonDetectionConfig = field(default_factory=PersonDetectionConfig)
    keypoint_head: KeypointHeadConfig = field(default_factory=KeypointHeadConfig)
    heatmap_head: HeatmapHeadConfig = field(default_factory=HeatmapHeadConfig)
    num_keypoints: int = 17
    
    def validate(self):
        super().validate()
        # Validate compatibility between components
        assert self.backbone.out_channels == self.person_head.in_channels, \
            "Backbone output channels must match person head input channels"
        assert self.person_head.in_channels == self.keypoint_head.in_channels, \
            "Person head channels must match keypoint head input channels"
        assert self.num_keypoints == self.keypoint_head.num_keypoints, \
            "Number of keypoints must be consistent"
        assert self.num_keypoints == self.heatmap_head.num_keypoints, \
            "Number of keypoints must be consistent with heatmap head"


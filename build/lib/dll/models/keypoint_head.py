import torch
import torch.nn as nn
import torch.nn.functional as F
from dll.configs import KeypointHeadConfig, BackboneConfig
from dll.models.backbone import BACKBONE as BACKBONE
import yaml


class KEYPOINT_HEAD(nn.Module):
    def __init__(self, config: KeypointHeadConfig):
        super(KEYPOINT_HEAD, self).__init__()
        self.num_keypoints = config.num_keypoints
        in_channels = config.in_channels
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.regression_branch = nn.Sequential(
            ResidualBlock(in_channels, config.fine_branch_channels),  
            ResidualBlock(config.fine_branch_channels, config.regression_channels),
            nn.Conv2d(config.regression_channels, config.regression_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.regression_channels // 2),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((config.height//4, config.width//4)),
            nn.Flatten(),
            nn.Linear(config.regression_channels // 2 * (config.height//4) * (config.width//4), 256),
            nn.LayerNorm(256),
            nn.ReLU6(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, self.num_keypoints * 2)
        )
        
        self.visibility_branch = nn.Sequential(
            nn.Conv2d(in_channels, config.visibility_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.visibility_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(config.visibility_channels * 16, 128),
            nn.LayerNorm(128),
            nn.ReLU6(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, self.num_keypoints * 3)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        attention = self.spatial_attention(x)
        x = x * attention
        
        keypoint_offsets = self.regression_branch(x)
        keypoints = torch.sigmoid(keypoint_offsets).view(batch_size, self.num_keypoints, 2)

        visibility_logits = self.visibility_branch(x)
        visibility = torch.sigmoid(visibility_logits).view(batch_size, self.num_keypoints, 3)
        
        return keypoints, visibility

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x       
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)       
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu6(out)
        return out


def main():
    backbone_config = BackboneConfig.from_default('model.backbone')
    keypoint_config = KeypointHeadConfig.from_default('model.keypoint_head')

    x = torch.randn(1, 128, 28, 28)
    y = torch.randn(1, 1, 224, 224)
    backbone = BACKBONE(backbone_config)
    model = KEYPOINT_HEAD(keypoint_config)
    backbone_outputs = backbone(y)
    print(backbone_outputs[0].shape)
    keypoints, visibility = model(backbone_outputs[0])
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Visibilities shape: {visibility.shape}")

if __name__ == '__main__':
    main()

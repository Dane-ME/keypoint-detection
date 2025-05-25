import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from dll.configs.model_config import BackboneConfig
from torchvision.models.feature_extraction import create_feature_extractor

class LightweightFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(LightweightFPN, self).__init__()
        
        # Validate input channels
        if not isinstance(in_channels_list, list):
            raise ValueError("in_channels_list must be a list of input channel sizes")
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, stride=1, bias=False) 
            for in_ch in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])

    def forward(self, features):
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"Expected {len(self.lateral_convs)} features, got {len(features)}")
        
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals)-1, 0, -1):  
            upsampled = nn.functional.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            laterals[i-1] += upsampled
                
        return [fpn_conv(f) for fpn_conv, f in zip(self.fpn_convs, laterals)]

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # Tối ưu bằng expand_as thay vì broadcast thủ công

class ConvBNActivation(nn.Module):
    """Standard convolution with BN and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation_layer=nn.Hardswish):
        super(ConvBNActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if activation_layer:
            layers.append(activation_layer(inplace=True))
        self.layer = nn.Sequential(*layers)  # Gộp thành một Sequential để giảm overhead

    def forward(self, x):
        return self.layer(x)

class InvertedResidual(nn.Module):
    """MBConv block: Mobile Inverted Residual Bottleneck"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=None, activation_layer=nn.Hardswish):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        exp_channels = int(in_channels * expand_ratio)
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_channels, exp_channels, kernel_size=1, activation_layer=activation_layer))
        
        layers.append(ConvBNActivation(exp_channels, exp_channels, kernel_size=kernel_size, stride=stride, 
                                       groups=exp_channels, activation_layer=activation_layer))
        
        if se_ratio:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SEModule(exp_channels, reduction=exp_channels // se_channels))
        
        layers.append(ConvBNActivation(exp_channels, out_channels, kernel_size=1, activation_layer=None))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_res_connect else out  # Tối ưu phép cộng residual

class LimbAttentionModule(nn.Module):
    """Module tập trung vào các chi (limbs) như cánh tay và vai"""
    def __init__(self, in_channels):
        super(LimbAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.conv2 = nn.Conv2d(in_channels//2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Tạo attention map
        attn = self.conv1(x)
        attn = nn.functional.relu(self.bn1(attn))
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # Áp dụng attention
        return x * attn.expand_as(x)

class BACKBONE(nn.Module):
    """
    Custom MobileNet V3 backbone
    
    Args:
        config (BackboneConfig): Configuration object containing model parameters
    """
    def __init__(self, config: BackboneConfig):
        super(BACKBONE, self).__init__()
        self.config = config
        
        # Move to specified device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize channels
        input_channel = 16
        last_channel = 1024

        # Initialize FPN
        fpn = LightweightFPN(
            in_channels_list=[16, 40, 96, 576], 
            out_channels=config.out_channels
        )
        self.fpn = fpn.to(device)

        # Channel rounding function
        def _make_divisible(v, divisor=8):
            return max(divisor, ((v + divisor // 2) // divisor) * divisor)
        
        # Apply width multiplier
        input_channel = _make_divisible(input_channel * config.width_mult)
        last_channel = _make_divisible(last_channel * max(1.0, config.width_mult))

        # Model architecture definition
        inverted_residual_setting = [
            # [expand_ratio, out_channels, kernel_size, stride, se_ratio, activation]
            [1, 16, 3, 2, 0.25, nn.ReLU],        # 112 -> 56
            [4.5, 24, 3, 2, None, nn.ReLU],      # 56 -> 28
            [3.67, 24, 3, 1, None, nn.ReLU],     # 28 -> 28
            [4, 40, 5, 2, 0.25, nn.Hardswish],   # 28 -> 14
            [6, 40, 5, 1, 0.25, nn.Hardswish],   # 14 -> 14
            [6, 40, 5, 1, 0.25, nn.Hardswish],   # 14 -> 14
            [3, 48, 5, 1, 0.25, nn.Hardswish],   # 14 -> 14
            [3, 48, 5, 1, 0.25, nn.Hardswish],   # 14 -> 14
            [6, 96, 5, 2, 0.25, nn.Hardswish],   # 14 -> 7
            [6, 96, 5, 1, 0.25, nn.Hardswish],   # 7 -> 7
            [6, 96, 5, 1, 0.25, nn.Hardswish],   # 7 -> 7
        ]
        
        # Initial convolution layer
        layers = [
            ConvBNActivation(
                config.in_channels, 
                input_channel,
                kernel_size=3,
                stride=2,
                activation_layer=nn.Hardswish
            )
        ]
        
        self.selected_layers = [1, 6, 10, 12]  # Layers to extract features from
        self.backbone_outputs = []

        # Build inverted residual blocks
        for i, (expand_ratio, outch, kernel, stride, se_ratio, activation) in enumerate(inverted_residual_setting):
            outch = _make_divisible(outch * config.width_mult)
            layers.append(
                InvertedResidual(
                    input_channel, 
                    outch,
                    kernel,
                    stride,
                    expand_ratio,
                    se_ratio,
                    activation
                )
            )
            input_channel = outch
        
        # Final convolution layer
        self.lastconv_output_channel = _make_divisible(576 * config.width_mult)
        layers.append(
            ConvBNActivation(
                input_channel,
                self.lastconv_output_channel,
                kernel_size=1,
                activation_layer=nn.Hardswish
            )
        )
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._initialize_weights()

        # Thêm limb attention module cho FPN
        self.limb_attention = LimbAttentionModule(config.out_channels)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)   

    def forward(self, x):
        self.backbone_outputs = [] 
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.selected_layers:
                self.backbone_outputs.append(x)
        
        # Lấy output từ FPN
        fpn_outputs = self.fpn(self.backbone_outputs)
        fpn_outputs[0] = self.limb_attention(fpn_outputs[0])
        
        return fpn_outputs

class MobileNetV3Wrapper(nn.Module):
    def __init__(self, config: BackboneConfig, out_channels=128, attention_module=None):   
        super(MobileNetV3Wrapper, self).__init__()
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        if config.in_channels != 3:
            model.features[0][0] = nn.Conv2d(config.in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        return_nodes = {'features.0': 'feat0', 'features.3': 'feat1', 'features.8': 'feat2', 'features.12': 'feat3'}
        self.body = create_feature_extractor(model, return_nodes=return_nodes)        
        self.fpn = LightweightFPN([16, 24, 48, 576], out_channels)
        self.attention_module = attention_module
                
    def forward(self, x):
        feats = self.body(x)
        features = [feats['feat0'], feats['feat1'], feats['feat2'], feats['feat3']]  
        fpn_outs = self.fpn(features) 
        if self.attention_module:  
            fpn_outs[0] = self.attention_module(fpn_outs[0])
        return fpn_outs

def main():
    """Test backbone model"""
    config = BackboneConfig(
        width_mult=1.0,
        in_channels=1,
        out_channels=128,
        input_size=224,
        convert_to_grayscale=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BACKBONE(config).to(device)
    x = torch.randn(1, config.in_channels, config.input_size, config.input_size).to(device)
    
    # Forward pass
    outputs = model(x)
    
    print(f"\nUsing device: {device}")
    print("\nModel outputs:")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class HeatmapHeadConfig:
    in_channels: int = 64
    hidden_channels: int = 64
    num_keypoints: int = 17
    heatmap_size: Tuple[int, int] = (56, 56)
    dropout_rate: float = 0.1
    use_attention: bool = True
    num_deconv_layers: int = 2
    deconv_kernel_sizes: Tuple[int, ...] = (4, 4)
    deconv_channels: Tuple[int, ...] = (256, 256)

class HeatmapHead(nn.Module):
    def __init__(self, config: HeatmapHeadConfig):
        super().__init__()
        self.config = config
        
        if config.use_attention:
            self.channel_attention = ChannelAttention(config.in_channels)
            self.spatial_attention = SpatialAttention()
        
        self.deconv_layers = self._make_deconv_layers()
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                config.deconv_channels[-1],
                config.hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(config.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.hidden_channels,
                config.num_keypoints,
                kernel_size=1
            )
        )
        
        self.init_weights()

    def _make_deconv_layers(self):
        layers = []
        for i in range(self.config.num_deconv_layers):
            in_channels = self.config.in_channels if i == 0 else self.config.deconv_channels[i-1]
            
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels,
                    self.config.deconv_channels[i],
                    kernel_size=self.config.deconv_kernel_sizes[i],
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                nn.BatchNorm2d(self.config.deconv_channels[i]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.config.dropout_rate)
            ])
            
        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the heatmap head
        
        Args:
            x: Input tensor of shape (B, C, H, W) from ROI Align
            
        Returns:
            heatmaps: Predicted heatmaps of shape (B, K, H', W')
            attention_weights: Optional attention weights if use_attention is True
        """
        attention_weights = None
        
        if self.config.use_attention:
            # Apply channel attention
            channel_weights = self.channel_attention(x)
            x = x * channel_weights
            
            # Apply spatial attention
            spatial_weights = self.spatial_attention(x)
            x = x * spatial_weights
            attention_weights = (channel_weights, spatial_weights)
        
        # Apply deconvolution layers
        x = self.deconv_layers(x)
        
        # Generate heatmaps
        heatmaps = self.final_layer(x)
        
        # Apply sigmoid to get probability maps
        heatmaps = torch.sigmoid(heatmaps)
        
        return heatmaps, attention_weights

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention map using average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        
        return torch.sigmoid(x)

def get_gaussian_kernel(size: int = 7, sigma: float = 3) -> torch.Tensor:
    """Generate 2D Gaussian kernel"""
    coords = torch.arange(size).float() - (size - 1) / 2
    coords = coords.view(-1, 1).repeat(1, size)
    
    gaussian = torch.exp(-(coords ** 2 + coords.t() ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    
    return gaussian

def generate_target_heatmap(keypoints: torch.Tensor, 
                          heatmap_size: Tuple[int, int],
                          sigma: float = 2.0) -> torch.Tensor:
    """
    Generate target heatmaps from keypoint coordinates
    
    Args:
        keypoints: Tensor of shape (B, K, 2) containing normalized keypoint coordinates
        heatmap_size: Tuple of (H, W) for output heatmap size
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Target heatmaps of shape (B, K, H, W)
    """
    batch_size, num_keypoints, _ = keypoints.shape
    height, width = heatmap_size
    
    # Convert normalized coordinates to pixel coordinates
    keypoints = keypoints.clone()
    keypoints[..., 0] *= width
    keypoints[..., 1] *= height
    
    heatmaps = torch.zeros(batch_size, num_keypoints, height, width,
                          device=keypoints.device)
    
    # Generate Gaussian kernel
    kernel_size = 6 * int(sigma) + 1
    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma).to(keypoints.device)
    
    for b in range(batch_size):
        for k in range(num_keypoints):
            x, y = keypoints[b, k]
            
            # Skip if keypoint is outside the image
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
                
            # Get integer coordinates
            x0, y0 = int(x), int(y)
            
            # Calculate kernel bounds
            left = max(0, x0 - kernel_size // 2)
            right = min(width, x0 + kernel_size // 2 + 1)
            top = max(0, y0 - kernel_size // 2)
            bottom = min(height, y0 + kernel_size // 2 + 1)
            
            # Calculate kernel crop bounds
            kernel_left = max(0, kernel_size // 2 - x0)
            kernel_right = min(kernel_size, kernel_size // 2 + (width - x0))
            kernel_top = max(0, kernel_size // 2 - y0)
            kernel_bottom = min(kernel_size, kernel_size // 2 + (height - y0))
            
            # Apply Gaussian kernel
            heatmaps[b, k, top:bottom, left:right] = gaussian_kernel[
                kernel_top:kernel_bottom,
                kernel_left:kernel_right
            ]
    
    return heatmaps

def decode_heatmaps(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode heatmaps to normalized keypoint coordinates (0-1) and confidence scores
    
    Args:
        heatmaps: Tensor of shape (B, K, H, W) or (K, H, W)
        
    Returns:
        keypoints: Tensor of shape (B, K, 2) or (K, 2) with normalized coordinates
        scores: Tensor of shape (B, K) or (K,) with confidence scores
    """
    if len(heatmaps.shape) == 3:
        # Add batch dimension if not present
        heatmaps = heatmaps.unsqueeze(0)
        
    batch_size, num_keypoints, height, width = heatmaps.shape
    
    # Flatten the spatial dimensions
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)
    
    # Find the indices of maximum values
    max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
    
    # Convert indices to x,y coordinates
    keypoints_x = (max_indices % width).float() / (width - 1)  # Normalize to [0, 1]
    keypoints_y = (max_indices // width).float() / (height - 1)  # Normalize to [0, 1]
    
    # Stack x,y coordinates
    keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)
    
    # Get confidence scores (maximum values from heatmaps)
    scores = max_vals
    
    # Remove batch dimension if it wasn't present in input
    if len(heatmaps.shape) == 3:
        keypoints = keypoints.squeeze(0)
        scores = scores.squeeze(0)
    
    return keypoints, scores

def decode_heatmaps_subpixel(heatmaps: torch.Tensor, window_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode heatmaps to normalized keypoint coordinates (0-1) using subpixel maximum
    
    Args:
        heatmaps: Tensor of shape (B, K, H, W) or (K, H, W)
        window_size: Size of window for subpixel maximum (must be odd)
        
    Returns:
        keypoints: Tensor of shape (B, K, 2) or (K, 2) with normalized coordinates
        scores: Tensor of shape (B, K) or (K,) with confidence scores
    """
    if len(heatmaps.shape) == 3:
        heatmaps = heatmaps.unsqueeze(0)
        
    batch_size, num_keypoints, height, width = heatmaps.shape
    
    # Find rough maximum locations
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)
    max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
    
    # Convert indices to x,y coordinates
    x_rough = max_indices % width
    y_rough = max_indices // width
    
    # Initialize results
    refined_x = torch.zeros_like(x_rough, dtype=torch.float32)
    refined_y = torch.zeros_like(y_rough, dtype=torch.float32)
    refined_scores = torch.zeros_like(max_vals, dtype=torch.float32)
    
    pad = window_size // 2
    
    for b in range(batch_size):
        for k in range(num_keypoints):
            x_c = x_rough[b, k]
            y_c = y_rough[b, k]
            
            # Extract local window around maximum
            x_start = max(0, x_c - pad)
            x_end = min(width, x_c + pad + 1)
            y_start = max(0, y_c - pad)
            y_end = min(height, y_c + pad + 1)
            
            window = heatmaps[b, k, y_start:y_end, x_start:x_end]
            
            if window.numel() > 0:
                # Calculate weighted average for subpixel accuracy
                total_mass = window.sum()
                if total_mass > 0:
                    x_coords = torch.arange(x_start, x_end, device=heatmaps.device)
                    y_coords = torch.arange(y_start, y_end, device=heatmaps.device)
                    
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    
                    x_weighted = (x_grid * window).sum() / total_mass
                    y_weighted = (y_grid * window).sum() / total_mass
                    
                    refined_x[b, k] = x_weighted
                    refined_y[b, k] = y_weighted
                    refined_scores[b, k] = window.max()
    
    # Normalize coordinates to [0, 1]
    keypoints = torch.stack([
        refined_x / (width - 1),
        refined_y / (height - 1)
    ], dim=-1)
    
    # Remove batch dimension if it wasn't present in input
    if len(heatmaps.shape) == 3:
        keypoints = keypoints.squeeze(0)
        refined_scores = refined_scores.squeeze(0)
    
    return keypoints, refined_scores

def visualize_heatmap_results(image: torch.Tensor, 
                            heatmaps: torch.Tensor, 
                            keypoints: torch.Tensor,
                            scores: torch.Tensor,
                            threshold: float = 0.5,
                            save_path: Optional[str] = None):
    """
    Visualize decoded keypoints on original image with heatmaps
    
    Args:
        image: Tensor of shape (C, H, W) or (H, W, C)
        heatmaps: Tensor of shape (K, H', W')
        keypoints: Tensor of shape (K, 2) with normalized coordinates
        scores: Tensor of shape (K,) with confidence scores
        threshold: Confidence threshold for displaying keypoints
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Convert image to numpy if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    # Create figure with subplots
    num_keypoints = heatmaps.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original image with keypoints
    axes[0].imshow(image)
    h, w = image.shape[:2]
    
    # Plot keypoints with confidence > threshold
    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        if score > threshold:
            x, y = kp[0] * w, kp[1] * h
            axes[0].plot(x, y, 'ro', markersize=5)
            axes[0].text(x, y, f'{i}:{score:.2f}', fontsize=8)
    
    axes[0].set_title('Detected Keypoints')
    
    # Plot combined heatmap
    combined_heatmap = heatmaps.max(dim=0)[0]
    axes[1].imshow(combined_heatmap.cpu(), cmap='hot')
    axes[1].set_title('Combined Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    config = HeatmapHeadConfig()
    model = HeatmapHead(config)
    
    x = torch.randn(1, config.in_channels, 56, 56)
    
    heatmaps, attention = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output heatmaps shape: {heatmaps.shape}")
    if attention:
        print(f"Channel attention shape: {attention[0].shape}")
        print(f"Spatial attention shape: {attention[1].shape}")

if __name__ == '__main__':
    main()

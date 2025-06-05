import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
from pathlib import Path
from dll.data.transforms import ITransform
# Thay thế import BACKBONE bằng MobileNetV3
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from dll.configs.model_config import BackboneConfig

# Lớp wrapper cho MobileNetV3 để có cùng interface với BACKBONE
class MobileNetV3Wrapper0(nn.Module):
    def __init__(self, config: BackboneConfig):
        super(MobileNetV3Wrapper0, self).__init__()
        # Tải MobileNetV3 với weights pretrained
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)
        
        # Điều chỉnh lớp đầu vào nếu cần
        if config.in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                config.in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        
        # Lưu trữ các feature maps trung gian
        self.backbone_outputs = []
        self.selected_layers = [1, 3, 8, 12]  # Các layer để trích xuất features
        
        # FPN đơn giản để xử lý các feature maps
        self.fpn = nn.ModuleList([
            nn.Conv2d(16, config.out_channels, kernel_size=1),
            nn.Conv2d(24, config.out_channels, kernel_size=1),
            nn.Conv2d(48, config.out_channels, kernel_size=1),
            nn.Conv2d(576, config.out_channels, kernel_size=1)
        ])
        
    def forward(self, x):
        self.backbone_outputs = []
        
        # Trích xuất features từ các layer trung gian
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if i in self.selected_layers:
                self.backbone_outputs.append(x)
        
        # Áp dụng FPN
        fpn_outputs = []
        for i, feature in enumerate(self.backbone_outputs):
            fpn_outputs.append(self.fpn[i](feature))
            
        return fpn_outputs

def parse_yolo_label(label_path):
    """
    Parse YOLO format label file
    
    Args:
        label_path: Path to YOLO label file
        
    Returns:
        List of bounding boxes in format [class_id, center_x, center_y, width, height]
    """
    bboxes = []
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found")
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:
                # YOLO format: class_id, center_x, center_y, width, height
                bbox = [
                    int(data[0]),             # class_id
                    float(data[1]),           # center_x (normalized)
                    float(data[2]),           # center_y (normalized)
                    float(data[3]),           # width (normalized)
                    float(data[4])            # height (normalized)
                ]
                bboxes.append(bbox)
    
    return bboxes

def crop_by_bbox(image, bbox, padding=0.1):
    """
    Crop image by bounding box with padding
    
    Args:
        image: PIL Image or numpy array
        bbox: [class_id, center_x, center_y, width, height] in normalized coordinates
        padding: Padding ratio to add around the bbox
        
    Returns:
        Cropped PIL Image
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    
    h, w = img_np.shape[:2]
    
    # Extract bbox coordinates
    _, cx, cy, bw, bh = bbox
    
    # Convert normalized coordinates to pixel coordinates
    cx_px = int(cx * w)
    cy_px = int(cy * h)
    bw_px = int(bw * w)
    bh_px = int(bh * h)
    
    # Calculate bbox corners with padding
    pad_w = int(bw_px * padding)
    pad_h = int(bh_px * padding)
    
    x1 = max(0, cx_px - bw_px // 2 - pad_w)
    y1 = max(0, cy_px - bh_px // 2 - pad_h)
    x2 = min(w, cx_px + bw_px // 2 + pad_w)
    y2 = min(h, cy_px + bh_px // 2 + pad_h)
    
    # Crop image
    cropped = img_np[y1:y2, x1:x2]
    
    # Convert back to PIL Image
    return Image.fromarray(cropped)

def visualize_backbone_outputs(model, image_tensor, save_dir=None, max_features=64, original_image=None):
    """
    Visualize the outputs of the backbone model
    
    Args:
        model: BACKBONE model instance
        image_tensor: Input image tensor [1, C, H, W]
        save_dir: Directory to save visualizations (if None, will display)
        max_features: Maximum number of feature channels to visualize per layer
        original_image: Optional original image for reference
    """
    # Create save directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Forward pass
    with torch.no_grad():
        fpn_outputs = model(image_tensor)
    
    # Get backbone outputs before FPN
    backbone_features = model.backbone_outputs
    
    # Visualize original image
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    
    # Normalize to [0, 1] for display
    if img_np.shape[2] == 1:  # Grayscale
        img_np = np.repeat(img_np, 3, axis=2)
    
    # Normalize to [0, 1]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # Visualize backbone features
    plt.figure(figsize=(15, 10))
    plt.suptitle("Backbone Intermediate Features", fontsize=16)
    
    # Display original image
    plt.subplot(2, 3, 1)
    
    # If original image is provided, display it alongside the cropped version
    if original_image is not None:
        # Create a figure with two subplots
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        plt.title(f"Cropped Input {image_tensor.shape}")
        plt.axis('off')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "original_vs_cropped.png"), bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
        
        # Create new figure for backbone features
        plt.figure(figsize=(15, 10))
        plt.suptitle("Backbone Intermediate Features", fontsize=16)
    
    # Display input image
    plt.subplot(2, 3, 1)
    plt.imshow(img_np)
    plt.title(f"Input Image {image_tensor.shape}")
    plt.axis('off')
    
    # Display backbone features
    for i, feature in enumerate(backbone_features):
        plt.subplot(2, 3, i+2)
        # Take mean across channels for visualization
        feature_vis = feature[0].mean(dim=0).cpu().numpy()
        feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min() + 1e-8)
        plt.imshow(feature_vis, cmap='viridis')
        plt.title(f"Backbone Layer {i+1}\n{feature.shape}")
        plt.axis('off')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "backbone_features.png"), bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    
    # Visualize FPN outputs
    plt.figure(figsize=(15, 10))
    plt.suptitle("FPN Outputs", fontsize=16)
    
    # Display FPN outputs
    for i, fpn_out in enumerate(fpn_outputs):
        plt.subplot(2, 2, i+1)
        fpn_vis = fpn_out[0].mean(dim=0).cpu().numpy()
        fpn_vis = (fpn_vis - fpn_vis.min()) / (fpn_vis.max() - fpn_vis.min() + 1e-8)
        plt.imshow(fpn_vis, cmap='viridis')
        plt.title(f"FPN Output {i+1}\n{fpn_out.shape}")
        plt.axis('off')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "fpn_outputs.png"), bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    
    # Visualize individual feature channels
    for i, fpn_out in enumerate(fpn_outputs):
        # Only visualize the first FPN output in detail
        if i > 0:
            continue
            
        #n_channels = min(max_features, fpn_out.shape[1])
        attention = ChannelAttention().to(fpn_out.device)
        channel_scores = attention(fpn_out)
        
        # Lấy top-k channels có attention score cao nhất
        k = min(max_features, fpn_out.shape[1])
        _, top_indices = torch.topk(channel_scores[0], k=k)
        
        rows = int(np.ceil(k / 4))
        plt.figure(figsize=(15, rows * 3))
        plt.suptitle(f"FPN Output {i+1} - Top {k} Important Channels", fontsize=16)
        
        for idx, j in enumerate(top_indices):
            plt.subplot(rows, 4, idx+1)
            channel_vis = fpn_out[0, j].cpu().numpy()
            channel_vis = (channel_vis - channel_vis.min()) / (channel_vis.max() - channel_vis.min() + 1e-8)
            plt.imshow(channel_vis, cmap='inferno')
            score = channel_scores[0][j].item()
            plt.title(f"Channel {j+1}\nScore: {score:.3f}")
            plt.axis('off')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"fpn_output_{i+1}_channels.png"), bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()

# Update ChannelAttention class implementation
class ChannelAttention(nn.Module):
    def __init__(self, in_channels=128, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling
        
        # Ensure reduction preserves at least 1 channel
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine both branches
        out = avg_out + max_out
        return torch.sigmoid(out)

def visualize_from_image_file(image_path, save_dir=None, label_path=None, bbox_idx=0, use_grayscale=False, use_itransform=False):
    """
    Load an image file and visualize backbone outputs
    
    Args:
        image_path: Path to image file
        save_dir: Directory to save visualizations
        label_path: Path to YOLO label file (if None, will try to find matching .txt file)
        bbox_idx: Index of bounding box to use if multiple are found (default: 0)
        use_grayscale: Whether to convert image to grayscale
        use_itransform: Whether to use ITransform for preprocessing
    """
    # Load image
    image = Image.open(image_path)
    original_image = image.copy()
    
    # Find label file if not provided
    if label_path is None:
        img_path = Path(image_path)
        potential_label_path = img_path.with_suffix('.txt')
        if potential_label_path.exists():
            label_path = potential_label_path
    
    # Crop image if label file exists
    cropped_image = None
    if label_path and os.path.exists(label_path):
        bboxes = parse_yolo_label(label_path)
        if bboxes and bbox_idx < len(bboxes):
            cropped_image = crop_by_bbox(image, bboxes[bbox_idx], padding=0.1)
            print(f"Cropped image using bounding box: {bboxes[bbox_idx]}")
            image = cropped_image
        else:
            print(f"No valid bounding box found at index {bbox_idx}")
    
    # Convert to grayscale if requested
    if use_grayscale and image.mode != 'L':
        image = image.convert('L')
        if original_image.mode != 'L':
            original_image = original_image.convert('L')
    
    # Determine channels
    in_channels = 1 if image.mode == 'L' else 3
    
    # Create transform
    if use_itransform:
        # Use ITransform for advanced preprocessing
        transform = ITransform(
            img_size=224,
            clip_limit=2.0,
            tile_size=(8, 8),
            grayscale=use_grayscale
        )
        print("Using ITransform for preprocessing")
    else:
        # Use basic transforms with CLAHE
        transform = transforms.Compose([
            transforms.Lambda(lambda img: apply_clahe(img, clip_limit=2.0, tile_size=(8, 8))),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print("Using basic transforms with CLAHE")
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0)
    
    # Create model config
    config = BackboneConfig(
        width_mult=1.0,
        in_channels=in_channels,
        out_channels=128,
        input_size=224,
        convert_to_grayscale=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Sử dụng MobileNetV3Wrapper thay vì BACKBONE
    model = MobileNetV3Wrapper0(config).to(device)
    image_tensor = image_tensor.to(device)
    
    # Convert original image for visualization
    original_image_np = np.array(original_image)
    if len(original_image_np.shape) == 2:  # Grayscale
        original_image_np = np.stack([original_image_np] * 3, axis=2)
    
    # Draw bounding box on original image if available
    if label_path and os.path.exists(label_path):
        bboxes = parse_yolo_label(label_path)
        if bboxes and bbox_idx < len(bboxes):
            bbox = bboxes[bbox_idx]
            h, w = original_image_np.shape[:2]
            
            # Extract bbox coordinates
            _, cx, cy, bw, bh = bbox
            
            # Convert normalized coordinates to pixel coordinates
            cx_px = int(cx * w)
            cy_px = int(cy * h)
            bw_px = int(bw * w)
            bh_px = int(bh * h)
            
            # Calculate bbox corners
            x1 = max(0, cx_px - bw_px // 2)
            y1 = max(0, cy_px - bh_px // 2)
            x2 = min(w, cx_px + bw_px // 2)
            y2 = min(h, cy_px + bh_px // 2)
            
            # Draw rectangle
            cv2_img = original_image_np.copy()
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            original_image_np = cv2_img
    
    # Visualize
    visualize_backbone_outputs(model, image_tensor, save_dir, original_image=original_image_np)

def main():
    """Test visualization with a sample image"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize backbone outputs')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save visualizations')
    parser.add_argument('--label', type=str, default=None, help='Path to YOLO label file')
    parser.add_argument('--bbox_idx', type=int, default=0, help='Index of bounding box to use')
    parser.add_argument('--grayscale', action='store_true', help='Convert image to grayscale')
    parser.add_argument('--it', action='store_true', help='Use ITransform for preprocessing')
    
    args = parser.parse_args()
    
    visualize_from_image_file(
        args.image, 
        args.save_dir, 
        args.label, 
        args.bbox_idx,
        args.grayscale,
        args.it
    )

if __name__ == "__main__":
    main()

def apply_clahe(img, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    # Apply CLAHE to grayscale or each channel of RGB
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        img_np = clahe.apply(img_np)
    else:  # RGB
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        img_np[:,:,0] = clahe.apply(img_np[:,:,0])
        img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL Image if input was PIL
    if isinstance(img, Image.Image):
        return Image.fromarray(img_np)
    return img_np




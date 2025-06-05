"""
Inference script for Bounding Box Detection Model
"""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dll.models.bbox_detector import BBoxDetector
from dll.configs.model_config import BackboneConfig, PersonDetectionConfig


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config_dict, device):
    """Create and load the trained model"""
    
    # Create configs
    backbone_config = BackboneConfig(
        width_mult=config_dict['model']['backbone']['width_mult'],
        in_channels=config_dict['model']['backbone']['in_channels'],
        out_channels=config_dict['model']['backbone']['out_channels'],
        input_size=config_dict['model']['backbone']['input_size'],
        convert_to_grayscale=config_dict['model']['backbone'].get('convert_to_grayscale', False)
    )
    
    detection_config = PersonDetectionConfig(
        in_channels=config_dict['model']['person_head']['in_channels'],
        num_classes=config_dict['model']['person_head']['num_classes'],
        conf_threshold=config_dict['model']['person_head']['conf_threshold'],
        nms_iou_threshold=config_dict['model']['person_head']['nms_iou_threshold'],
        anchor_sizes=config_dict['model']['person_head']['anchor_sizes']
    )
    
    # Create model
    model = BBoxDetector(backbone_config, detection_config)
    model = model.to(device)
    model.eval()
    
    return model, backbone_config


def preprocess_image(image_path, input_size, convert_to_grayscale=False):
    """
    Preprocess image for inference
    
    Args:
        image_path: Path to input image
        input_size: Model input size
        convert_to_grayscale: Whether to convert to grayscale
        
    Returns:
        Preprocessed image tensor and original image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_image = image.copy()
    original_h, original_w = image.shape[:2]
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale if needed
    if convert_to_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.expand_dims(image, axis=2)  # Add channel dimension
    
    # Resize image
    image = cv2.resize(image, (input_size, input_size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    if convert_to_grayscale:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
    else:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor, original_image, (original_w, original_h)


def postprocess_detections(detections, original_size, input_size):
    """
    Convert normalized detections back to original image coordinates
    
    Args:
        detections: List of detection tensors [N, 4] in format [cx, cy, w, h]
        original_size: (width, height) of original image
        input_size: Model input size
        
    Returns:
        List of detections in original image coordinates
    """
    original_w, original_h = original_size
    
    processed_detections = []
    
    for detection in detections:
        if len(detection) == 0:
            processed_detections.append([])
            continue
        
        # Convert normalized coordinates to original image coordinates
        detection_orig = detection.clone()
        
        # Scale coordinates
        detection_orig[:, 0] *= original_w  # cx
        detection_orig[:, 1] *= original_h  # cy
        detection_orig[:, 2] *= original_w  # w
        detection_orig[:, 3] *= original_h  # h
        
        # Convert to corner format [x1, y1, x2, y2]
        x1 = detection_orig[:, 0] - detection_orig[:, 2] / 2
        y1 = detection_orig[:, 1] - detection_orig[:, 3] / 2
        x2 = detection_orig[:, 0] + detection_orig[:, 2] / 2
        y2 = detection_orig[:, 1] + detection_orig[:, 3] / 2
        
        # Clamp to image boundaries
        x1 = torch.clamp(x1, 0, original_w)
        y1 = torch.clamp(y1, 0, original_h)
        x2 = torch.clamp(x2, 0, original_w)
        y2 = torch.clamp(y2, 0, original_h)
        
        # Stack back to [N, 4] format
        detection_corners = torch.stack([x1, y1, x2, y2], dim=1)
        processed_detections.append(detection_corners)
    
    return processed_detections


def visualize_detections(image, detections, output_path=None, show=True):
    """
    Visualize detections on image
    
    Args:
        image: Original image (BGR format)
        detections: List of detection tensors in corner format [x1, y1, x2, y2]
        output_path: Path to save visualization
        show: Whether to display the image
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Draw bounding boxes
    for i, detection in enumerate(detections):
        if len(detection) == 0:
            continue
        
        for box in detection:
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add confidence text (if available)
            ax.text(x1, y1 - 5, f'Person', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                   fontsize=10, color='white')
    
    ax.set_title(f'Detected Objects: {sum(len(det) for det in detections)}')
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def predict_single_image(model, image_path, config_dict, device, output_dir=None):
    """
    Predict bounding boxes for a single image
    
    Args:
        model: Trained BBoxDetector model
        image_path: Path to input image
        config_dict: Configuration dictionary
        device: Device to run inference on
        output_dir: Directory to save results
        
    Returns:
        List of detections
    """
    # Get model parameters
    input_size = config_dict['model']['backbone']['input_size']
    convert_to_grayscale = config_dict['model']['backbone'].get('convert_to_grayscale', False)
    
    # Preprocess image
    image_tensor, original_image, original_size = preprocess_image(
        image_path, input_size, convert_to_grayscale
    )
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        detections = model(image_tensor)
    
    # Postprocess detections
    detections_orig = postprocess_detections(detections, original_size, input_size)
    
    # Visualize results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_detections.jpg")
        visualize_detections(original_image, detections_orig, output_path, show=False)
    else:
        visualize_detections(original_image, detections_orig)
    
    return detections_orig


def predict_batch_images(model, image_dir, config_dict, device, output_dir):
    """
    Predict bounding boxes for all images in a directory
    
    Args:
        model: Trained BBoxDetector model
        image_dir: Directory containing input images
        config_dict: Configuration dictionary
        device: Device to run inference on
        output_dir: Directory to save results
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for image_path in image_files:
        print(f"Processing: {image_path.name}")
        try:
            detections = predict_single_image(
                model, str(image_path), config_dict, device, output_dir
            )
            
            # Print detection summary
            total_detections = sum(len(det) for det in detections)
            print(f"  Detected {total_detections} objects")
            
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Bounding Box Detection Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Create and load model
    model, backbone_config = create_model(config_dict, device)
    
    # Load trained weights
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from: {args.model}")
    
    # Run inference
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing single image: {args.input}")
        detections = predict_single_image(model, args.input, config_dict, device, args.output)
        
        # Print results
        total_detections = sum(len(det) for det in detections)
        print(f"Detected {total_detections} objects")
        
    elif os.path.isdir(args.input):
        # Directory of images
        print(f"Processing directory: {args.input}")
        if not args.output:
            args.output = os.path.join(args.input, 'detections')
        
        predict_batch_images(model, args.input, config_dict, device, args.output)
        
    else:
        print(f"Invalid input path: {args.input}")


if __name__ == '__main__':
    main()

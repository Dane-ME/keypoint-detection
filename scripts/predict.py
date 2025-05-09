import argparse
import torch
from pathlib import Path
from PIL import Image
import logging
from dll.data import create_dataloader
from dll.models import MultiPersonKeypointModel
from dll.data.transforms import ITransform
from dll.data.dataloader import visualize_keypoints_multi_person
from dll.configs import (
    ModelConfig, 
    TrainingConfig,
    BackboneConfig,
    PersonDetectionConfig,
    KeypointHeadConfig
)

import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model(model_path, config_dict, device):
    """Load model with proper configurations"""
    # Create model config
    model_config = ModelConfig(
        backbone=BackboneConfig(**config_dict['model']['backbone']),
        person_head=PersonDetectionConfig(**config_dict['model']['person_head']),
        keypoint_head=KeypointHeadConfig(**config_dict['model']['keypoint_head']),
        num_keypoints=config_dict['model']['keypoint_head']['num_keypoints']
    )
    
    # Create training config with default values since we're only doing inference
    training_config = TrainingConfig()
    
    # Initialize model
    model = MultiPersonKeypointModel(model_config, training_config)
    
    # Load weights - modified to handle full checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        # Remove unexpected keys before loading
        state_dict = checkpoint['model_state_dict']
        # Filter out unexpected keys
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        # If checkpoint contains only model state
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    return model

def predict_single_image(model, image_path, transform, device, gt_path=None, output_path=None):
    """Predict keypoints for a single image and return only the keypoint coordinates."""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    
    # Transform image
    transformed_image = transform(image)
    if isinstance(transformed_image, torch.Tensor):
        input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    # Load GT bounding boxes if provided
    if gt_path and gt_path.exists():
        with open(gt_path, 'r') as f:
            bboxes = []
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:
                    bbox = [
                        float(data[1]),
                        float(data[2]),
                        float(data[3]),
                        float(data[4])
                    ]
                    bboxes.append(bbox)
            bboxes = torch.tensor(bboxes, device=device)
    else:
        bboxes = None
    
    # Get predictions
    with torch.no_grad():
        batch = {
            'image': input_tensor,
            'bboxes': bboxes.unsqueeze(0) if bboxes is not None else None
        }
        outputs = model(batch)
    
    # Extract predictions for the first person
    keypoints = outputs['keypoints']  # Get keypoints tensor
    
    # Convert to numpy and ensure correct shape
    keypoints = keypoints.squeeze().cpu().numpy()  # Remove extra dimensions
    
    # Define keypoint names
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Print keypoints
    print("\nPredicted Keypoints:")
    print("-" * 40)
    print(f"Keypoints shape: {keypoints.shape}")  # Debug print
    
    # Handle different possible shapes
    if len(keypoints.shape) == 3:  # If shape is [person, keypoint, coord]
        keypoints = keypoints[0]  # Take first person
    
    for i in range(len(keypoint_names)):
        if i < len(keypoints):
            x, y = keypoints[i]
            print(f"{i+1:2d}. {keypoint_names[i]:<15} ({x:.3f}, {y:.3f})")
    
    return keypoints

def predict_directory(model, input_dir, transform, device, output_dir):
    """Predict keypoints for all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    logging.info(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        try:
            output_path = output_dir / f"{img_path.stem}_keypoints{img_path.suffix}"
            logging.info(f"Processing {img_path.name}...")
            
            predict_single_image(
                model=model,
                image_path=img_path,
                transform=transform,
                device=device,
                output_path=output_path
            )
            
            logging.info(f"Saved visualization to {output_path}")
            
        except Exception as e:
            logging.error(f"Error processing {img_path.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Predict keypoints in images')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--gt', type=str, help='Path to ground truth label file or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    config = load_config(args.config)
    device = torch.device(args.device)
    
    # Initialize transform
    transform = ITransform(
        img_size=config['model']['backbone']['input_size'],
        clip_limit=2.0,
        tile_size=(8, 8)
    )
    
    # Load model
    logging.info(f"Loading model from {args.model}")
    model = load_model(args.model, config, device)
    
    # Process input
    input_path = Path(args.input)
    gt_path = Path(args.gt) if args.gt else None

    if input_path.is_file():
        # Single image
        output_path = Path(args.output) / f"{input_path.stem}_keypoints{input_path.suffix}"
        gt_file = gt_path if gt_path and gt_path.is_file() else None
        predict_single_image(
            model=model,
            image_path=input_path,
            transform=transform,
            device=device,
            gt_path=gt_file,
            output_path=output_path
        )
        logging.info(f"Saved visualization to {output_path}")
    else:
        # Directory of images
        for img_path in input_path.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                output_path = Path(args.output) / f"{img_path.stem}_keypoints{img_path.suffix}"
                # Look for corresponding GT file
                gt_file = gt_path / f"{img_path.stem}.txt" if gt_path else None
                predict_single_image(
                    model=model,
                    image_path=img_path,
                    transform=transform,
                    device=device,
                    gt_path=gt_file,
                    output_path=output_path
                )
                logging.info(f"Saved visualization to {output_path}")
    
    logging.info("Prediction completed successfully!")

if __name__ == '__main__':
    main()

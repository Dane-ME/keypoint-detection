"""
Training script for Bounding Box Detection Model
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dll.models.bbox_detector import BBoxDetector
from dll.losses.bbox_loss import BBoxDetectionLoss
from dll.configs.model_config import BackboneConfig, PersonDetectionConfig
from dll.configs.training_config import TrainingConfig, OptimizerConfig
from dll.data.dataloader import KeypointDataLoader


def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_configs(config_dict):
    """Create configuration objects from dictionary"""
    
    # Backbone config
    backbone_config = BackboneConfig(
        width_mult=config_dict['model']['backbone']['width_mult'],
        in_channels=config_dict['model']['backbone']['in_channels'],
        out_channels=config_dict['model']['backbone']['out_channels'],
        input_size=config_dict['model']['backbone']['input_size'],
        convert_to_grayscale=config_dict['model']['backbone'].get('convert_to_grayscale', False)
    )
    
    # Person detection config
    detection_config = PersonDetectionConfig(
        in_channels=config_dict['model']['person_head']['in_channels'],
        num_classes=config_dict['model']['person_head']['num_classes'],
        conf_threshold=config_dict['model']['person_head']['conf_threshold'],
        nms_iou_threshold=config_dict['model']['person_head']['nms_iou_threshold'],
        anchor_sizes=config_dict['model']['person_head']['anchor_sizes']
    )
    
    # Training config
    training_config = TrainingConfig(
        batch_size=config_dict['training']['batch_size'],
        num_epochs=config_dict['training']['num_epochs'],
        optimizer=OptimizerConfig(
            name=config_dict['training']['optimizer']['name'],
            learning_rate=config_dict['training']['optimizer']['learning_rate'],
            weight_decay=config_dict['training']['optimizer']['weight_decay']
        )
    )
    
    return backbone_config, detection_config, training_config


def create_data_loaders(config_dict, backbone_config):
    """Create training and validation data loaders"""
    
    data_dir = config_dict['paths']['data_dir']
    batch_size = config_dict['training']['batch_size']
    
    # Training dataloader
    train_loader = KeypointDataLoader(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        img_size=backbone_config.input_size,
        grayscale=backbone_config.convert_to_grayscale,
        enable_caching=True
    )
    
    # Validation dataloader
    val_loader = KeypointDataLoader(
        data_dir=data_dir,
        split='val',
        batch_size=batch_size,
        shuffle=False,
        img_size=backbone_config.input_size,
        grayscale=backbone_config.convert_to_grayscale,
        enable_caching=True
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_coord_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['image'].to(device)
        bboxes = batch['bboxes']  # List of tensors
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get model predictions (raw predictions for loss computation)
        features = model.backbone(images)
        if isinstance(features, list):
            features = features[-1]
        
        # Get raw predictions from detection head
        x = model.detection_head.conv_layers(features)
        predictions = model.detection_head.output_conv(x)
        
        # Reshape predictions for loss computation
        B, _, H, W = predictions.shape
        predictions = predictions.view(B, model.detection_head.num_anchors, -1, H, W)
        predictions = predictions.permute(0, 3, 4, 1, 2)  # [B, H, W, num_anchors, output_dim]
        
        # Compute loss
        targets = {'bboxes': bboxes}
        loss_dict = criterion(predictions, targets)
        
        total_loss_batch = loss_dict['total_loss']
        
        # Backward pass
        total_loss_batch.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        total_coord_loss += loss_dict['coord_loss'].item()
        total_obj_loss += loss_dict['obj_loss'].item()
        total_noobj_loss += loss_dict['noobj_loss'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{total_loss_batch.item():.4f}",
            'Coord': f"{loss_dict['coord_loss'].item():.4f}",
            'Obj': f"{loss_dict['obj_loss'].item():.4f}",
            'NoObj': f"{loss_dict['noobj_loss'].item():.4f}"
        })
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}: Loss={total_loss_batch.item():.4f}")
    
    # Return average losses
    return {
        'total_loss': total_loss / num_batches,
        'coord_loss': total_coord_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'noobj_loss': total_noobj_loss / num_batches
    }


def validate_epoch(model, dataloader, criterion, device, logger):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_coord_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(device)
            bboxes = batch['bboxes']
            
            # Forward pass
            features = model.backbone(images)
            if isinstance(features, list):
                features = features[-1]
            
            # Get raw predictions
            x = model.detection_head.conv_layers(features)
            predictions = model.detection_head.output_conv(x)
            
            # Reshape predictions
            B, _, H, W = predictions.shape
            predictions = predictions.view(B, model.detection_head.num_anchors, -1, H, W)
            predictions = predictions.permute(0, 3, 4, 1, 2)
            
            # Compute loss
            targets = {'bboxes': bboxes}
            loss_dict = criterion(predictions, targets)
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_coord_loss += loss_dict['coord_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()
            total_noobj_loss += loss_dict['noobj_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}"
            })
    
    # Return average losses
    return {
        'total_loss': total_loss / num_batches,
        'coord_loss': total_coord_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'noobj_loss': total_noobj_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train Bounding Box Detection Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_dict = load_config(args.config)
    backbone_config, detection_config, training_config = create_configs(config_dict)
    
    # Setup output directory
    output_dir = args.output_dir or config_dict['paths']['output_dir']
    output_dir = os.path.join(output_dir, 'bbox_detection')
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting bounding box detection training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = BBoxDetector(backbone_config, detection_config)
    model = model.to(device)
    
    # Create loss function
    criterion = BBoxDetectionLoss(
        num_classes=detection_config.num_classes,
        anchor_sizes=detection_config.anchor_sizes,
        grid_size=(56, 56)  # Assuming 224->56 feature map
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.optimizer.learning_rate,
        weight_decay=training_config.optimizer.weight_decay
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config_dict, backbone_config)
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, training_config.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{training_config.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, logger)
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'best_val_loss': best_val_loss,
            'config': config_dict
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(output_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

"""
Training script for keypoint detection model
"""

import argparse
import sys
import yaml
from pathlib import Path
import logging
import torch
from tqdm import tqdm
from dll.configs import (
    ModelConfig, 
    TrainingConfig,
    BackboneConfig,
    PersonDetectionConfig,
    KeypointHeadConfig,
    OptimizerConfig,
    AugmentationConfig,
    LossConfig
)
from dll.training import Trainer
from dll.models import MultiPersonKeypointModel
from dll.data import create_dataloader

def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train keypoint detection model')
    parser.add_argument('--config', required=True, type=str,
                      help='Path to config file')
    parser.add_argument('--data_dir', required=True, type=str,
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', required=True, type=str,
                      help='Path to output directory')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load YAML config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    logging.info("=== Starting Training Pipeline ===")
    
    try:
        # Create proper config instances with nested configs
        model_config = ModelConfig(
            backbone=BackboneConfig(**config_dict['model']['backbone']),
            person_head=PersonDetectionConfig(**config_dict['model']['person_head']),
            keypoint_head=KeypointHeadConfig(**config_dict['model']['keypoint_head']),
            num_keypoints=config_dict['model']['keypoint_head']['num_keypoints']
        )
        
        # Properly initialize training config with nested configs
        training_config = TrainingConfig(
            num_epochs=config_dict['training']['num_epochs'],
            batch_size=config_dict['training']['batch_size'],
            num_workers=config_dict['training']['num_workers'],
            optimizer=OptimizerConfig(**config_dict['training']['optimizer']),
            augmentation=AugmentationConfig(**config_dict['training']['augmentation']),
            loss=LossConfig(**config_dict['training']['loss'])
        )
        
        # Initialize model with proper config objects
        model = MultiPersonKeypointModel(model_config, training_config)
        
        # Create dataloaders
        logging.info("Creating dataloaders...")
        train_dataloader = create_dataloader(
            dataset_dir=args.data_dir,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            split='train',
            img_size=model_config.backbone.input_size,
            grayscale=model_config.backbone.convert_to_grayscale
        )
        
        val_dataloader = create_dataloader(
            dataset_dir=args.data_dir,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            split='val',
            img_size=model_config.backbone.input_size,
            grayscale=model_config.backbone.convert_to_grayscale
        )
        
        logging.info(f"Train batches: {len(train_dataloader)}")
        logging.info(f"Validation batches: {len(val_dataloader)}")
        
        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = Trainer(
            model=model,    
            device=torch.device('cuda'),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=training_config,
            output_dir=output_dir
        )
        # Start training
        logging.info("Starting training...")
        model, history = trainer.train()
        
        # Save final model
        torch.save(model.state_dict(), output_dir / 'final_model.pth')
        
        # Print final metrics
        logging.info("\nTraining completed!")
        logging.info("\nFinal metrics:")
        if history['train_loss']:
            logging.info(f"Train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            logging.info(f"Validation loss: {history['val_loss'][-1]:.4f}")
            logging.info(f"Average Distance Error: {history['avg_ADE'][-1]:.4f}")
            
            for thresh in training_config.pck_thresholds:
                key = f'pck_{thresh}'
                if key in history and history[key]:
                    logging.info(f"PCK@{thresh}: {history[key][-1]:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()

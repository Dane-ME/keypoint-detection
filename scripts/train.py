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
    HeatmapHeadConfig,
    OptimizerConfig,
    AugmentationConfig,
    LossConfig,
    LRSchedulerConfig,
    DeviceConfig
)
from dll.configs.config_loader import load_config as load_yaml_config
from dll.training import Trainer
from dll.models import MultiPersonKeypointModel
from dll.data import create_optimized_dataloader, OptimizedKeypointsDataset
from dll.utils import initialize_device_manager, get_device

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
def get_dataloader(split, data_dir, config, model_config):
    return create_optimized_dataloader(
        dataset_dir=data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,  # Use num_workers from config
        max_persons=10,
        split=split,
        img_size=model_config.backbone.input_size,
        grayscale=model_config.backbone.convert_to_grayscale,
        enable_caching=True
    )
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def create_configs(config_dict):
    model_config = ModelConfig(
        backbone=BackboneConfig(**config_dict['model']['backbone']),
        person_head=PersonDetectionConfig(**config_dict['model']['person_head']),
        keypoint_head=KeypointHeadConfig(**config_dict['model']['keypoint_head']),
        heatmap_head=HeatmapHeadConfig(**config_dict['model']['heatmap_head']),
        num_keypoints=config_dict['model']['keypoint_head']['num_keypoints']
    )

    # Create device config from config dict
    device_config = DeviceConfig(**config_dict.get('device', {}))

    training_config = TrainingConfig(
        num_epochs=config_dict['training']['num_epochs'],
        batch_size=config_dict['training']['batch_size'],
        num_workers=config_dict['training']['num_workers'],
        optimizer=OptimizerConfig(**config_dict['training']['optimizer']),
        augmentation=AugmentationConfig(**config_dict['training']['augmentation']),
        loss=LossConfig(**config_dict['training']['loss']),
        device=device_config
    )
    return model_config, training_config
def train_pipeline(args):
    # Use the new config loader
    config = load_yaml_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger()
    logging.info("=== Starting Training Pipeline ===")

    model_config = config.model
    training_config = config.training

    # Initialize device manager with config
    initialize_device_manager(training_config.device)
    logger.info(f"Device manager initialized with: {training_config.device}")

    model = MultiPersonKeypointModel(model_config, training_config)

    train_dataloader = get_dataloader('train', args.data_dir, training_config, model_config)
    val_dataloader = get_dataloader('val', args.data_dir, training_config, model_config)

    # Use device from device manager instead of local function
    device = get_device()
    trainer = Trainer(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config,
        output_dir=output_dir
    )

    model, history = trainer.train()
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    logger.info("\nTraining completed!")
    logger.info("\nFinal metrics:")
    if history['train_loss']:
        logger.info(f"Train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"Validation loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Average Distance Error: {history['avg_ADE'][-1]:.4f}")
        for thresh in training_config.pck_thresholds:
            key = f'pck_{thresh}'
            if key in history and history[key]:
                logger.info(f"PCK@{thresh}: {history[key][-1]:.4f}")
def main():
    args = parse_args()
    try :
        train_pipeline(args)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
if __name__ == '__main__':
    main()

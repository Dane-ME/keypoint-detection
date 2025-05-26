#!/usr/bin/env python3
"""
Debug script for training step
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_configs(config_dict):
    from dll.configs import (
        ModelConfig, TrainingConfig, BackboneConfig, PersonDetectionConfig,
        KeypointHeadConfig, HeatmapHeadConfig, OptimizerConfig,
        AugmentationConfig, LossConfig, DeviceConfig
    )

    model_config = ModelConfig(
        backbone=BackboneConfig(**config_dict['model']['backbone']),
        person_head=PersonDetectionConfig(**config_dict['model']['person_head']),
        keypoint_head=KeypointHeadConfig(**config_dict['model']['keypoint_head']),
        heatmap_head=HeatmapHeadConfig(**config_dict['model']['heatmap_head']),
        num_keypoints=config_dict['model']['keypoint_head']['num_keypoints']
    )

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

from dll.models.keypoint_model import MultiPersonKeypointModel
from dll.data import create_optimized_dataloader
from dll.utils.device_manager import get_device_manager, move_batch_to_device

def debug_training_step():
    """Debug a single training step"""

    # Load config
    config_dict = load_config("configs/default_config.yaml")
    model_config, training_config = create_configs(config_dict)

    # Initialize device manager
    device_manager = get_device_manager()
    device_manager.initialize(training_config.device)

    # Create model
    model = MultiPersonKeypointModel(model_config, training_config)
    model = model.to(device_manager.device)
    model.train()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataloader
    dataloader = create_optimized_dataloader(
        dataset_dir="C:/Users/pc/Desktop/project_root-20250322T120048Z-001/dataset10k/dataset10k",
        split='train',
        batch_size=1,  # Use batch size 1 for debugging
        num_workers=0,
        img_size=model_config.backbone.input_size,
        grayscale=model_config.backbone.convert_to_grayscale,
        enable_caching=True
    )

    # Get one batch
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch)

    print("=== DEBUGGING TRAINING STEP ===")
    print(f"Batch keys: {batch.keys()}")

    # Check input shapes
    print(f"\nInput shapes:")
    print(f"  image: {batch['image'].shape}")
    print(f"  keypoints: {batch['keypoints'].shape}")
    print(f"  visibilities: {batch['visibilities'].shape}")
    print(f"  bboxes: {[b.shape for b in batch['bboxes']]}")

    try:
        # Forward pass
        print("\n=== FORWARD PASS ===")
        optimizer.zero_grad()

        outputs = model(batch)

        print(f"Model outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

        if 'loss' in outputs:
            print(f"\nLoss components:")
            print(f"  Total loss: {outputs['loss'].item():.6f}")
            print(f"  Keypoint loss: {outputs.get('keypoint_loss', 'N/A')}")
            print(f"  Visibility loss: {outputs.get('visibility_loss', 'N/A')}")
            print(f"  Coordinate loss: {outputs.get('coordinate_loss', 'N/A')}")

            # Debug why keypoint_loss is 0
            if outputs.get('keypoint_loss', 0) == 0:
                print("\n⚠️  WARNING: Keypoint loss is 0!")
                print("Checking heatmap shapes and values...")
                if 'heatmap' in outputs:
                    heatmap = outputs['heatmap']
                    print(f"  Predicted heatmap shape: {heatmap.shape}")
                    print(f"  Predicted heatmap min/max: {heatmap.min().item():.6f}/{heatmap.max().item():.6f}")
                if 'heatmaps' in batch:
                    gt_heatmap = batch['heatmaps']
                    print(f"  GT heatmap shape: {gt_heatmap.shape}")
                    print(f"  GT heatmap min/max: {gt_heatmap.min().item():.6f}/{gt_heatmap.max().item():.6f}")

            # Backward pass
            print("\n=== BACKWARD PASS ===")
            loss = outputs['loss']
            print(f"Loss requires_grad: {loss.requires_grad}")
            print(f"Loss device: {loss.device}")

            if loss.requires_grad:
                loss.backward()
                print("✅ Backward pass successful!")

                optimizer.step()
                print("✅ Optimizer step successful!")
            else:
                print("❌ Loss does not require grad - cannot backpropagate!")

        else:
            print("❌ No loss in outputs")

    except Exception as e:
        print(f"❌ Error during training step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_step()

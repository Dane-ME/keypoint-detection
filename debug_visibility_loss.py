#!/usr/bin/env python3
"""
Debug script to investigate why visibility loss is always 0
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
from dll.data import OptimizedKeypointsDataset
from torch.utils.data import DataLoader
from dll.utils.device_manager import get_device_manager, move_batch_to_device

def debug_visibility_loss():
    """Debug visibility loss computation"""

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

    # Create dataloader
    from dll.data import create_optimized_dataloader

    dataloader = create_optimized_dataloader(
        dataset_dir="C:/Users/pc/Desktop/project_root-20250322T120048Z-001/dataset10k/dataset10k",
        split='train',
        batch_size=2,
        num_workers=0,
        img_size=model_config.backbone.input_size,
        grayscale=model_config.backbone.convert_to_grayscale,
        enable_caching=True
    )

    # Get one batch
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch)

    print("=== DEBUGGING VISIBILITY LOSS ===")
    print(f"Batch keys: {batch.keys()}")

    # Check input shapes
    print(f"\nInput shapes:")
    print(f"  image: {batch['image'].shape}")
    print(f"  keypoints: {batch['keypoints'].shape}")
    print(f"  visibilities: {batch['visibilities'].shape}")
    print(f"  bboxes: {[b.shape for b in batch['bboxes']]}")

    # Check visibility values
    print(f"\nVisibility values:")
    print(f"  Min: {batch['visibilities'].min().item()}")
    print(f"  Max: {batch['visibilities'].max().item()}")
    print(f"  Unique values: {torch.unique(batch['visibilities'])}")
    print(f"  Visible keypoints: {(batch['visibilities'] > 0).sum().item()}")

    # Forward pass
    with torch.no_grad():
        outputs = model(batch)

    print(f"\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # Check if visibilities are in outputs
    if 'visibilities' in outputs:
        pred_vis = outputs['visibilities']
        print(f"\nPredicted visibilities:")
        print(f"  Shape: {pred_vis.shape}")
        print(f"  Min: {pred_vis.min().item()}")
        print(f"  Max: {pred_vis.max().item()}")
        print(f"  Sample values: {pred_vis[0, 0, :5]}")  # First 5 keypoints of first person
    else:
        print("\n❌ ERROR: 'visibilities' not found in model outputs!")

    # Check loss computation
    if 'loss' in outputs:
        print(f"\nLoss components:")
        print(f"  Total loss: {outputs['loss'].item():.6f}")
        print(f"  Keypoint loss: {outputs.get('keypoint_loss', 'N/A')}")
        print(f"  Visibility loss: {outputs.get('visibility_loss', 'N/A')}")
        print(f"  Coordinate loss: {outputs.get('coordinate_loss', 'N/A')}")

    # Debug loss function directly
    print(f"\n=== DEBUGGING LOSS FUNCTION DIRECTLY ===")

    # Prepare inputs for loss function
    if 'visibilities' in outputs:
        pred_vis = outputs['visibilities']
        gt_vis = batch['visibilities']

        # Handle shape mismatches
        if pred_vis.dim() == 4:  # [B, P, K, 3]
            pred_vis = torch.max(pred_vis, dim=1)[0]  # [B, K, 3]
        if gt_vis.dim() == 3:  # [B, P, K]
            gt_vis = torch.max(gt_vis, dim=1)[0]  # [B, K]

        print(f"Processed shapes:")
        print(f"  pred_vis: {pred_vis.shape}")
        print(f"  gt_vis: {gt_vis.shape}")

        # Check if shapes match for loss computation
        if pred_vis.size(0) == gt_vis.size(0) and pred_vis.size(1) == gt_vis.size(1):
            print(f"✅ Shapes compatible for loss computation")

            # Try computing visibility loss manually
            try:
                pred_vis_flat = pred_vis.view(-1, 3)  # [B*K, 3]
                gt_vis_flat = gt_vis.view(-1).long().clamp(0, 2)  # [B*K]

                print(f"Flattened shapes:")
                print(f"  pred_vis_flat: {pred_vis_flat.shape}")
                print(f"  gt_vis_flat: {gt_vis_flat.shape}")
                print(f"  gt_vis_flat unique: {torch.unique(gt_vis_flat)}")

                # Compute cross entropy loss
                vis_loss = torch.nn.functional.cross_entropy(pred_vis_flat, gt_vis_flat)
                print(f"Manual visibility loss: {vis_loss.item():.6f}")

            except Exception as e:
                print(f"❌ Error computing manual visibility loss: {e}")
        else:
            print(f"❌ Shape mismatch for loss computation!")

    # Check loss function configuration
    print(f"\n=== LOSS FUNCTION CONFIG ===")
    loss_fn = model.loss_fn
    print(f"Loss function type: {type(loss_fn)}")
    print(f"Device: {loss_fn.device}")
    print(f"Lambda keypoint: {loss_fn.config.lambda_keypoint}")
    print(f"Lambda visibility: {loss_fn.config.lambda_visibility}")

    # Check visibility criterion
    if hasattr(loss_fn, 'visibility_criterion'):
        print(f"Visibility criterion: {type(loss_fn.visibility_criterion)}")
        if hasattr(loss_fn.visibility_criterion, 'gamma'):
            print(f"  Gamma: {loss_fn.visibility_criterion.gamma}")
        if hasattr(loss_fn.visibility_criterion, 'alpha'):
            print(f"  Alpha: {loss_fn.visibility_criterion.alpha}")
    else:
        print("❌ No visibility criterion found!")

if __name__ == "__main__":
    debug_visibility_loss()

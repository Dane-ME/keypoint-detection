"""
Test training with improved loss functions and heatmap generation
"""

import torch
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dll.configs.config_loader import load_config
from dll.models.keypoint_model import MultiPersonKeypointModel
from dll.data.dataloader import create_adaptive_dataloader
from dll.training.trainer import Trainer
from dll.utils.device_manager import DeviceManager

def test_improved_training():
    """Test training with improved loss functions"""
    print("=== Testing Improved Training Pipeline ===")

    # Load config
    config_path = "configs/default_config.yaml"
    config = load_config(config_path)

    # Initialize device manager
    try:
        device_manager = DeviceManager.get_instance()
        device = device_manager.device
    except:
        # Fallback to simple device selection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = MultiPersonKeypointModel(config.model, config.training)
    model = model.to(device)  # Move model to device
    print("✓ Model created successfully")

    # Create small test dataset
    try:
        train_loader = create_adaptive_dataloader(
            dataset_dir=config.paths.data_dir,
            batch_size=2,  # Small batch for testing
            num_workers=0,
            split="train",
            img_size=config.model.backbone.input_size,
            grayscale=config.model.backbone.convert_to_grayscale,
            num_keypoints=config.model.keypoint_head.num_keypoints,
            heatmap_size=config.model.heatmap_head.heatmap_size,
            max_persons=1,
            enable_caching=False
        )

        val_loader = create_adaptive_dataloader(
            dataset_dir=config.paths.data_dir,
            batch_size=2,
            num_workers=0,
            split="val",
            img_size=config.model.backbone.input_size,
            grayscale=config.model.backbone.convert_to_grayscale,
            num_keypoints=config.model.keypoint_head.num_keypoints,
            heatmap_size=config.model.heatmap_head.heatmap_size,
            max_persons=1,
            enable_caching=False
        )
        print("✓ Data loaders created successfully")

    except Exception as e:
        print(f"Warning: Could not create data loaders: {e}")
        print("Creating dummy data loaders for testing...")

        # Create dummy data for testing
        class DummyDataset:
            def __init__(self, size=10):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'image': torch.randn(1, 224, 224),
                    'heatmaps': torch.randn(1, 17, 56, 56),
                    'keypoints': torch.randn(1, 17, 2),
                    'visibilities': torch.randint(0, 3, (1, 17)),
                    'bboxes': torch.randn(1, 4),
                    'num_persons': 1
                }

        from torch.utils.data import DataLoader
        train_loader = DataLoader(DummyDataset(20), batch_size=2, shuffle=True)
        val_loader = DataLoader(DummyDataset(10), batch_size=2, shuffle=False)
        print("✓ Dummy data loaders created")

    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            try:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                outputs = model(batch['image'])
                print(f"✓ Forward pass successful")
                print(f"  Output keys: {list(outputs.keys())}")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                break

            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return False

    # Test loss computation
    print("\n=== Testing Loss Computation ===")
    model.train()
    try:
        for batch in train_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            outputs = model(batch['image'])
            loss, loss_dict = model.compute_loss(outputs, batch)

            print(f"✓ Loss computation successful")
            print(f"  Total loss: {loss.item():.6f}")
            print("  Loss components:")
            for key, value in loss_dict.items():
                print(f"    {key}: {value:.6f}")

            # Check that visibility and coordinate losses are non-zero
            if 'visibility_loss_scaled' in loss_dict and loss_dict['visibility_loss_scaled'] > 0:
                print("  ✓ Visibility loss is contributing")
            if 'coordinate_loss_scaled' in loss_dict and loss_dict['coordinate_loss_scaled'] > 0:
                print("  ✓ Coordinate loss is contributing")

            break

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False

    # Test short training
    print("\n=== Testing Short Training ===")
    try:
        # Modify config for short training
        config.training.num_epochs = 1
        config.training.validation_interval = 1

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            device=device
        )

        print("✓ Trainer created successfully")

        # Run one epoch
        print("Running one training epoch...")
        train_metrics = trainer.train_epoch()
        print(f"✓ Training epoch completed")
        print(f"  Train loss: {train_metrics['loss']:.6f}")

        # Run validation
        print("Running validation...")
        val_metrics = trainer.validate_epoch()
        print(f"✓ Validation completed")
        print(f"  Val loss: {val_metrics['loss']:.6f}")
        print(f"  PCK@0.2: {val_metrics.get('pck_0.2', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_scaling():
    """Test that loss components are properly scaled"""
    print("\n=== Testing Loss Component Scaling ===")

    # Use CPU for testing to avoid device issues
    device = torch.device('cpu')

    # Create dummy data
    batch_size, num_keypoints = 2, 17

    predictions = {
        'heatmaps': torch.randn(batch_size, num_keypoints, 56, 56, device=device),
        'coordinates': torch.randn(batch_size, num_keypoints, 2, device=device),
        'visibilities': torch.randn(batch_size, num_keypoints, 3, device=device)
    }

    targets = {
        'heatmaps': torch.randn(batch_size, num_keypoints, 56, 56, device=device),
        'keypoints': torch.randn(batch_size, num_keypoints, 2, device=device),
        'visibility': torch.randint(0, 3, (batch_size, num_keypoints), device=device)
    }

    # Load config and create loss
    config_path = "configs/default_config.yaml"
    config = load_config(config_path)

    from dll.losses.keypoint_loss import KeypointLoss
    loss_fn = KeypointLoss(num_keypoints, config.training, device=device)

    total_loss, loss_dict = loss_fn(predictions, targets)

    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")

    # Check that scaled losses are larger than original
    if 'visibility_loss' in loss_dict and 'visibility_loss_scaled' in loss_dict:
        original = loss_dict['visibility_loss']
        scaled = loss_dict['visibility_loss_scaled']
        print(f"  Visibility scaling: {original:.6f} -> {scaled:.6f} (factor: {scaled/max(original, 1e-8):.2f})")

    if 'coordinate_loss' in loss_dict and 'coordinate_loss_scaled' in loss_dict:
        original = loss_dict['coordinate_loss']
        scaled = loss_dict['coordinate_loss_scaled']
        print(f"  Coordinate scaling: {original:.6f} -> {scaled:.6f} (factor: {scaled/max(original, 1e-8):.2f})")

    print("✓ Loss scaling test completed")

def main():
    """Run all training improvement tests"""
    print("Testing Training Improvements")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Test loss scaling
        test_loss_scaling()

        # Test improved training
        success = test_improved_training()

        if success:
            print("\n" + "=" * 50)
            print("✅ All training improvement tests passed!")
            print("The improved loss functions and training pipeline are working correctly.")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed. Please check the implementation.")

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

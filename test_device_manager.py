#!/usr/bin/env python3
"""
Test script to verify DeviceManager functionality and device consistency
"""

import torch
import yaml
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from dll.utils.device_manager import initialize_device_manager, get_device_manager, DeviceConfig
from dll.configs import TrainingConfig, DeviceConfig as ConfigDeviceConfig
from dll.losses.keypoint_loss import KeypointLoss

def test_device_manager():
    """Test DeviceManager initialization and functionality"""
    print("=== Testing DeviceManager ===")
    
    # Test 1: Initialize with config
    device_config = DeviceConfig(
        type='auto',
        force_cpu=False,
        mixed_precision=True,
        pin_memory=True
    )
    
    initialize_device_manager(device_config)
    device_manager = get_device_manager()
    
    print(f"Device: {device_manager.device}")
    print(f"Mixed precision: {device_manager.mixed_precision}")
    print(f"Pin memory: {device_manager.pin_memory}")
    
    # Test 2: Create tensors on managed device
    print("\n=== Testing tensor creation ===")
    tensor1 = device_manager.zeros(2, 3, 4)
    tensor2 = device_manager.ones(2, 3, 4)
    tensor3 = device_manager.randn(2, 3, 4)
    
    print(f"Tensor1 device: {tensor1.device}")
    print(f"Tensor2 device: {tensor2.device}")
    print(f"Tensor3 device: {tensor3.device}")
    
    # Test 3: Move existing tensors
    print("\n=== Testing tensor movement ===")
    cpu_tensor = torch.randn(2, 3, 4)
    print(f"Original tensor device: {cpu_tensor.device}")
    
    moved_tensor = device_manager.to_device(cpu_tensor)
    print(f"Moved tensor device: {moved_tensor.device}")
    
    # Test 4: Move batch
    print("\n=== Testing batch movement ===")
    batch = {
        'image': torch.randn(1, 3, 224, 224),
        'heatmaps': torch.randn(1, 17, 56, 56),
        'visibilities': torch.randn(1, 17),
        'bboxes': [torch.randn(1, 4)],
        'num_persons': torch.tensor([1]),
        'img_path': ['test.jpg'],
        'orig_size': [(224, 224)]
    }
    
    print("Original batch devices:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.device}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  {key}: {value[0].device}")
    
    moved_batch = device_manager.move_batch_to_device(batch)
    print("\nMoved batch devices:")
    for key, value in moved_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.device}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  {key}: {value[0].device}")
    
    # Test 5: Device info
    print("\n=== Device Information ===")
    info = device_manager.get_device_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    return True

def test_loss_with_device_manager():
    """Test KeypointLoss with DeviceManager"""
    print("\n=== Testing KeypointLoss with DeviceManager ===")
    
    # Create a simple training config
    config = TrainingConfig()
    
    # Create loss function
    loss_fn = KeypointLoss(num_keypoints=17, config=config, device=None)
    print(f"Loss function device: {loss_fn.device}")
    
    # Create test data
    device_manager = get_device_manager()
    
    predictions = {
        'heatmaps': device_manager.randn(2, 17, 56, 56),
        'visibilities': device_manager.randn(2, 17, 3),
        'coordinates': device_manager.randn(2, 17, 2)
    }
    
    targets = {
        'heatmaps': device_manager.randn(2, 17, 56, 56),
        'visibility': device_manager.zeros(2, 17).long(),
        'keypoints': device_manager.randn(2, 17, 2)
    }
    
    print("Prediction devices:")
    for key, value in predictions.items():
        print(f"  {key}: {value.device}")
    
    print("Target devices:")
    for key, value in targets.items():
        print(f"  {key}: {value.device}")
    
    # Compute loss
    try:
        total_loss, loss_dict = loss_fn(predictions, targets)
        print(f"\nLoss computation successful!")
        print(f"Total loss: {total_loss.item():.4f}")
        print("Loss components:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.4f}")
        return True
    except Exception as e:
        print(f"Error computing loss: {e}")
        return False

def test_config_integration():
    """Test integration with config system"""
    print("\n=== Testing Config Integration ===")
    
    # Load config from YAML
    config_path = Path(__file__).parent / 'configs' / 'default_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        device_config = DeviceConfig(**config_dict.get('device', {}))
        print(f"Device config from YAML: {device_config}")
        
        # Re-initialize with new config
        initialize_device_manager(device_config)
        device_manager = get_device_manager()
        print(f"Updated device: {device_manager.device}")
        
        return True
    else:
        print(f"Config file not found: {config_path}")
        return False

def main():
    """Run all tests"""
    print("Starting DeviceManager tests...\n")
    
    tests = [
        ("DeviceManager Basic Functionality", test_device_manager),
        ("KeypointLoss Integration", test_loss_with_device_manager),
        ("Config Integration", test_config_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAILED - {e}")
        print("-" * 50)
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! DeviceManager is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    return passed == total

if __name__ == "__main__":
    main()

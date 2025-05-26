#!/usr/bin/env python3
"""
Test device handling fix for the refactored keypoint loss.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dll.losses.keypoint_loss import DeviceManager, FocalLoss, KeypointLoss


def test_device_manager():
    """Test DeviceManager with CPU fallback"""
    print("=== Testing DeviceManager ===")
    
    # Test with explicit CPU device
    dm_cpu = DeviceManager(torch.device('cpu'))
    print(f"✓ DeviceManager with explicit CPU: {dm_cpu.device}")
    
    # Test with None (should auto-resolve to CPU)
    dm_auto = DeviceManager(None)
    print(f"✓ DeviceManager with auto-resolution: {dm_auto.device}")
    
    # Test tensor operations
    tensor = torch.randn(2, 3)
    result = dm_cpu.to_device(tensor)
    print(f"✓ Tensor device after to_device: {result.device}")
    
    result = dm_cpu.ensure_device(tensor)
    print(f"✓ Tensor device after ensure_device: {result.device}")
    
    return True


def test_focal_loss_cpu():
    """Test FocalLoss creation with CPU device"""
    print("\n=== Testing FocalLoss on CPU ===")
    
    try:
        # Create FocalLoss with explicit CPU device
        focal_loss = FocalLoss(gamma=2.0, alpha=0.25, device=torch.device('cpu'))
        print(f"✓ FocalLoss created successfully")
        print(f"  Device: {focal_loss.device_manager.device}")
        
        # Test forward pass
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = focal_loss(inputs, targets)
        print(f"✓ Forward pass successful, loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FocalLoss test failed: {e}")
        return False


def test_keypoint_loss_cpu():
    """Test KeypointLoss creation with CPU device"""
    print("\n=== Testing KeypointLoss on CPU ===")
    
    try:
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.lambda_keypoint = 15.0
                self.lambda_visibility = 5.0
                self.loss = type('obj', (object,), {
                    'focal_gamma': 2.0,
                    'focal_alpha': 0.25,
                    'learnable_focal_params': False,
                    'weighted_loss': type('obj', (object,), {
                        'enabled': False
                    })()
                })()
        
        config = MockConfig()
        
        # Create KeypointLoss with explicit CPU device
        keypoint_loss = KeypointLoss(17, config, torch.device('cpu'))
        print(f"✓ KeypointLoss created successfully")
        print(f"  Device: {keypoint_loss.device_manager.device}")
        print(f"  Heatmap criterion: {type(keypoint_loss.heatmap_criterion).__name__}")
        print(f"  Visibility criterion: {type(keypoint_loss.visibility_criterion).__name__}")
        
        # Test forward pass
        predictions = {
            'heatmaps': torch.randn(2, 17, 56, 56)
        }
        targets = {
            'heatmaps': torch.randn(2, 17, 56, 56)
        }
        
        total_loss, loss_dict = keypoint_loss(predictions, targets)
        print(f"✓ Forward pass successful")
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  Loss components: {list(loss_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ KeypointLoss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all device handling tests"""
    print("Testing Device Handling Fix")
    print("=" * 50)
    
    tests = [
        test_device_manager,
        test_focal_loss_cpu,
        test_keypoint_loss_cpu
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All device handling tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

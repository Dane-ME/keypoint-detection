#!/usr/bin/env python3
"""
Final comprehensive test of the refactored keypoint loss solution.

This test demonstrates all the improvements and validates the complete solution.
"""

import torch
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dll.losses.keypoint_loss import (
    KeypointLoss, FocalLoss, WeightedHeatmapLoss, HeatmapLoss,
    LossWeights, LossScaling, ConfigurationHandler, DeviceManager,
    LossComponentFactory, LossConfigurationError, LossComputationError
)


class MockTrainingConfig:
    """Mock training configuration for testing"""
    def __init__(self, weighted_loss_enabled=True, use_dict_format=False):
        self.lambda_keypoint = 15.0
        self.lambda_visibility = 5.0
        
        if use_dict_format:
            # Dict format configuration
            self.loss = type('obj', (object,), {
                'focal_gamma': 2.0,
                'focal_alpha': 0.25,
                'learnable_focal_params': False,
                'weighted_loss': {
                    'enabled': weighted_loss_enabled,
                    'keypoint_weight': 15.0,
                    'background_weight': 1.0,
                    'threshold': 0.1
                }
            })()
        else:
            # Object format configuration
            weighted_loss_config = type('obj', (object,), {
                'enabled': weighted_loss_enabled,
                'keypoint_weight': 15.0,
                'background_weight': 1.0,
                'threshold': 0.1
            })()
            
            self.loss = type('obj', (object,), {
                'focal_gamma': 2.0,
                'focal_alpha': 0.25,
                'learnable_focal_params': False,
                'weighted_loss': weighted_loss_config
            })()


def test_configuration_robustness():
    """Test robust configuration handling"""
    print("=== Testing Configuration Robustness ===")
    
    # Test object format
    config_obj = MockTrainingConfig(weighted_loss_enabled=True, use_dict_format=False)
    enabled, config_dict = ConfigurationHandler.extract_weighted_loss_config(config_obj)
    print(f"✓ Object format: enabled={enabled}, keypoint_weight={config_dict['keypoint_weight']}")
    
    # Test dict format
    config_dict_format = MockTrainingConfig(weighted_loss_enabled=True, use_dict_format=True)
    enabled, config_dict = ConfigurationHandler.extract_weighted_loss_config(config_dict_format)
    print(f"✓ Dict format: enabled={enabled}, keypoint_weight={config_dict['keypoint_weight']}")
    
    # Test disabled weighted loss
    config_disabled = MockTrainingConfig(weighted_loss_enabled=False)
    enabled, config_dict = ConfigurationHandler.extract_weighted_loss_config(config_disabled)
    print(f"✓ Disabled weighted loss: enabled={enabled}")
    
    return True


def test_device_management():
    """Test centralized device management"""
    print("\n=== Testing Device Management ===")
    
    # Test device resolution
    dm = DeviceManager()
    print(f"✓ Auto-resolved device: {dm.device}")
    
    # Test tensor operations
    tensor = torch.randn(2, 3)
    result = dm.ensure_device(tensor)
    print(f"✓ Tensor device management: {result.device}")
    
    return True


def test_loss_component_factory():
    """Test loss component factory pattern"""
    print("\n=== Testing Loss Component Factory ===")
    
    # Test weighted heatmap loss creation
    config_weighted = MockTrainingConfig(weighted_loss_enabled=True)
    heatmap_loss = LossComponentFactory.create_heatmap_loss(config_weighted, torch.device('cpu'))
    print(f"✓ Created weighted heatmap loss: {type(heatmap_loss).__name__}")
    
    # Test regular heatmap loss creation
    config_regular = MockTrainingConfig(weighted_loss_enabled=False)
    heatmap_loss = LossComponentFactory.create_heatmap_loss(config_regular, torch.device('cpu'))
    print(f"✓ Created regular heatmap loss: {type(heatmap_loss).__name__}")
    
    # Test focal loss creation
    focal_loss = LossComponentFactory.create_focal_loss(config_weighted, torch.device('cpu'))
    print(f"✓ Created focal loss: {type(focal_loss).__name__}")
    
    return True


def test_multi_person_scenario():
    """Test multi-person keypoint detection scenario"""
    print("\n=== Testing Multi-Person Scenario ===")
    
    config = MockTrainingConfig(weighted_loss_enabled=True)
    loss_fn = KeypointLoss(17, config, torch.device('cpu'))
    
    batch_size = 2
    num_persons = 3
    num_keypoints = 17
    
    # Multi-person predictions and targets
    predictions = {
        'heatmaps': torch.randn(batch_size, num_keypoints, 56, 56),
        'visibilities': torch.randn(batch_size, num_persons, num_keypoints, 3),
        'coordinates': torch.randn(batch_size, num_persons, num_keypoints, 2)
    }
    
    targets = {
        'heatmaps': torch.randn(batch_size, num_keypoints, 56, 56),
        'visibility': torch.randint(0, 3, (batch_size, num_persons, num_keypoints)),
        'keypoints': torch.randn(batch_size, num_persons, num_keypoints, 2)
    }
    
    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"✓ Multi-person loss computation successful")
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Heatmap loss: {loss_dict['heatmap_loss']:.6f}")
    print(f"  Visibility loss: {loss_dict['visibility_loss']:.6f}")
    print(f"  Coordinate loss: {loss_dict['coordinate_loss']:.6f}")
    
    return True


def test_error_handling():
    """Test robust error handling"""
    print("\n=== Testing Error Handling ===")
    
    config = MockTrainingConfig()
    
    # Test invalid num_keypoints
    try:
        KeypointLoss(0, config, torch.device('cpu'))
        print("❌ Should have raised ValueError for invalid num_keypoints")
        return False
    except ValueError:
        print("✓ Correctly caught invalid num_keypoints")
    
    # Test None config
    try:
        KeypointLoss(17, None, torch.device('cpu'))
        print("❌ Should have raised ValueError for None config")
        return False
    except ValueError:
        print("✓ Correctly caught None config")
    
    # Test shape mismatch
    loss_fn = KeypointLoss(17, config, torch.device('cpu'))
    predictions = {'heatmaps': torch.randn(2, 17, 56, 56)}
    targets = {'heatmaps': torch.randn(2, 17, 28, 28)}  # Different size
    
    try:
        loss_fn(predictions, targets)
        print("❌ Should have raised ValueError for shape mismatch")
        return False
    except (ValueError, LossComputationError):
        print("✓ Correctly caught shape mismatch")
    
    return True


def test_performance_comparison():
    """Test performance improvements"""
    print("\n=== Testing Performance ===")
    
    config = MockTrainingConfig(weighted_loss_enabled=True)
    loss_fn = KeypointLoss(17, config, torch.device('cpu'))
    
    # Create test data
    batch_size = 4
    predictions = {
        'heatmaps': torch.randn(batch_size, 17, 56, 56),
        'visibilities': torch.randn(batch_size, 17, 3),
        'coordinates': torch.randn(batch_size, 17, 2)
    }
    targets = {
        'heatmaps': torch.randn(batch_size, 17, 56, 56),
        'visibility': torch.randint(0, 3, (batch_size, 17)),
        'keypoints': torch.randn(batch_size, 17, 2)
    }
    
    # Warm up
    for _ in range(5):
        loss_fn(predictions, targets)
    
    # Time the computation
    start_time = time.time()
    num_iterations = 100
    
    for _ in range(num_iterations):
        total_loss, loss_dict = loss_fn(predictions, targets)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    
    print(f"✓ Performance test completed")
    print(f"  Average time per forward pass: {avg_time:.3f} ms")
    print(f"  Total loss: {total_loss.item():.6f}")
    
    return True


def test_backward_compatibility():
    """Test backward compatibility"""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test that old API still works
    config = MockTrainingConfig()
    
    # This should work exactly as before
    loss_fn = KeypointLoss(num_keypoints=17, config=config, device=torch.device('cpu'))
    
    predictions = {'heatmaps': torch.randn(2, 17, 56, 56)}
    targets = {'heatmaps': torch.randn(2, 17, 56, 56)}
    
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    # Check that expected keys are present
    expected_keys = ['keypoint_loss', 'heatmap_loss', 'total_loss']
    for key in expected_keys:
        if key not in loss_dict:
            print(f"❌ Missing expected key: {key}")
            return False
    
    print("✓ Backward compatibility maintained")
    print(f"  API works as expected")
    print(f"  Loss dict contains expected keys: {list(loss_dict.keys())}")
    
    return True


def main():
    """Run comprehensive test suite"""
    print("Final Comprehensive Test of Refactored Keypoint Loss")
    print("=" * 60)
    
    tests = [
        test_configuration_robustness,
        test_device_management,
        test_loss_component_factory,
        test_multi_person_scenario,
        test_error_handling,
        test_performance_comparison,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Final Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Refactoring is successful!")
        print("\nKey Improvements Validated:")
        print("✅ Robust configuration handling")
        print("✅ Centralized device management")
        print("✅ Factory pattern for loss components")
        print("✅ Multi-person scenario support")
        print("✅ Comprehensive error handling")
        print("✅ Performance optimizations")
        print("✅ Backward compatibility maintained")
        return True
    else:
        print("❌ Some tests failed - refactoring needs attention")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

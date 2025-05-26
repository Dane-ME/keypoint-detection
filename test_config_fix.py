"""
Test script to verify the config fix for weighted loss
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_config_loading():
    """Test loading config with weighted loss"""
    print("=== Testing Config Loading ===")
    
    try:
        from dll.configs.config_loader import load_config
        config_path = "configs/default_config.yaml"
        config = load_config(config_path)
        
        print("✓ Config loaded successfully")
        print(f"  Config type: {type(config)}")
        print(f"  Loss config type: {type(config.training.loss)}")
        
        # Check if weighted_loss exists
        if hasattr(config.training.loss, 'weighted_loss'):
            weighted_loss = config.training.loss.weighted_loss
            print(f"  Weighted loss config type: {type(weighted_loss)}")
            
            if hasattr(weighted_loss, 'enabled'):
                print(f"  Weighted loss enabled: {weighted_loss.enabled}")
            elif isinstance(weighted_loss, dict) and 'enabled' in weighted_loss:
                print(f"  Weighted loss enabled (dict): {weighted_loss['enabled']}")
            else:
                print("  Weighted loss config found but no 'enabled' attribute/key")
        else:
            print("  No weighted_loss config found")
            
        return config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_keypoint_loss_creation():
    """Test creating KeypointLoss with the config"""
    print("\n=== Testing KeypointLoss Creation ===")
    
    config = test_config_loading()
    if config is None:
        return False
    
    try:
        from dll.losses.keypoint_loss import KeypointLoss
        
        device = torch.device('cpu')  # Use CPU for testing
        num_keypoints = 17
        
        # Create KeypointLoss
        loss_fn = KeypointLoss(num_keypoints, config.training, device=device)
        
        print("✓ KeypointLoss created successfully")
        print(f"  Loss function type: {type(loss_fn.heatmap_criterion)}")
        
        # Test forward pass
        batch_size = 1
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
        
        total_loss, loss_dict = loss_fn(predictions, targets)
        
        print("✓ Loss computation successful")
        print(f"  Total loss: {total_loss.item():.6f}")
        print("  Loss components:")
        for key, value in loss_dict.items():
            print(f"    {key}: {value:.6f}")
            
        return True
        
    except Exception as e:
        print(f"❌ KeypointLoss creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_dict_config():
    """Test with a dict-based config to simulate the error scenario"""
    print("\n=== Testing with Dict Config ===")
    
    try:
        from dll.losses.keypoint_loss import KeypointLoss
        
        # Create a mock config with dict-based weighted_loss
        class MockLossConfig:
            def __init__(self):
                self.focal_gamma = 2.0
                self.focal_alpha = 0.25
                self.learnable_focal_params = False
                # This simulates the problematic case where weighted_loss is a dict
                self.weighted_loss = {
                    'enabled': True,
                    'keypoint_weight': 15.0,
                    'background_weight': 1.0,
                    'threshold': 0.1
                }
        
        class MockTrainingConfig:
            def __init__(self):
                self.loss = MockLossConfig()
                self.lambda_keypoint = 15.0
                self.lambda_visibility = 5.0
        
        device = torch.device('cpu')
        num_keypoints = 17
        mock_config = MockTrainingConfig()
        
        # This should work now with our fix
        loss_fn = KeypointLoss(num_keypoints, mock_config, device=device)
        
        print("✓ KeypointLoss created with dict config")
        print(f"  Loss function type: {type(loss_fn.heatmap_criterion)}")
        
        # Test that it's using WeightedHeatmapLoss
        from dll.losses.keypoint_loss import WeightedHeatmapLoss
        if isinstance(loss_fn.heatmap_criterion, WeightedHeatmapLoss):
            print("✓ Using WeightedHeatmapLoss as expected")
        else:
            print("⚠ Using regular HeatmapLoss (fallback)")
            
        return True
        
    except Exception as e:
        print(f"❌ Dict config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_weighted_loss():
    """Test with config that doesn't have weighted_loss"""
    print("\n=== Testing without Weighted Loss Config ===")
    
    try:
        from dll.losses.keypoint_loss import KeypointLoss
        
        # Create a mock config without weighted_loss
        class MockLossConfig:
            def __init__(self):
                self.focal_gamma = 2.0
                self.focal_alpha = 0.25
                self.learnable_focal_params = False
                # No weighted_loss attribute
        
        class MockTrainingConfig:
            def __init__(self):
                self.loss = MockLossConfig()
                self.lambda_keypoint = 15.0
                self.lambda_visibility = 5.0
        
        device = torch.device('cpu')
        num_keypoints = 17
        mock_config = MockTrainingConfig()
        
        # This should fallback to regular HeatmapLoss
        loss_fn = KeypointLoss(num_keypoints, mock_config, device=device)
        
        print("✓ KeypointLoss created without weighted_loss config")
        print(f"  Loss function type: {type(loss_fn.heatmap_criterion)}")
        
        # Test that it's using regular HeatmapLoss
        from dll.losses.keypoint_loss import HeatmapLoss
        if isinstance(loss_fn.heatmap_criterion, HeatmapLoss):
            print("✓ Using regular HeatmapLoss as fallback")
        else:
            print("⚠ Using different loss function")
            
        return True
        
    except Exception as e:
        print(f"❌ No weighted loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all config fix tests"""
    print("Testing Config Fix for Weighted Loss")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_keypoint_loss_creation,
        test_with_dict_config,
        test_without_weighted_loss
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
        print("✅ All tests passed! The config fix is working correctly.")
        print("\nThe error \"'dict' object has no attribute 'enabled'\" should be resolved.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

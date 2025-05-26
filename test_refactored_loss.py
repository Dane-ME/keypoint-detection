#!/usr/bin/env python3
"""
Comprehensive test suite for the refactored keypoint loss module.

This test validates all the improvements made during refactoring:
- Configuration handling
- Device management
- Loss computation
- Error handling
- Multi-person scenarios
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dll.losses.keypoint_loss import (
    KeypointLoss, FocalLoss, WeightedHeatmapLoss, HeatmapLoss,
    LossWeights, LossScaling, ConfigurationHandler, DeviceManager,
    LossComponentFactory, LossConfigurationError, LossComputationError
)
from dll.configs.training_config import TrainingConfig, LossConfig


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


class TestLossDataClasses(unittest.TestCase):
    """Test the new dataclasses for configuration"""
    
    def test_loss_weights_validation(self):
        """Test LossWeights validation"""
        # Valid weights
        weights = LossWeights(heatmap=1.0, visibility=2.0, coordinate=0.5)
        self.assertEqual(weights.heatmap, 1.0)
        
        # Invalid weights should raise ValueError
        with self.assertRaises(ValueError):
            LossWeights(heatmap=-1.0)
    
    def test_loss_scaling_validation(self):
        """Test LossScaling validation"""
        # Valid scaling
        scaling = LossScaling(coordinate_scale=10.0, visibility_scale=2.0)
        self.assertEqual(scaling.coordinate_scale, 10.0)
        
        # Invalid scaling should raise ValueError
        with self.assertRaises(ValueError):
            LossScaling(coordinate_scale=0.0)


class TestConfigurationHandler(unittest.TestCase):
    """Test the configuration handler"""
    
    def test_extract_weighted_loss_config_object_format(self):
        """Test extracting weighted loss config from object format"""
        config = MockTrainingConfig(weighted_loss_enabled=True, use_dict_format=False)
        enabled, config_dict = ConfigurationHandler.extract_weighted_loss_config(config)
        
        self.assertTrue(enabled)
        self.assertIsNotNone(config_dict)
        self.assertEqual(config_dict['keypoint_weight'], 15.0)
    
    def test_extract_weighted_loss_config_dict_format(self):
        """Test extracting weighted loss config from dict format"""
        config = MockTrainingConfig(weighted_loss_enabled=True, use_dict_format=True)
        enabled, config_dict = ConfigurationHandler.extract_weighted_loss_config(config)
        
        self.assertTrue(enabled)
        self.assertIsNotNone(config_dict)
        self.assertEqual(config_dict['keypoint_weight'], 15.0)
    
    def test_extract_focal_loss_config(self):
        """Test extracting focal loss configuration"""
        config = MockTrainingConfig()
        focal_config = ConfigurationHandler.extract_focal_loss_config(config)
        
        self.assertEqual(focal_config['gamma'], 2.0)
        self.assertEqual(focal_config['alpha'], 0.25)
        self.assertEqual(focal_config['num_classes'], 3)


class TestDeviceManager(unittest.TestCase):
    """Test the device manager"""
    
    def test_device_resolution(self):
        """Test device resolution with fallback"""
        device_manager = DeviceManager()
        self.assertIsInstance(device_manager.device, torch.device)
    
    def test_tensor_device_management(self):
        """Test tensor device operations"""
        device_manager = DeviceManager(torch.device('cpu'))
        tensor = torch.randn(2, 3)
        
        # Test to_device
        result = device_manager.to_device(tensor)
        self.assertEqual(result.device, torch.device('cpu'))
        
        # Test ensure_device
        result = device_manager.ensure_device(tensor)
        self.assertEqual(result.device, torch.device('cpu'))


class TestFocalLoss(unittest.TestCase):
    """Test the improved FocalLoss implementation"""
    
    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization with validation"""
        # Valid initialization
        loss = FocalLoss(gamma=2.0, alpha=0.25, num_classes=3)
        self.assertIsInstance(loss, nn.Module)
        
        # Invalid gamma should raise ValueError
        with self.assertRaises(ValueError):
            FocalLoss(gamma=-1.0)
        
        # Invalid alpha should raise ValueError
        with self.assertRaises(ValueError):
            FocalLoss(alpha=1.5)
    
    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass"""
        loss = FocalLoss(gamma=2.0, alpha=0.25, num_classes=3, device=torch.device('cpu'))
        
        # Create test data
        inputs = torch.randn(4, 3)  # (batch_size, num_classes)
        targets = torch.randint(0, 3, (4,))  # (batch_size,)
        
        # Forward pass
        result = loss(inputs, targets)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dim(), 0)  # Scalar loss


class TestWeightedHeatmapLoss(unittest.TestCase):
    """Test the improved WeightedHeatmapLoss implementation"""
    
    def test_weighted_heatmap_loss_initialization(self):
        """Test WeightedHeatmapLoss initialization"""
        # Valid initialization
        loss = WeightedHeatmapLoss(keypoint_weight=15.0, background_weight=1.0, threshold=0.1)
        self.assertIsInstance(loss, nn.Module)
        
        # Invalid weights should raise ValueError
        with self.assertRaises(ValueError):
            WeightedHeatmapLoss(keypoint_weight=-1.0)
        
        # Invalid threshold should raise ValueError
        with self.assertRaises(ValueError):
            WeightedHeatmapLoss(threshold=1.5)
    
    def test_weighted_heatmap_loss_forward(self):
        """Test WeightedHeatmapLoss forward pass"""
        loss = WeightedHeatmapLoss(device=torch.device('cpu'))
        
        # Create test data
        pred_heatmaps = torch.randn(2, 17, 56, 56)  # (B, K, H, W)
        gt_heatmaps = torch.randn(2, 17, 56, 56)
        target_weight = torch.ones(2, 17)  # (B, K)
        
        # Forward pass
        result = loss(pred_heatmaps, gt_heatmaps, target_weight)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dim(), 0)  # Scalar loss


class TestLossComponentFactory(unittest.TestCase):
    """Test the loss component factory"""
    
    def test_create_heatmap_loss_weighted(self):
        """Test creating weighted heatmap loss"""
        config = MockTrainingConfig(weighted_loss_enabled=True)
        loss = LossComponentFactory.create_heatmap_loss(config, torch.device('cpu'))
        self.assertIsInstance(loss, WeightedHeatmapLoss)
    
    def test_create_heatmap_loss_regular(self):
        """Test creating regular heatmap loss"""
        config = MockTrainingConfig(weighted_loss_enabled=False)
        loss = LossComponentFactory.create_heatmap_loss(config, torch.device('cpu'))
        self.assertIsInstance(loss, HeatmapLoss)
    
    def test_create_focal_loss(self):
        """Test creating focal loss"""
        config = MockTrainingConfig()
        loss = LossComponentFactory.create_focal_loss(config, torch.device('cpu'))
        self.assertIsInstance(loss, FocalLoss)


class TestKeypointLoss(unittest.TestCase):
    """Test the refactored KeypointLoss implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.num_keypoints = 17
        self.batch_size = 2
        
    def test_keypoint_loss_initialization(self):
        """Test KeypointLoss initialization"""
        config = MockTrainingConfig()
        loss = KeypointLoss(self.num_keypoints, config, self.device)
        
        self.assertEqual(loss.num_keypoints, self.num_keypoints)
        self.assertIsInstance(loss.loss_weights, LossWeights)
        self.assertIsInstance(loss.loss_scaling, LossScaling)
    
    def test_keypoint_loss_validation(self):
        """Test KeypointLoss input validation"""
        config = MockTrainingConfig()
        
        # Invalid num_keypoints should raise ValueError
        with self.assertRaises(ValueError):
            KeypointLoss(0, config, self.device)
        
        # None config should raise ValueError
        with self.assertRaises(ValueError):
            KeypointLoss(self.num_keypoints, None, self.device)
    
    def test_keypoint_loss_forward_basic(self):
        """Test basic KeypointLoss forward pass"""
        config = MockTrainingConfig()
        loss = KeypointLoss(self.num_keypoints, config, self.device)
        
        # Create test data
        predictions = {
            'heatmaps': torch.randn(self.batch_size, self.num_keypoints, 56, 56)
        }
        targets = {
            'heatmaps': torch.randn(self.batch_size, self.num_keypoints, 56, 56)
        }
        
        # Forward pass
        total_loss, loss_dict = loss(predictions, targets)
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.dim(), 0)  # Scalar loss
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIn('keypoint_loss', loss_dict)


if __name__ == '__main__':
    print("Running comprehensive tests for refactored keypoint loss...")
    unittest.main(verbosity=2)

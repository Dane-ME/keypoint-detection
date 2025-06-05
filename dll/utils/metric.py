"""
Metrics calculation utilities for keypoint detection evaluation
"""

import torch
from typing import Dict, Optional
import torch.nn as nn
from dll.configs.training_config import TrainingConfig

# Lấy config metrics từ TrainingConfig
DEFAULT_CONFIG = TrainingConfig()
METRIC_CONFIG = {
    'pck_thresholds': DEFAULT_CONFIG.pck_thresholds,
    'default_validation_threshold': DEFAULT_CONFIG.default_validation_threshold
}

def create_base_metrics() -> Dict:
    """Initialize base metrics dictionary."""
    metrics = {
        'loss': 0.0,
        'keypoint_loss': 0.0,
        'visibility_loss': 0.0,
        'total_loss': 0.0,
        'total_persons': 0,
        'total_ADE': 0.0,
        'num_batches': 0,
        'avg_ADE': 0.0,
        'num_detections': 0  # Add num_detections metric
    }
    
    # Add PCK metrics
    for thresh in METRIC_CONFIG['pck_thresholds']:
        metrics[f'pck_{thresh}'] = 0.0
        
    return metrics

def compute_batch_metrics(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    is_training: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute metrics for a single batch using model's step methods."""
    
    # Validate required batch keys
    required_keys = ['image', 'keypoints', 'visibilities', 'bboxes']
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"Required key '{key}' not found in batch. Available keys: {list(batch.keys())}")
    
    # Move batch data to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}

    if is_training:
        batch_metrics = model.train_step(batch, optimizer)
    else:
        batch_metrics = model.validate_step(batch)

    return batch_metrics

def compute_epoch_metrics(metrics: Dict) -> Dict:
    """Compute average metrics for the entire epoch."""
    num_batches = max(1, metrics['num_batches'])
    total_persons = max(1, metrics['total_persons'])
    
    results = {
        'avg_loss': metrics['loss'] / num_batches,
        'avg_ADE': metrics['total_ADE'] / total_persons,
    }
    
    # Calculate average PCK scores
    for thresh in METRIC_CONFIG['pck_thresholds']:
        key = f'pck_{thresh}'
        if key in metrics:
            results[key] = metrics[key] / total_persons
            
    return results

def compute_pck(pred_keypoints: torch.Tensor, 
                gt_keypoints: torch.Tensor,
                visibilities: torch.Tensor,
                threshold: float) -> float:
    """
    Compute PCK (Percentage of Correct Keypoints) metric
    """
    # Calculate distances between predicted and ground truth keypoints
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)  # [B, K]
    
    # Apply visibility mask
    valid_distances = distances[visibilities > 0]
    
    if len(valid_distances) == 0:
        print(f"Warning: No valid keypoints found for PCK@{threshold}")
        return 0.0
    
    # Count correct predictions (distance < threshold)
    correct = (valid_distances < threshold).float().mean().item()
    
    print(f"PCK@{threshold}: {correct:.4f} (from {len(valid_distances)} valid points)")
    return correct

def compute_ade(pred_keypoints: torch.Tensor,
                gt_keypoints: torch.Tensor,
                visibilities: torch.Tensor) -> float:
    """
    Compute Average Distance Error
    """
    # Calculate Euclidean distances
    distances = torch.norm(pred_keypoints - gt_keypoints, dim=-1)  # [B, K]
    
    # Apply visibility mask
    valid_distances = distances[visibilities > 0]
    
    if len(valid_distances) == 0:
        print("Warning: No valid keypoints found for ADE calculation")
        return 0.0
    
    ade = valid_distances.mean().item()
    print(f"ADE: {ade:.4f} (from {len(valid_distances)} valid points)")
    return ade

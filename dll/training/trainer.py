"""
Training utilities for keypoint detection model
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset, DataLoader
import logging
from collections import defaultdict
import os

from dll.utils.metric import (
    compute_batch_metrics,
    compute_epoch_metrics,
    create_base_metrics
)
from dll.utils.device_manager import get_device_manager, move_batch_to_device
from dll.configs.training_config import TrainingConfig

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        output_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = False
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_amp = use_amp

        # Setup device - use device manager if available
        try:
            device_manager = get_device_manager()
            self.device = device_manager.device
            self.model = device_manager.to_device(self.model)
            self.use_amp = device_manager.mixed_precision
        except RuntimeError:
            # Fallback if device manager not initialized
            self.device = device
            self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup AMP scaler for mixed precision - use self.use_amp which may be updated by DeviceManager
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Setup scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )

        # Initialize metrics history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_keypoint_loss': [],
            'val_keypoint_loss': [],
            'train_visibility_loss': [],
            'val_visibility_loss': [],
            'learning_rate': [],
            'avg_ADE': []
        }

        # Add PCK metrics
        for thresh in self.config.pck_thresholds:
            self.history[f'pck_{thresh}'] = []

        # Setup logging
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=self.output_dir / 'training.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )

    def _create_optimizer(self):
        """Create optimizer based on config."""
        opt_config = self.config.optimizer
        if opt_config.name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.name.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config.name}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.output_dir:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model if specified
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = create_base_metrics()

        progress_bar = tqdm(self.train_dataloader,
                           desc='Training',
                           leave=True,
                           dynamic_ncols=True)

        for batch in progress_bar:
            # Move data to device using device manager
            batch = move_batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                # Mixed precision forward pass
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch)

                # Backward pass with gradient scaling
                if 'loss' in outputs:
                    self.scaler.scale(outputs['loss']).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Standard precision
                outputs = self.model(batch)

                # Backward pass
                if 'loss' in outputs:
                    outputs['loss'].backward()
                    self.optimizer.step()

            # Update metrics (common for both AMP and standard)
            if 'loss' in outputs:
                epoch_metrics['loss'] += outputs['loss'].item()
                epoch_metrics['keypoint_loss'] += outputs.get('keypoint_loss', 0)
                epoch_metrics['visibility_loss'] += outputs.get('visibility_loss', 0)
                epoch_metrics['num_batches'] += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{outputs['loss'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
            else:
                logging.warning("No loss computed for this batch")

        epoch_metrics['loss'] /= epoch_metrics['num_batches']
        epoch_metrics['keypoint_loss'] /= epoch_metrics['num_batches']
        epoch_metrics['visibility_loss'] /= epoch_metrics['num_batches']
        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with detailed loss computation like training."""
        self.model.eval()
        epoch_metrics = create_base_metrics()

        # Add additional validation metrics
        epoch_metrics.update({
            'avg_ADE': 0.0,
            'pck_0.002': 0.0,
            'pck_0.05': 0.0,
            'pck_0.2': 0.0
        })

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader,
                              desc='Validating',
                              leave=False,
                              dynamic_ncols=True)

            for batch in progress_bar:
                # Move data to device using device manager
                batch = move_batch_to_device(batch)

                # Forward pass (same as training but without backward)
                if self.use_amp:
                    # Mixed precision forward pass
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch)
                else:
                    # Standard precision
                    outputs = self.model(batch)

                # Update metrics (same as training)
                if 'loss' in outputs:
                    epoch_metrics['loss'] += outputs['loss'].item()
                    epoch_metrics['keypoint_loss'] += outputs.get('keypoint_loss', 0)
                    epoch_metrics['visibility_loss'] += outputs.get('visibility_loss', 0)
                    epoch_metrics['num_batches'] += 1

                    # Calculate additional validation metrics (PCK, ADE)
                    if 'keypoints' in outputs and 'keypoints' in batch:
                        additional_metrics = self._calculate_validation_metrics(outputs, batch)
                        for key, value in additional_metrics.items():
                            epoch_metrics[key] += value

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{outputs['loss'].item():.4f}",
                        'ADE': f"{epoch_metrics['avg_ADE'] / max(epoch_metrics['num_batches'], 1):.4f}",
                        'PCK@0.2': f"{epoch_metrics['pck_0.2'] / max(epoch_metrics['num_batches'], 1):.4f}"
                    })
                else:
                    logging.warning("No loss computed for validation batch")

        # Average all metrics
        if epoch_metrics['num_batches'] > 0:
            for key in epoch_metrics:
                if key != 'num_batches':
                    epoch_metrics[key] /= epoch_metrics['num_batches']

        return dict(epoch_metrics)

    def _calculate_validation_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate validation metrics like PCK and ADE."""
        metrics = {}

        try:
            # Get predicted and ground truth keypoints
            pred_keypoints = outputs['keypoints']
            gt_keypoints = batch['keypoints']
            gt_visibilities = batch['visibilities']

            # Handle dimension mismatches
            if pred_keypoints.dim() == 5:  # [B, P, 1, K, 2]
                pred_keypoints = pred_keypoints.squeeze(2)  # -> [B, P, K, 2]
            if pred_keypoints.dim() == 4 and gt_keypoints.dim() == 3:  # [B, P, K, 2] vs [B, K, 2]
                # Take first person for comparison
                pred_keypoints = pred_keypoints[:, 0, :, :]  # -> [B, K, 2]

            # Ensure same shape
            if pred_keypoints.shape != gt_keypoints.shape:
                # Skip metrics calculation if shapes don't match
                return {
                    'avg_ADE': 0.0,
                    'pck_0.002': 0.0,
                    'pck_0.05': 0.0,
                    'pck_0.2': 0.0
                }

            # Calculate Average Distance Error (ADE)
            dist = torch.norm(pred_keypoints - gt_keypoints, dim=-1)  # [B, K]

            # Only consider visible keypoints
            visible_mask = gt_visibilities > 0
            if visible_mask.sum() > 0:
                metrics['avg_ADE'] = dist[visible_mask].mean().item()
            else:
                metrics['avg_ADE'] = 0.0

            # Calculate PCK (Percentage of Correct Keypoints)
            for threshold in self.config.pck_thresholds:
                if visible_mask.sum() > 0:
                    correct = (dist <= threshold) & visible_mask
                    metrics[f'pck_{threshold}'] = correct.float().mean().item()
                else:
                    metrics[f'pck_{threshold}'] = 0.0

        except Exception as e:
            logging.warning(f"Error calculating validation metrics: {e}")
            # Return default metrics
            metrics = {
                'avg_ADE': 0.0,
                'pck_0.002': 0.0,
                'pck_0.05': 0.0,
                'pck_0.2': 0.0
            }

        return metrics

    def train(self) -> Tuple[nn.Module, Dict]:
        """Train the model for the specified number of epochs."""
        best_val_loss = float('inf')

        logging.info("=== Starting Training Pipeline ===")

        for epoch in range(1, self.config.num_epochs + 1):
            logging.info(f"\nEpoch {epoch}/{self.config.num_epochs}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch()

            # Log detailed metrics
            logging.info(f"Train Loss: {train_metrics['loss']:.4f} (keypoint: {train_metrics['keypoint_loss']:.4f}, visibility: {train_metrics['visibility_loss']:.4f})")
            logging.info(f"Val Loss: {val_metrics['loss']:.4f} (keypoint: {val_metrics['keypoint_loss']:.4f}, visibility: {val_metrics['visibility_loss']:.4f})")
            logging.info(f"ADE: {val_metrics['avg_ADE']:.4f}")

            # Log PCK metrics
            logging.info("PCK Metrics:")
            for thresh in self.config.pck_thresholds:
                key = f'pck_{thresh}'
                if key in val_metrics:
                    logging.info(f"  PCK@{thresh}: {val_metrics[key]:.4f}")

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            self.scheduler.step(val_metrics['pck_0.2'])

            # Update history
            self._update_history(epoch, train_metrics, val_metrics)

            # Keep track of best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)

        # Final check for focal loss parameters
        if self.config.loss.learnable_focal_params:
            focal_loss = self.model.loss_fn.visibility_criterion
            final_gamma = focal_loss.gamma.item()
            logging.info(f"\nFinal focal gamma: {final_gamma:.4f}")

            if hasattr(focal_loss, 'alpha') and focal_loss.alpha is not None:
                if focal_loss.alpha.dim() == 0:
                    final_alpha = focal_loss.alpha.item()
                else:
                    final_alpha = focal_loss.alpha.mean().item()
                logging.info(f"Final focal alpha: {final_alpha:.4f}")

        # Cuối cùng chỉ log metrics một lần
        if self.history['train_loss']:
            self.log_metrics({
                'train_loss': self.history['train_loss'][-1],
                'val_loss': self.history['val_loss'][-1],
                'avg_ADE': self.history['avg_ADE'][-1],
                **{f'pck_{thresh}': self.history[f'pck_{thresh}'][-1]
                   for thresh in self.config.pck_thresholds}
            }, prefix="Final metrics")

        return self.model, self.history

    def _update_history(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Update history with training and validation metrics."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_keypoint_loss'].append(train_metrics['keypoint_loss'])
        self.history['val_keypoint_loss'].append(val_metrics['keypoint_loss'])
        self.history['train_visibility_loss'].append(train_metrics['visibility_loss'])
        self.history['val_visibility_loss'].append(val_metrics['visibility_loss'])
        self.history['learning_rate'].append(
            self.optimizer.param_groups[0]['lr']
        )
        self.history['avg_ADE'].append(val_metrics['avg_ADE'])

        # Theo dõi tham số tự học của Focal Loss nếu có
        if self.config.loss.learnable_focal_params:
            # Lấy giá trị gamma từ visibility_criterion của mô hình
            focal_loss = self.model.loss_fn.visibility_criterion
            gamma_value = focal_loss.gamma.item()
            self.history.setdefault('focal_gamma', []).append(gamma_value)

            if hasattr(focal_loss, 'alpha') and focal_loss.alpha is not None:
                if focal_loss.alpha.dim() == 0:
                    alpha_value = focal_loss.alpha.item()
                    self.history.setdefault('focal_alpha', []).append(alpha_value)
                else:
                    alpha_value = focal_loss.alpha.mean().item()
                    self.history.setdefault('focal_alpha', []).append(alpha_value)

        for thresh in self.config.pck_thresholds:
            self.history[f'pck_{thresh}'].append(
                val_metrics[f'pck_{thresh}']
            )

    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics once, avoiding duplication"""
        if prefix:
            logging.info(f"\n{prefix}:")

        if 'loss' in metrics:
            logging.info(f"Train loss: {metrics['loss']:.4f}")
        if 'val_loss' in metrics:
            logging.info(f"Validation loss: {metrics['val_loss']:.4f}")
        if 'avg_ADE' in metrics:
            logging.info(f"Average Distance Error: {metrics['avg_ADE']:.4f}")

        if 'focal_gamma' in metrics:
            logging.info(f"Focal gamma: {metrics['focal_gamma']:.4f}")
        if 'focal_alpha' in metrics:
            logging.info(f"Focal alpha: {metrics['focal_alpha']:.4f}")

        pck_metrics = {k: v for k, v in metrics.items() if k.startswith('pck_')}
        if pck_metrics:
            if prefix:
                logging.info("\nPCK metrics:")
            for k, v in pck_metrics.items():
                thresh = k.split('_')[1]
                logging.info(f"PCK@{thresh}: {v:.4f}")

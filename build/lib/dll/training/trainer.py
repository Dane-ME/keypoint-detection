"""
Training utilities for keypoint detection model - Refactored Version
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
from torch.utils.data import DataLoader
import logging
from collections import defaultdict

from dll.utils.metric import create_base_metrics
from dll.utils.device_manager import get_device_manager, move_batch_to_device
from dll.configs.training_config import TrainingConfig


class MetricsAccumulator:
    """Efficient metrics accumulation with automatic averaging."""

    def __init__(self, metric_names: List[str]):
        self.metrics = {name: 0.0 for name in metric_names}
        self.count = 0

    def update(self, batch_metrics: Dict[str, float]):
        """Update metrics with batch values."""
        for name, value in batch_metrics.items():
            if name in self.metrics:
                self.metrics[name] += value
        self.count += 1

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        if self.count == 0:
            return self.metrics.copy()
        return {name: value / self.count for name, value in self.metrics.items()}


class TrainingHistory:
    """Efficient training history management."""

    def __init__(self, pck_thresholds: List[float]):
        self.data = defaultdict(list)
        self.pck_thresholds = pck_thresholds

    def update(self, epoch: int, train_metrics: Dict[str, float],
               val_metrics: Dict[str, float], learning_rate: float,
               focal_params: Optional[Dict[str, float]] = None):
        """Update history with new epoch data."""
        self.data['epoch'].append(epoch)
        self.data['learning_rate'].append(learning_rate)

        # Update train metrics
        for metric_name, value in train_metrics.items():
            self.data[f'train_{metric_name}'].append(value)

        # Update validation metrics
        for metric_name, value in val_metrics.items():
            if metric_name.startswith('pck_') or metric_name in ['loss', 'keypoint_loss', 'visibility_loss', 'avg_ADE']:
                prefix = 'val_' if metric_name in ['keypoint_loss', 'visibility_loss'] else ''
                self.data[f'{prefix}{metric_name}'].append(value)

        # Update focal loss parameters if available
        if focal_params:
            for param_name, value in focal_params.items():
                self.data[param_name].append(value)

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        return self.data[metric_name][-1] if self.data[metric_name] else None

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for compatibility."""
        return dict(self.data)

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

        self._setup_device_and_model(device, use_amp)
        self._setup_training_components()
        self._initialize_history()
        self._setup_logging()

    def _setup_device_and_model(self, device: torch.device, use_amp: bool):
        """Setup device management and model placement."""
        try:
            device_manager = get_device_manager()
            self.device = device_manager.device
            self.model = device_manager.to_device(self.model)
            self.use_amp = device_manager.mixed_precision
        except RuntimeError:
            # Fallback if device manager not initialized
            self.device = device
            self.model = self.model.to(self.device)
            self.use_amp = use_amp

    def _setup_training_components(self):
        """Initialize optimizer, scheduler, and AMP scaler."""
        self.optimizer = self._create_optimizer()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Setup learning rate scheduler with configuration."""
        scheduler_config = self.config.lr_scheduler

        # Create scheduler with explicit type conversion
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=str(scheduler_config.mode),  # Ensure string
            factor=float(scheduler_config.factor),  # Ensure float
            patience=int(scheduler_config.patience),  # Ensure int
            min_lr=float(scheduler_config.min_lr),  # Ensure float
            threshold=float(scheduler_config.threshold)  # Ensure float
        )
        self.scheduler_metric = str(scheduler_config.metric)  # Ensure string

        # Initialize scheduler state to avoid comparison issues
        self._scheduler_initialized = False

    def _initialize_history(self):
        """Initialize training history with efficient management."""
        self.history = TrainingHistory(self.config.pck_thresholds)

    def _setup_logging(self):
        """Setup logging configuration."""
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

    def _forward_pass(self, batch: Dict[str, torch.Tensor], training: bool = True) -> Dict[str, torch.Tensor]:
        """Unified forward pass for training and validation."""
        batch = move_batch_to_device(batch)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model(batch)
        else:
            outputs = self.model(batch)

        if training and 'loss' in outputs:
            self._backward_pass(outputs['loss'])

        return outputs

    def _backward_pass(self, loss: torch.Tensor):
        """Handle backward pass with optional AMP scaling."""
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _create_progress_bar(self, dataloader: DataLoader, desc: str, leave: bool = True) -> tqdm:
        """Create optimized progress bar with memory efficiency."""
        return tqdm(
            dataloader,
            desc=desc,
            leave=leave,
            dynamic_ncols=True,
            miniters=max(1, len(dataloader) // 100),  # Update every 1%
            maxinterval=10.0  # Max 10 seconds between updates
        )

    def _initialize_validation_metrics(self) -> Dict[str, float]:
        """Initialize validation metrics based on configuration."""
        metrics = {'avg_ADE': 0.0}

        # Dynamic PCK metrics from config
        for threshold in self.config.pck_thresholds:
            metrics[f'pck_{threshold}'] = 0.0

        return metrics

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

                # Safely extract keypoint and visibility losses
                keypoint_loss = outputs.get('keypoint_loss', 0)
                if hasattr(keypoint_loss, 'item'):
                    keypoint_loss = keypoint_loss.item()
                epoch_metrics['keypoint_loss'] += float(keypoint_loss)

                visibility_loss = outputs.get('visibility_loss', 0)
                if hasattr(visibility_loss, 'item'):
                    visibility_loss = visibility_loss.item()
                epoch_metrics['visibility_loss'] += float(visibility_loss)

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

        # Add additional validation metrics dynamically from config
        epoch_metrics.update({'avg_ADE': 0.0})

        # Add PCK metrics based on config thresholds
        for threshold in self.config.pck_thresholds:
            epoch_metrics[f'pck_{threshold}'] = 0.0

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

                    # Safely extract keypoint and visibility losses
                    keypoint_loss = outputs.get('keypoint_loss', 0)
                    if hasattr(keypoint_loss, 'item'):
                        keypoint_loss = keypoint_loss.item()
                    epoch_metrics['keypoint_loss'] += float(keypoint_loss)

                    visibility_loss = outputs.get('visibility_loss', 0)
                    if hasattr(visibility_loss, 'item'):
                        visibility_loss = visibility_loss.item()
                    epoch_metrics['visibility_loss'] += float(visibility_loss)

                    epoch_metrics['num_batches'] += 1

                    # Calculate additional validation metrics (PCK, ADE)
                    if 'keypoints' in outputs and 'keypoints' in batch:
                        additional_metrics = self._calculate_validation_metrics(outputs, batch)
                        for key, value in additional_metrics.items():
                            epoch_metrics[key] += value

                    # Update progress bar with dynamic PCK threshold
                    postfix_dict = {
                        'loss': f"{outputs['loss'].item():.4f}",
                        'ADE': f"{epoch_metrics['avg_ADE'] / max(epoch_metrics['num_batches'], 1):.4f}"
                    }

                    # Add the largest PCK threshold for display
                    if self.config.pck_thresholds:
                        largest_thresh = max(self.config.pck_thresholds)
                        pck_key = f'pck_{largest_thresh}'
                        if pck_key in epoch_metrics:
                            postfix_dict[f'PCK@{largest_thresh}'] = f"{epoch_metrics[pck_key] / max(epoch_metrics['num_batches'], 1):.4f}"

                    progress_bar.set_postfix(postfix_dict)
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
                return self._get_default_metrics()

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
            metrics = self._get_default_metrics()

        return metrics

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics based on config."""
        metrics = {'avg_ADE': 0.0}
        for threshold in self.config.pck_thresholds:
            metrics[f'pck_{threshold}'] = 0.0
        return metrics

    def _get_scheduler_metric(self, val_metrics: Dict[str, float]) -> float:
        """Extract and validate scheduler metric."""
        if self.scheduler_metric not in val_metrics:
            logging.warning(f"Metric '{self.scheduler_metric}' not found. Using 'loss' as fallback.")
            return self._safe_float_conversion(val_metrics.get('loss', 0.0))

        metric_value = val_metrics[self.scheduler_metric]
        return self._safe_float_conversion(metric_value)

    def _safe_float_conversion(self, value) -> float:
        """Safely convert any value to float."""
        try:
            # Handle tensor values
            if hasattr(value, 'item'):
                return float(value.item())
            # Handle numpy values
            elif hasattr(value, 'dtype'):
                return float(value)
            # Handle string values
            elif isinstance(value, str):
                return float(value)
            # Handle regular numbers
            else:
                return float(value)
        except (ValueError, TypeError, AttributeError) as e:
            logging.warning(f"Could not convert {value} (type: {type(value)}) to float: {e}")
            return 0.0

    def _debug_metrics_types(self, val_metrics: Dict[str, float]) -> None:
        """Debug method to log metrics types."""
        logging.info("=== METRICS DEBUG INFO ===")
        for key, value in val_metrics.items():
            logging.info(f"  {key}: {value} (type: {type(value)})")
        logging.info("=== END DEBUG INFO ===")

    def _safe_scheduler_step(self, metric_value: float) -> None:
        """Safely step the scheduler with proper error handling."""
        try:
            # Ensure metric_value is a proper float
            metric_value = float(metric_value)

            # For first call, initialize scheduler properly
            if not self._scheduler_initialized:
                # Reset scheduler internal state
                self.scheduler.best = None
                self.scheduler.num_bad_epochs = 0
                self._scheduler_initialized = True
                logging.info("Scheduler initialized for first use")

            # Log scheduler state before step
            logging.info(f"Scheduler step with value: {metric_value} (type: {type(metric_value)})")
            logging.info(f"Scheduler best: {getattr(self.scheduler, 'best', 'None')}")
            logging.info(f"Scheduler num_bad_epochs: {getattr(self.scheduler, 'num_bad_epochs', 'None')}")

            # Call scheduler step
            self.scheduler.step(metric_value)

        except Exception as e:
            logging.error(f"Scheduler step failed with metric {metric_value}: {e}")
            # Try to reset scheduler and step again
            try:
                self.scheduler.best = float('inf') if self.scheduler.mode == 'min' else float('-inf')
                self.scheduler.num_bad_epochs = 0
                self.scheduler.step(float(metric_value))
                logging.info("Scheduler reset and stepped successfully")
            except Exception as reset_error:
                logging.error(f"Scheduler reset also failed: {reset_error}")
                raise

    def _update_learning_rate(self, val_metrics: Dict[str, float]) -> None:
        """Safely update learning rate with proper error handling."""
        try:
            # Debug metrics types if needed
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                self._debug_metrics_types(val_metrics)

            metric_value = self._get_scheduler_metric(val_metrics)
            logging.info(f"Scheduler metric '{self.scheduler_metric}' value: {metric_value} (type: {type(metric_value)})")

            # Use safe scheduler step
            self._safe_scheduler_step(metric_value)
            self._log_learning_rate_info()

        except Exception as e:
            logging.error(f"Learning rate update failed: {e}")
            # Log debug info on error
            self._debug_metrics_types(val_metrics)
            self._fallback_scheduler_update(val_metrics)

    def _log_learning_rate_info(self) -> None:
        """Log learning rate information."""
        current_lr = self.optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.2e}")
        if hasattr(self.scheduler, 'num_bad_epochs'):
            logging.info(f"Epochs without improvement: {self.scheduler.num_bad_epochs}/{self.scheduler.patience}")

    def _fallback_scheduler_update(self, val_metrics: Dict[str, float]) -> None:
        """Fallback scheduler update using loss metric."""
        try:
            fallback_value = self._safe_float_conversion(val_metrics.get('loss', 0.0))
            self._safe_scheduler_step(fallback_value)
            logging.info(f"Using fallback metric 'loss': {fallback_value}")
        except Exception as fallback_error:
            logging.error(f"Fallback also failed: {fallback_error}")
            # Last resort: use a default value
            try:
                self._safe_scheduler_step(1.0)
                logging.info("Using default value 1.0 for scheduler")
            except Exception as final_error:
                logging.error(f"Even default value failed: {final_error}")
                # Ultimate fallback: recreate scheduler
                try:
                    self._setup_scheduler()
                    self._safe_scheduler_step(1.0)
                    logging.info("Recreated scheduler and used default value")
                except Exception as ultimate_error:
                    logging.error(f"Ultimate fallback failed: {ultimate_error}")

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

            # Update learning rate - only call scheduler once with the configured metric
            self._update_learning_rate(val_metrics)

            # Update history
            self._update_history(epoch, train_metrics, val_metrics)

            # Keep track of best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)

        # Final check for focal loss parameters
        self._log_final_focal_params()

        # Log final metrics
        self._log_final_metrics()

        return self.model, self.history.to_dict()

    def _log_final_focal_params(self):
        """Log final focal loss parameters if learnable."""
        if self.config.loss.learnable_focal_params:
            try:
                focal_loss = self.model.loss_fn.visibility_criterion
                final_gamma = focal_loss.gamma.item()
                logging.info(f"\nFinal focal gamma: {final_gamma:.4f}")

                if hasattr(focal_loss, 'alpha') and focal_loss.alpha is not None:
                    if focal_loss.alpha.dim() == 0:
                        final_alpha = focal_loss.alpha.item()
                    else:
                        final_alpha = focal_loss.alpha.mean().item()
                    logging.info(f"Final focal alpha: {final_alpha:.4f}")
            except Exception as e:
                logging.warning(f"Could not log focal parameters: {e}")

    def _log_final_metrics(self):
        """Log final training metrics."""
        try:
            latest_train_loss = self.history.get_latest('train_loss')
            latest_val_loss = self.history.get_latest('loss')
            latest_ade = self.history.get_latest('avg_ADE')

            if latest_train_loss is not None:
                logging.info("\nFinal metrics:")
                logging.info(f"Train loss: {latest_train_loss:.4f}")
                logging.info(f"Validation loss: {latest_val_loss:.4f}")
                logging.info(f"Average Distance Error: {latest_ade:.4f}")

                for thresh in self.config.pck_thresholds:
                    pck_value = self.history.get_latest(f'pck_{thresh}')
                    if pck_value is not None:
                        logging.info(f"PCK@{thresh}: {pck_value:.4f}")
        except Exception as e:
            logging.warning(f"Could not log final metrics: {e}")

    def _update_history(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Update history with training and validation metrics."""
        learning_rate = self.optimizer.param_groups[0]['lr']

        # Get focal parameters if available
        focal_params = None
        if self.config.loss.learnable_focal_params:
            focal_params = self._extract_focal_params()

        # Update history using the efficient TrainingHistory class
        self.history.update(epoch, train_metrics, val_metrics, learning_rate, focal_params)

    def _extract_focal_params(self) -> Optional[Dict[str, float]]:
        """Extract focal loss parameters for history tracking."""
        try:
            focal_loss = self.model.loss_fn.visibility_criterion
            params = {'focal_gamma': focal_loss.gamma.item()}

            if hasattr(focal_loss, 'alpha') and focal_loss.alpha is not None:
                if focal_loss.alpha.dim() == 0:
                    params['focal_alpha'] = focal_loss.alpha.item()
                else:
                    params['focal_alpha'] = focal_loss.alpha.mean().item()

            return params
        except Exception as e:
            logging.warning(f"Could not extract focal parameters: {e}")
            return None

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

"""
Centralized device management for the keypoint detection project.
This module provides a singleton DeviceManager to ensure all tensors and models
are consistently placed on the same device throughout the application.
"""

import torch
import logging
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Configuration for device management."""
    type: str = 'auto'  # 'auto', 'cuda', 'cpu', or specific like 'cuda:0'
    force_cpu: bool = False
    mixed_precision: bool = True
    pin_memory: bool = True


class DeviceManager:
    """
    Singleton class for centralized device management.
    
    This ensures all components use the same device and provides utilities
    for moving tensors and models to the correct device.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._device = None
            self._config = None
            self._mixed_precision = False
            self._pin_memory = False
            DeviceManager._initialized = True
    
    def initialize(self, config: Union[DeviceConfig, Dict[str, Any]]):
        """
        Initialize the device manager with configuration.
        
        Args:
            config: DeviceConfig object or dictionary with device settings
        """
        if isinstance(config, dict):
            config = DeviceConfig(**config)
        
        self._config = config
        self._device = self._determine_device(config)
        self._mixed_precision = config.mixed_precision and self._device.type == 'cuda'
        self._pin_memory = config.pin_memory and self._device.type == 'cuda'
        
        logger.info(f"DeviceManager initialized with device: {self._device}")
        logger.info(f"Mixed precision: {self._mixed_precision}")
        logger.info(f"Pin memory: {self._pin_memory}")
    
    def _determine_device(self, config: DeviceConfig) -> torch.device:
        """Determine the appropriate device based on configuration."""
        if config.force_cpu:
            return torch.device('cpu')
        
        if config.type == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = torch.device('cpu')
                logger.info("CUDA not available. Using CPU.")
            return device
        
        elif config.type == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
        
        elif config.type == 'cpu':
            return torch.device('cpu')
        
        else:
            # Specific device like 'cuda:0'
            try:
                device = torch.device(config.type)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    logger.warning(f"Device {config.type} requested but CUDA not available. Using CPU.")
                    return torch.device('cpu')
                return device
            except Exception as e:
                logger.error(f"Invalid device specification: {config.type}. Error: {e}")
                return torch.device('cpu')
    
    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            raise RuntimeError("DeviceManager not initialized. Call initialize() first.")
        return self._device
    
    @property
    def mixed_precision(self) -> bool:
        """Whether mixed precision is enabled."""
        return self._mixed_precision
    
    @property
    def pin_memory(self) -> bool:
        """Whether pin memory is enabled."""
        return self._pin_memory
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """
        Move tensor or model to the managed device.
        
        Args:
            tensor_or_model: Tensor or model to move
            
        Returns:
            Tensor or model on the correct device
        """
        if self._device is None:
            raise RuntimeError("DeviceManager not initialized. Call initialize() first.")
        
        return tensor_or_model.to(self._device)
    
    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move a batch dictionary to the managed device.
        
        Args:
            batch: Dictionary containing tensors and other data
            
        Returns:
            Batch with tensors moved to device
        """
        if self._device is None:
            raise RuntimeError("DeviceManager not initialized. Call initialize() first.")
        
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self._device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                # Handle list of tensors
                device_batch[key] = [tensor.to(self._device) for tensor in value]
            else:
                # Keep non-tensor values as is
                device_batch[key] = value
        
        return device_batch
    
    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """
        Create a tensor on the managed device.
        
        Args:
            *args, **kwargs: Arguments passed to torch.tensor()
            
        Returns:
            Tensor on the managed device
        """
        if self._device is None:
            raise RuntimeError("DeviceManager not initialized. Call initialize() first.")
        
        # Remove device from kwargs if present to avoid conflicts
        kwargs.pop('device', None)
        tensor = torch.tensor(*args, **kwargs)
        return tensor.to(self._device)
    
    def zeros(self, *args, **kwargs) -> torch.Tensor:
        """Create zeros tensor on managed device."""
        kwargs['device'] = self._device
        return torch.zeros(*args, **kwargs)
    
    def ones(self, *args, **kwargs) -> torch.Tensor:
        """Create ones tensor on managed device."""
        kwargs['device'] = self._device
        return torch.ones(*args, **kwargs)
    
    def randn(self, *args, **kwargs) -> torch.Tensor:
        """Create random normal tensor on managed device."""
        kwargs['device'] = self._device
        return torch.randn(*args, **kwargs)
    
    def empty(self, *args, **kwargs) -> torch.Tensor:
        """Create empty tensor on managed device."""
        kwargs['device'] = self._device
        return torch.empty(*args, **kwargs)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        info = {
            'device': str(self._device),
            'mixed_precision': self._mixed_precision,
            'pin_memory': self._pin_memory
        }
        
        if self._device.type == 'cuda':
            info.update({
                'cuda_available': True,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_current_device': torch.cuda.current_device(),
                'cuda_device_name': torch.cuda.get_device_name(),
                'cuda_memory_allocated': torch.cuda.memory_allocated(),
                'cuda_memory_reserved': torch.cuda.memory_reserved(),
                'cuda_max_memory_allocated': torch.cuda.max_memory_allocated(),
            })
        else:
            info['cuda_available'] = False
        
        return info


# Global instance
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    return device_manager


def initialize_device_manager(config: Union[DeviceConfig, Dict[str, Any]]):
    """Initialize the global device manager."""
    device_manager.initialize(config)


# Convenience functions
def get_device() -> torch.device:
    """Get the current managed device."""
    return device_manager.device


def to_device(tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    """Move tensor or model to managed device."""
    return device_manager.to_device(tensor_or_model)


def move_batch_to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Move batch to managed device."""
    return device_manager.move_batch_to_device(batch)

"""
Base configuration class
"""
from dataclasses import dataclass
from typing import Dict
import yaml
import os
from pathlib import Path

@dataclass
class BaseConfig:
    @classmethod
    def get_config_path(cls) -> str:
        """Get default config path from environment or find from project root"""
        if path := os.getenv('DLL_CONFIG_PATH'):
            return path
            
        current_dir = Path(__file__).resolve().parent
        while current_dir.name and not (current_dir / 'setup.py').exists():
            current_dir = current_dir.parent
            
        default_path = str(current_dir / 'configs' / 'default_config.yaml')
        return default_path

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_default(cls, key: str = None) -> 'BaseConfig':
        """Quick load from default config file with optional key."""
        config_path = cls.get_config_path()
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            # Update path from config if exists
            if 'paths' in config_dict:
                config_path = config_dict['paths'].get('default_config', config_path)
            if key:
                for k in key.split('.'):
                    config_dict = config_dict[k]
            return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    def validate(self) -> None:
        """Validate config values."""
        pass

"""
Simplified training script with default paths
"""
from train import main
from dll.configs import BaseConfig
import yaml

if __name__ == '__main__':
    import sys
    
    # Load paths from config
    config = yaml.safe_load(open(BaseConfig.get_config_path()))['paths']
    
    sys.argv.extend([
        '--config', config['default_config'],
        '--data_dir', config['data_dir'],
        '--output_dir', config['output_dir']
    ])
    
    main()

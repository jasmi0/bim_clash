"""
Utility functions for the BIM Clash Detection project
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file: str = "logs/bim_clash.log"):
    """Set up logging configuration"""
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/uploaded_files",
        "data/processed",
        "models/saved_models",
        "logs"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

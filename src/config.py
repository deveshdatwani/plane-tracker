"""
Configuration loader for plane-tracker.
Loads settings from config.yaml and provides default fallbacks.
"""

from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

# Try to import yaml, fallback to defaults if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ModuleNotFoundError:
    YAML_AVAILABLE = False

# Default configuration
DEFAULTS = {
    "detection": {
        "model_path": "yolov8n-seg.pt",
        "confidence_threshold": 0.45,
        "iou_threshold": 0.4,
        "verbose": False,
    },
    "tracker": {
        "iou_threshold": 0.2,
        "max_age": 30,
        "min_hits": 10,
    },
    "ocr": {
        "enabled": True,
        "interval": 10,
    },
    "hangar": {
        "cooldown_frames": 30,
        "iou_threshold": 0.01,
        "flash_on_event": True,
        "flash_color": [140, 200, 220],
        "flash_rate": 4,
    },
    "debug": {
        "level": 2,
        "show_frame_id": True,
        "show_detections": True,
        "show_tracks": True,
        "show_latency": True,
        "show_fps": True,
        "processing_debug": False,
        "show_ground_truth": False,
        "metrics_iou_threshold": 0.5,
    },
    "visualization": {
        "track_color_active": [200, 220, 100],
        "track_color_lost": [120, 120, 200],
        "hangar_boundary_color": [60, 60, 60],
        "tail_number_color": [0, 255, 255],
        "gt_color": [255, 0, 255],
        "font_scale": 0.45,
        "font_thickness": 1,
    },
}

_config: Dict[str, Any] = {}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file, merging with defaults."""
    global _config
    
    config = DEFAULTS.copy()
    
    path = Path(config_path)
    if path.exists() and YAML_AVAILABLE:
        try:
            with open(path, "r") as f:
                user_config = yaml.safe_load(f) or {}
            # Deep merge user config into defaults
            for section, values in user_config.items():
                if section in config and isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
    elif not YAML_AVAILABLE:
        logger.warning("PyYAML not installed. Using default configuration. Install via `pip install pyyaml`")
    else:
        logger.info(f"Config file {config_path} not found. Using defaults.")
    
    _config = config
    return config


def get_config() -> Dict[str, Any]:
    """Get the current configuration. Loads from file if not yet loaded."""
    if not _config:
        load_config()
    return _config


def get(section: str, key: str, default: Any = None) -> Any:
    """Get a specific config value."""
    cfg = get_config()
    return cfg.get(section, {}).get(key, default)

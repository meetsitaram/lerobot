"""Configuration for synthetic camera views."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..configs import CameraConfig, ColorMode

__all__ = ["SyntheticCameraConfig", "ColorMode"]


@CameraConfig.register_subclass("synthetic")
@dataclass
class SyntheticCameraConfig(CameraConfig):
    """
    Configuration for synthetic camera that generates clean orthographic views.
    
    Synthetic cameras use multi-camera detection and 3D triangulation to create
    clean, simulator-like views with the robot arm visualization.
    
    Requires the synthetic-camera package to be installed:
        pip install -e /path/to/synthetic-camera
    
    Example:
        ```python
        from lerobot.cameras.synthetic import SyntheticCameraConfig
        
        # Detection prompt is loaded from scene_config.yaml by default
        config = SyntheticCameraConfig(
            view="front",
            config_dir="/path/to/synthetic-camera/config",
        )
        ```
    
    For use in robot configs:
        ```python
        robot_config = SOFollowerRobotConfig(
            port="/dev/ttyUSB0",
            use_degrees=True,  # Required for synthetic cameras!
            cameras={
                "front": SyntheticCameraConfig(view="front", config_dir="..."),
                "top": SyntheticCameraConfig(view="top", config_dir="..."),
                "side": SyntheticCameraConfig(view="side", config_dir="..."),
            },
        )
        ```
    
    Note: Synthetic cameras need access to robot joint positions for rendering.
    This is handled automatically when using SOFollower or similar robots.
    """
    
    # View to render: "front", "top", or "side"
    view: str = "front"
    
    # Path to synthetic-camera config directory (contains calibration, scene config, URDF)
    config_dir: Optional[str] = None
    
    # Frame rate
    fps: Optional[int] = 30
    
    # Output resolution
    width: Optional[int] = 400
    height: Optional[int] = 300
    
    # Color mode for output
    color_mode: ColorMode = ColorMode.RGB
    
    # Real camera IDs for multi-camera capture
    camera_ids: Optional[List[int]] = None
    
    # Real camera resolution
    camera_resolution: Tuple[int, int] = (1280, 720)
    
    # Detection settings (None = use scene_config.yaml defaults)
    detection_prompt: Optional[str] = None  # If None, uses scene_config.yaml
    detection_confidence: Optional[float] = None  # If None, uses scene_config.yaml
    detection_device: Optional[str] = None  # None for auto (CUDA if available)
    
    # Robot offset from calibration origin [x, y, z] in centimeters
    robot_offset_cm: Tuple[float, float, float] = (0.0, 6.0, 40.0)
    
    # Camera tilt compensation in degrees
    camera_tilt_deg: float = 0.0
    
    # Orthographic view scale (meters visible)
    ortho_scale: float = 0.4
    
    # Rate limit for joint position reads (Hz)
    joint_read_rate_hz: float = 30.0
    
    def __post_init__(self) -> None:
        if self.view not in ("front", "top", "side"):
            raise ValueError(f"view must be 'front', 'top', or 'side', got '{self.view}'")
        
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"color_mode must be RGB or BGR, got {self.color_mode}")


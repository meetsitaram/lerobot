"""
Synthetic camera implementation for LeRobot.

This camera generates clean, orthographic views by:
1. Capturing from multiple real USB cameras
2. Running YOLO-World object detection
3. Triangulating objects to 3D
4. Rendering synthetic views with the robot arm

Requires the synthetic-camera package to be installed.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..camera import Camera
from ..configs import ColorMode
from .configuration_synthetic import SyntheticCameraConfig

logger = logging.getLogger(__name__)


class SyntheticCamera(Camera):
    """
    Synthetic camera that outputs rendered orthographic views.
    
    This camera wraps the synthetic-camera package's functionality,
    providing clean views generated from multi-camera detection.
    
    Architecture:
    - Background thread: captures frames + YOLO detection (slow)
    - Main thread: rendering (OpenGL requires main thread)
    
    The camera needs a robot reference to render joint positions.
    This is injected via set_robot() after the robot is created.
    """
    
    # Class-level render timing
    _last_render_time: float = 0.0
    _render_interval: float = 1.0 / 15.0
    _render_lock = threading.Lock()
    
    def __init__(self, config: SyntheticCameraConfig):
        """
        Initialize synthetic camera.
        
        Args:
            config: Camera configuration
        """
        super().__init__(config)
        self.config = config
        
        # Settings
        self.view = config.view
        self.fps = config.fps or 30
        self.width = config.width or 400
        self.height = config.height or 300
        self.color_mode = config.color_mode
        
        # Robot reference (set via set_robot())
        self._robot: Any = None
        
        # Renderer (from synthetic-camera package)
        self._renderer: Any = None
        self._is_connected = False
        
        # Frame cache
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
    
    def set_robot(self, robot: Any) -> None:
        """
        Set the robot reference for joint state access.
        
        This must be called before connect() to enable robot arm rendering.
        For SOFollower robots, this is called automatically.
        
        Args:
            robot: LeRobot robot instance
        """
        self._robot = robot
    
    def set_joint_positions(self, action: Dict[str, float], in_degrees: bool = True,
                            normalized: bool = False) -> None:
        """
        Set joint positions for rendering (bypasses robot bus reads).
        
        Call this in your teleop loop to update the rendered arm position
        without reading from the motor bus.
        
        Args:
            action: Dict of joint positions (e.g., {'shoulder_pan.pos': 45.0})
            in_degrees: If True, values are in degrees (only used if normalized=False)
            normalized: If True, values are in -100 to 100 range (LeRobot default)
                       This is recommended for policy-transferable datasets.
        """
        if self._renderer is not None:
            self._renderer.set_joint_positions(action, in_degrees=in_degrees, 
                                              normalized=normalized)
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """Return available synthetic camera views."""
        return [
            {"view": "front", "description": "Front orthographic view"},
            {"view": "top", "description": "Top-down orthographic view"},
            {"view": "side", "description": "Side orthographic view"},
        ]
    
    def connect(self, warmup: bool = True) -> None:
        """
        Connect the synthetic camera.
        
        This initializes the shared renderer and starts the background
        capture/detection thread.
        """
        if self._is_connected:
            logger.warning(f"SyntheticCamera({self.view}) already connected")
            return
        
        # Import from synthetic-camera package
        try:
            from synthetic_camera.lerobot.view_renderer import (
                SyntheticViewRenderer,
                SyntheticViewConfig,
            )
        except ImportError as e:
            raise ImportError(
                "synthetic-camera package not found. Install it with:\n"
                "  pip install -e /path/to/synthetic-camera"
            ) from e
        
        logger.info(f"Connecting SyntheticCamera({self.view})...")
        
        # Build renderer config
        if self.config.config_dir:
            renderer_config = SyntheticViewConfig.from_yaml(self.config.config_dir)
            renderer_config.camera_ids = self.config.camera_ids or renderer_config.camera_ids
            renderer_config.camera_width = self.config.camera_resolution[0]
            renderer_config.camera_height = self.config.camera_resolution[1]
            renderer_config.camera_fps = self.fps
            renderer_config.detection_prompt = self.config.detection_prompt
            renderer_config.detection_confidence = self.config.detection_confidence
            renderer_config.detection_device = self.config.detection_device
            renderer_config.view_width = self.width
            renderer_config.view_height = self.height
            renderer_config.ortho_scale = self.config.ortho_scale
            renderer_config.joint_read_rate_hz = self.config.joint_read_rate_hz
        else:
            renderer_config = SyntheticViewConfig(
                camera_ids=self.config.camera_ids or [6, 4, 2],
                camera_width=self.config.camera_resolution[0],
                camera_height=self.config.camera_resolution[1],
                camera_fps=self.fps,
                detection_prompt=self.config.detection_prompt,
                detection_confidence=self.config.detection_confidence,
                detection_device=self.config.detection_device,
                robot_x_cm=self.config.robot_offset_cm[0],
                robot_y_cm=self.config.robot_offset_cm[1],
                robot_z_cm=self.config.robot_offset_cm[2],
                camera_tilt_deg=self.config.camera_tilt_deg,
                view_width=self.width,
                view_height=self.height,
                ortho_scale=self.config.ortho_scale,
                joint_read_rate_hz=self.config.joint_read_rate_hz,
            )
        
        # Get shared renderer
        self._renderer = SyntheticViewRenderer.get_instance(
            renderer_config,
            self._robot,
        )
        
        self._is_connected = True
        
        # Warmup
        if warmup:
            time.sleep(0.2)
            self._renderer.render_if_needed()
            self._last_frame = self._renderer.get_view(self.view)
        
        logger.info(f"SyntheticCamera({self.view}) connected")
    
    def read(self, color_mode: Optional[ColorMode] = None) -> NDArray[Any]:
        """
        Read a frame from the synthetic camera.
        
        Returns the cached synthetic view. The background thread handles
        capture/detection, and rendering happens here (main thread).
        """
        if not self._is_connected:
            raise RuntimeError(f"SyntheticCamera({self.view}) not connected")
        
        # Render if needed (rate-limited, main thread only)
        current_time = time.time()
        with SyntheticCamera._render_lock:
            if current_time - SyntheticCamera._last_render_time >= SyntheticCamera._render_interval:
                self._renderer.render_if_needed()
                SyntheticCamera._last_render_time = current_time
        
        # Get cached view
        frame = self._renderer.get_view(self.view)
        
        # Color mode conversion
        output_mode = color_mode or self.color_mode
        if output_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with self._frame_lock:
            self._last_frame = frame.copy()
        
        return frame
    
    def async_read(self, timeout_ms: float = 200.0) -> NDArray[Any]:
        """Async read - same as read() for synthetic cameras."""
        return self.read()
    
    def disconnect(self) -> None:
        """Disconnect and release resources."""
        if not self._is_connected:
            return
        
        logger.info(f"Disconnecting SyntheticCamera({self.view})...")
        
        try:
            from synthetic_camera.lerobot.view_renderer import SyntheticViewRenderer
            SyntheticViewRenderer.release_instance()
        except ImportError:
            pass
        
        self._renderer = None
        self._is_connected = False
        logger.info(f"SyntheticCamera({self.view}) disconnected")
    
    def __repr__(self) -> str:
        return f"SyntheticCamera(view='{self.view}', connected={self._is_connected})"


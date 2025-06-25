#!/usr/bin/env python3
"""
AR Engine for Medical Augmented Reality Interface
Handles AR visualization, tracking, and medical data overlay
"""

import numpy as np
import cv2
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARMode(Enum):
    """AR visualization modes"""
    OVERLAY = "overlay"
    ANNOTATION = "annotation"
    MEASUREMENT = "measurement"
    COMPARISON = "comparison"
    GUIDANCE = "guidance"

@dataclass
class ARAnnotation:
    """AR annotation data structure"""
    id: str
    position: Tuple[float, float, float]  # 3D position
    text: str
    color: Tuple[int, int, int] = (255, 255, 255)
    size: float = 1.0
    visible: bool = True
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ARMeasurement:
    """AR measurement data structure"""
    id: str
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    measurement_type: str  # "distance", "angle", "area", "volume"
    value: float
    unit: str
    accuracy: float = 0.95
    visible: bool = True
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class AREngine:
    """
    AR Engine for medical applications
    Handles tracking, rendering, and interaction with medical data
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 resolution: Tuple[int, int] = (1280, 720),
                 fps: int = 30):
        """
        Initialize AR Engine
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        
        # Camera and tracking
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_calibrated = False
        
        # AR state
        self.is_running = False
        self.current_mode = ARMode.OVERLAY
        self.tracking_enabled = True
        
        # AR objects
        self.annotations: Dict[str, ARAnnotation] = {}
        self.measurements: Dict[str, ARMeasurement] = {}
        self.medical_overlays: Dict[str, Any] = {}
        
        # Tracking data
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.tracking_confidence = 0.0
        
        # Threading
        self.frame_thread = None
        self.tracking_thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
        # Callbacks
        self.frame_callback = None
        self.tracking_callback = None
        
        logger.info(f"AR Engine initialized with resolution {resolution} at {fps} FPS")
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} at {actual_fps} FPS")
            
            # Load camera calibration if available
            self.load_camera_calibration()
            
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            return False
    
    def load_camera_calibration(self, calibration_file: str = "camera_calibration.json"):
        """Load camera calibration parameters"""
        try:
            with open(calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            self.camera_matrix = np.array(calib_data['camera_matrix'])
            self.dist_coeffs = np.array(calib_data['dist_coeffs'])
            self.is_calibrated = True
            
            logger.info("Camera calibration loaded successfully")
            
        except FileNotFoundError:
            logger.warning("Camera calibration file not found, using default parameters")
            self.setup_default_calibration()
        except Exception as e:
            logger.error(f"Failed to load camera calibration: {str(e)}")
            self.setup_default_calibration()
    
    def setup_default_calibration(self):
        """Setup default camera calibration parameters"""
        # Default camera matrix for typical webcam
        fx = fy = self.resolution[0]  # Approximate focal length
        cx, cy = self.resolution[0] / 2, self.resolution[1] / 2  # Principal point
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        self.is_calibrated = False
        
        logger.info("Using default camera calibration")
    
    def start_ar_session(self) -> bool:
        """Start AR session with camera and tracking"""
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        
        # Start frame capture thread
        self.frame_thread = threading.Thread(target=self._frame_capture_loop)
        self.frame_thread.daemon = True
        self.frame_thread.start()
        
        # Start tracking thread
        if self.tracking_enabled:
            self.tracking_thread = threading.Thread(target=self._tracking_loop)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
        
        logger.info("AR session started")
        return True
    
    def stop_ar_session(self):
        """Stop AR session"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        # Wait for threads to finish
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1.0)
        
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
        
        logger.info("AR session stopped")
    
    def _frame_capture_loop(self):
        """Main frame capture loop"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Process frame for AR rendering
            ar_frame = self.render_ar_frame(frame)
            
            # Call frame callback if set
            if self.frame_callback:
                self.frame_callback(ar_frame)
            
            # Control frame rate
            time.sleep(1.0 / self.fps)
    
    def _tracking_loop(self):
        """Tracking loop for pose estimation"""
        while self.is_running:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                else:
                    time.sleep(0.01)
                    continue
            
            # Perform tracking
            pose, confidence = self.estimate_pose(frame)
            
            if pose is not None:
                self.current_pose = pose
                self.tracking_confidence = confidence
                
                # Call tracking callback if set
                if self.tracking_callback:
                    self.tracking_callback(pose, confidence)
            
            time.sleep(0.01)  # 100 Hz tracking
    
    def estimate_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Estimate camera pose using visual tracking"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple feature-based tracking (in real implementation, use SLAM)
            # This is a placeholder - real AR would use ARCore, ARKit, or SLAM
            
            # Detect features
            corners = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=100, 
                qualityLevel=0.01, 
                minDistance=10
            )
            
            if corners is not None and len(corners) > 10:
                # Simulate pose estimation
                # In real implementation, use PnP solver with known 3D points
                confidence = min(1.0, len(corners) / 50.0)
                
                # Return identity pose for now (placeholder)
                pose = np.eye(4)
                return pose, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {str(e)}")
            return None, 0.0
    
    def render_ar_frame(self, frame: np.ndarray) -> np.ndarray:
        """Render AR overlays on frame"""
        ar_frame = frame.copy()
        
        try:
            # Render annotations
            self._render_annotations(ar_frame)
            
            # Render measurements
            self._render_measurements(ar_frame)
            
            # Render medical overlays
            self._render_medical_overlays(ar_frame)
            
            # Render UI elements
            self._render_ui_elements(ar_frame)
            
        except Exception as e:
            logger.error(f"AR rendering failed: {str(e)}")
        
        return ar_frame
    
    def _render_annotations(self, frame: np.ndarray):
        """Render text annotations"""
        for annotation in self.annotations.values():
            if not annotation.visible:
                continue
            
            # Project 3D position to 2D screen coordinates
            screen_pos = self._project_3d_to_2d(annotation.position)
            if screen_pos is None:
                continue
            
            x, y = int(screen_pos[0]), int(screen_pos[1])
            
            # Render text with background
            text_size = cv2.getTextSize(
                annotation.text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                annotation.size, 
                2
            )[0]
            
            # Background rectangle
            cv2.rectangle(
                frame,
                (x - 5, y - text_size[1] - 5),
                (x + text_size[0] + 5, y + 5),
                (0, 0, 0),
                -1
            )
            
            # Text
            cv2.putText(
                frame,
                annotation.text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                annotation.size,
                annotation.color,
                2
            )
    
    def _render_measurements(self, frame: np.ndarray):
        """Render measurement overlays"""
        for measurement in self.measurements.values():
            if not measurement.visible:
                continue
            
            # Project 3D points to 2D
            start_2d = self._project_3d_to_2d(measurement.start_point)
            end_2d = self._project_3d_to_2d(measurement.end_point)
            
            if start_2d is None or end_2d is None:
                continue
            
            start_2d = (int(start_2d[0]), int(start_2d[1]))
            end_2d = (int(end_2d[0]), int(end_2d[1]))
            
            # Draw measurement line
            cv2.line(frame, start_2d, end_2d, (0, 255, 0), 2)
            
            # Draw endpoints
            cv2.circle(frame, start_2d, 5, (0, 255, 0), -1)
            cv2.circle(frame, end_2d, 5, (0, 255, 0), -1)
            
            # Draw measurement value
            mid_point = (
                (start_2d[0] + end_2d[0]) // 2,
                (start_2d[1] + end_2d[1]) // 2
            )
            
            measurement_text = f"{measurement.value:.2f} {measurement.unit}"
            cv2.putText(
                frame,
                measurement_text,
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    
    def _render_medical_overlays(self, frame: np.ndarray):
        """Render medical data overlays"""
        for overlay_id, overlay_data in self.medical_overlays.items():
            if not overlay_data.get('visible', True):
                continue
            
            overlay_type = overlay_data.get('type', 'image')
            
            if overlay_type == 'image':
                self._render_image_overlay(frame, overlay_data)
            elif overlay_type == 'heatmap':
                self._render_heatmap_overlay(frame, overlay_data)
            elif overlay_type == 'segmentation':
                self._render_segmentation_overlay(frame, overlay_data)
    
    def _render_image_overlay(self, frame: np.ndarray, overlay_data: Dict):
        """Render image overlay (e.g., medical scan)"""
        try:
            overlay_image = overlay_data.get('image')
            position = overlay_data.get('position', (50, 50))
            size = overlay_data.get('size', (200, 200))
            alpha = overlay_data.get('alpha', 0.7)
            
            if overlay_image is not None:
                # Resize overlay image
                resized_overlay = cv2.resize(overlay_image, size)
                
                # Blend with frame
                x, y = position
                h, w = resized_overlay.shape[:2]
                
                if x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                    roi = frame[y:y+h, x:x+w]
                    blended = cv2.addWeighted(roi, 1-alpha, resized_overlay, alpha, 0)
                    frame[y:y+h, x:x+w] = blended
                    
        except Exception as e:
            logger.error(f"Failed to render image overlay: {str(e)}")
    
    def _render_heatmap_overlay(self, frame: np.ndarray, overlay_data: Dict):
        """Render heatmap overlay for attention or analysis results"""
        try:
            heatmap = overlay_data.get('heatmap')
            alpha = overlay_data.get('alpha', 0.5)
            
            if heatmap is not None:
                # Resize heatmap to frame size
                heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                
                # Apply colormap
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                # Blend with frame
                frame[:] = cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)
                
        except Exception as e:
            logger.error(f"Failed to render heatmap overlay: {str(e)}")
    
    def _render_segmentation_overlay(self, frame: np.ndarray, overlay_data: Dict):
        """Render segmentation mask overlay"""
        try:
            mask = overlay_data.get('mask')
            color = overlay_data.get('color', (0, 255, 0))
            alpha = overlay_data.get('alpha', 0.3)
            
            if mask is not None:
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0] = color
                
                # Blend with frame
                frame[:] = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
                
        except Exception as e:
            logger.error(f"Failed to render segmentation overlay: {str(e)}")
    
    def _render_ui_elements(self, frame: np.ndarray):
        """Render UI elements like status indicators"""
        # Render tracking status
        status_color = (0, 255, 0) if self.tracking_confidence > 0.5 else (0, 0, 255)
        status_text = f"Tracking: {self.tracking_confidence:.2f}"
        
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )
        
        # Render current mode
        mode_text = f"Mode: {self.current_mode.value}"
        cv2.putText(
            frame,
            mode_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    def _project_3d_to_2d(self, point_3d: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Project 3D point to 2D screen coordinates"""
        if not self.is_calibrated:
            # Simple projection for demo
            return (point_3d[0] * 100 + self.resolution[0] // 2,
                   point_3d[1] * 100 + self.resolution[1] // 2)
        
        try:
            # Use camera matrix for proper projection
            point_3d_array = np.array([[point_3d]], dtype=np.float32)
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            
            projected_points, _ = cv2.projectPoints(
                point_3d_array, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            
            return tuple(projected_points[0][0])
            
        except Exception as e:
            logger.error(f"3D to 2D projection failed: {str(e)}")
            return None
    
    # Public API methods
    def add_annotation(self, annotation: ARAnnotation):
        """Add AR annotation"""
        self.annotations[annotation.id] = annotation
        logger.info(f"Added annotation: {annotation.id}")
    
    def remove_annotation(self, annotation_id: str):
        """Remove AR annotation"""
        if annotation_id in self.annotations:
            del self.annotations[annotation_id]
            logger.info(f"Removed annotation: {annotation_id}")
    
    def add_measurement(self, measurement: ARMeasurement):
        """Add AR measurement"""
        self.measurements[measurement.id] = measurement
        logger.info(f"Added measurement: {measurement.id}")
    
    def remove_measurement(self, measurement_id: str):
        """Remove AR measurement"""
        if measurement_id in self.measurements:
            del self.measurements[measurement_id]
            logger.info(f"Removed measurement: {measurement_id}")
    
    def add_medical_overlay(self, overlay_id: str, overlay_data: Dict):
        """Add medical data overlay"""
        self.medical_overlays[overlay_id] = overlay_data
        logger.info(f"Added medical overlay: {overlay_id}")
    
    def remove_medical_overlay(self, overlay_id: str):
        """Remove medical data overlay"""
        if overlay_id in self.medical_overlays:
            del self.medical_overlays[overlay_id]
            logger.info(f"Removed medical overlay: {overlay_id}")
    
    def set_mode(self, mode: ARMode):
        """Set AR visualization mode"""
        self.current_mode = mode
        logger.info(f"AR mode set to: {mode.value}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_ar_state(self) -> Dict[str, Any]:
        """Get current AR state"""
        return {
            "is_running": self.is_running,
            "current_mode": self.current_mode.value,
            "tracking_confidence": self.tracking_confidence,
            "annotations_count": len(self.annotations),
            "measurements_count": len(self.measurements),
            "overlays_count": len(self.medical_overlays),
            "camera_resolution": self.resolution,
            "is_calibrated": self.is_calibrated
        }
    
    def export_ar_session(self) -> Dict[str, Any]:
        """Export AR session data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "annotations": {k: {
                "id": v.id,
                "position": v.position,
                "text": v.text,
                "color": v.color,
                "size": v.size,
                "timestamp": v.timestamp
            } for k, v in self.annotations.items()},
            "measurements": {k: {
                "id": v.id,
                "start_point": v.start_point,
                "end_point": v.end_point,
                "measurement_type": v.measurement_type,
                "value": v.value,
                "unit": v.unit,
                "accuracy": v.accuracy,
                "timestamp": v.timestamp
            } for k, v in self.measurements.items()},
            "session_info": self.get_ar_state()
        }

# Utility functions
def create_ar_engine(**kwargs) -> AREngine:
    """Factory function to create AR engine"""
    return AREngine(**kwargs)

if __name__ == "__main__":
    # Example usage
    ar_engine = AREngine()
    
    # Add sample annotation
    annotation = ARAnnotation(
        id="sample_annotation",
        position=(0.0, 0.0, 1.0),
        text="Sample Medical Annotation",
        color=(255, 255, 0)
    )
    ar_engine.add_annotation(annotation)
    
    # Add sample measurement
    measurement = ARMeasurement(
        id="sample_measurement",
        start_point=(0.0, 0.0, 0.0),
        end_point=(1.0, 0.0, 0.0),
        measurement_type="distance",
        value=10.5,
        unit="mm"
    )
    ar_engine.add_measurement(measurement)
    
    print("AR Engine initialized successfully")
    print(f"AR State: {ar_engine.get_ar_state()}")
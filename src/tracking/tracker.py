"""
tracker.py

High-level tracking orchestration layer.

Responsibilities:
- Tracker initialization
- Detection validation
- Tracker backend abstraction
- Runtime statistics
- Tracking interface for pipeline

This module does NOT implement tracking algorithms directly.
Actual tracker logic should reside inside backend wrappers
(e.g., ByteTrackWrapper, DeepSORTWrapper).

association logic
ID assignment
track update
"""

from typing import List, Dict, Any
import time

from src.utils.logger import get_logger
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.tracking.track_manager import TrackManager


class ObjectTracker:
    """
    High-level object tracking interface.

    This class acts as an orchestration layer between:
    - Detection subsystem
    - Tracking backend
    - Future tracking state management

    The pipeline should interact ONLY with this class.
    """

    def __init__(self, tracking_config: Dict[str, Any]):

        self.logger = get_logger(__name__)
        self.config = tracking_config
        self.enabled = self.config.get("enabled", True)

        self.tracker_type = self.config.get("tracker_type","bytetrack")
        self.confidence_threshold = self.config.get("confidence_threshold",0.5)
        self.track_buffer = self.config.get("track_buffer", 30)
        self.match_threshold = self.config.get("match_threshold",0.8)
        self.min_box_area = self.config.get("min_box_area",10)

        self.frame_rate = self.config.get("frame_rate",30)
        self.trajectory_history = self.config.get("trajectory_history",50)

        # Runtime metrics
        self.total_frames_processed = 0
        self.total_tracks_generated = 0
        self.total_tracking_time = 0.0

        # Placeholder tracker backend
        self.tracker_backend = None
        self.track_manager = TrackManager(trajectory_history=self.trajectory_history)

        self._initialize_tracker()

    def _initialize_tracker(self) -> None:
        """
        Initialize tracker backend.

        Currently initializes placeholder backend.
        Future:
        - ByteTrackWrapper
        - DeepSORTWrapper
        """

        self.logger.info(f"Initializing tracker backend: {self.tracker_type}")
        
        if self.tracker_type.lower() == "bytetrack":
            # Placeholder for future implementation
            self.tracker_backend = ByteTrackWrapper(track_buffer=self.track_buffer,
                                                    match_threshold=self.match_threshold,
                                                    frame_rate=self.frame_rate)
            self.logger.info("ByteTrack backend placeholder initialized.")

        else:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}")

    def _validate_detections(self,detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter detections before tracking.

        Expected detection format:
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.91,
            "class_name": "person"
        }
        """

        valid_detections = []

        for detection in detections:

            if "bbox" not in detection:
                continue

            if "confidence" not in detection:
                continue

            bbox = detection["bbox"]

            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            if area < self.min_box_area:
                continue
            confidence = detection["confidence"]
            if confidence < self.confidence_threshold:
                continue

            valid_detections.append(detection)

        return valid_detections

    def update_tracks(self,detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Update tracker using current frame detections.

        Parameters
        ----------
        detections : List[Dict]
            Detector outputs from postprocessor.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Persistent tracked object outputs.

        Current output format:
        {
            1: {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.91,
                "class_name": "person"
            }
        ]
        """

        if not self.enabled:
            return []

        start_time = time.time()
        self.total_frames_processed += 1
        validated_detections = self._validate_detections(detections)

        # Placeholder tracking output
        tracked_objects = self.tracker_backend.update(validated_detections)
        persistent_tracks = self.track_manager.update_tracks(tracked_objects)

        tracking_time = time.time() - start_time
        self.total_tracking_time += tracking_time
        self.total_tracks_generated += len(tracked_objects)

        return persistent_tracks

    def reset(self) -> None:
        """
        Reset tracker state.
        """

        self.logger.info("Resetting tracker state.")

        self.total_frames_processed = 0
        self.total_tracks_generated = 0
        self.total_tracking_time = 0.0
        if self.tracker_backend is not None:
            self.track_manager.reset()

    def get_stats(self) -> Dict[str, Any]:
        """
        Return tracker runtime statistics.
        """

        average_tracking_time = 0.0

        if self.total_frames_processed > 0:
            average_tracking_time = (self.total_tracking_time / self.total_frames_processed )

        return {
            "tracker_type": self.tracker_type,
            "frames_processed": self.total_frames_processed,
            "tracks_generated": self.total_tracks_generated,
            "total_tracking_time": round(self.total_tracking_time,4),
            "average_tracking_time": round(average_tracking_time,4)
        }
"""
State manager for tracks.
Purpose: Persistent history of tracks, their states, and associated metadata.
"""
# -----------------------------------------------------------------------------

"""
track_manager.py

Persistent track state management for RTVIS.

Responsibilities:
- Track persistence
- Trajectory history
- Track lifecycle metadata
- Cleanup of stale tracks
- State access for event engine

This module is tracker-agnostic and does NOT implement
tracking algorithms directly.
"""

from typing import Dict, List, Any, Optional
from collections import deque
import time

from src.utils.logger import get_logger


class TrackManager:
    """
    Persistent track state manager.
    """

    def __init__(self,trajectory_history: int = 50,max_inactive_time: int = 2):

        self.logger = get_logger(__name__)
        self.trajectory_history = trajectory_history
        self.max_inactive_time = max_inactive_time

        # Main persistent track store
        self.tracks = {}

        self.logger.info("TrackManager initialized.")

    def _compute_center(self,bbox: List[float]) -> List[int]:
        """
        Compute center point from bbox.
        """

        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return [center_x, center_y]

    def _initialize_track(self,tracked_object: Dict[str, Any]) -> None:
        """
        Initialize new persistent track.
        """

        track_id = tracked_object["track_id"]

        center = self._compute_center(tracked_object["bbox"])

        current_time = time.time()

        self.tracks[track_id] = {
            "track_id": track_id,
            "class_name": tracked_object["class_name"],
            "confidence": tracked_object["confidence"],
            "bbox": tracked_object["bbox"],
            "history": deque([center],maxlen=self.trajectory_history),
            "first_seen": current_time,
            "last_seen": current_time,
            "total_visible_frames": 1
        }

    def _update_track(self,tracked_object: Dict[str, Any]) -> None:
        """
        Update existing persistent track.
        """

        track_id = tracked_object["track_id"]

        center = self._compute_center(tracked_object["bbox"])
        current_time = time.time()

        track = self.tracks[track_id]
        track["bbox"] = tracked_object["bbox"]
        track["confidence"] = tracked_object["confidence"]
        track["last_seen"] = current_time
        track["total_visible_frames"] += 1
        track["history"].append(center)

    def update_tracks(self,tracked_objects: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Update persistent track states.
        """

        for tracked_object in tracked_objects:
            track_id = tracked_object["track_id"]
            if track_id not in self.tracks:
                self._initialize_track(tracked_object)

            else:
                self._update_track(tracked_object)

        self._cleanup_tracks()

        return self.tracks

    def _cleanup_tracks(self) -> None:
        """
        Remove stale inactive tracks.
        """

        current_time = time.time()
        tracks_to_remove = []

        for track_id, track_data in self.tracks.items():
            inactive_time = (current_time - track_data["last_seen"])

            if inactive_time > self.max_inactive_time:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            self.logger.info(f"Removed inactive track ID: {track_id}")

    def get_track(self,track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get single track by ID.
        """
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> Dict[int, Dict[str, Any]]:
        """
        Return all active persistent tracks.
        """
        return self.tracks

    def reset(self) -> None:
        """
        Reset track manager state.
        """

        self.logger.info("Resetting TrackManager state.")
        self.tracks.clear()
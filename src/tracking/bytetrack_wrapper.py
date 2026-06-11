"""
bytetrack_wrapper.py

ByteTrack backend wrapper for RTVIS.

Responsibilities
----------------
- Detection format conversion
- Track update orchestration
- Track output formatting
- Backend abstraction

This module hides tracking backend complexity from
the main pipeline and orchestration layer.
"""

from typing import List, Dict, Any

import numpy as np
import supervision as sv

from src.utils.logger import setup_logger


class ByteTrackWrapper:
    """
    ByteTrack backend wrapper.

    Responsibilities
    ----------------
    - Convert detector outputs into Supervision format
    - Update ByteTrack state
    - Convert tracked objects into RTVIS format
    """

    def __init__(self, track_buffer: int = 30, 
                 match_threshold: float = 0.8, frame_rate: int = 30):

        self.logger = setup_logger(__name__)

        self.track_buffer = track_buffer
        self.match_threshold = match_threshold
        self.frame_rate = frame_rate

        self.byte_tracker = sv.ByteTrack(track_activation_threshold=0.5,lost_track_buffer=self.track_buffer,
                                         minimum_matching_threshold=self.match_threshold,frame_rate=self.frame_rate)

        self.logger.info("ByteTrack wrapper initialized.")

    def _convert_detections(self, detections: List[Dict[str, Any]]) -> sv.Detections:
        """
        Convert RTVIS detections into Supervision format.

        Expected RTVIS format
        ---------------------
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.91,
            "class_name": "person"
        }
        """

        if len(detections) == 0:
            return sv.Detections.empty()

        xyxy = []
        confidence = []
        class_names = []

        for detection in detections:

            xyxy.append(detection["bbox"])
            confidence.append(detection["confidence"])
            class_names.append(detection["class_name"])

        return sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            data={
                "class_name": np.array(class_names)
            }
        )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update ByteTrack and return RTVIS tracked objects.
        """

        sv_detections = self._convert_detections(
            detections
        )

        tracked_detections = (
            self.byte_tracker.update_with_detections(
                sv_detections
            )
        )

        tracked_objects = []

        for idx in range(len(tracked_detections)):

            bbox = (
                tracked_detections.xyxy[idx]
                .astype(int)
                .tolist()
            )

            confidence = float(
                tracked_detections.confidence[idx]
            )

            track_id = int(
                tracked_detections.tracker_id[idx]
            )

            class_name = str(
                tracked_detections.data[
                    "class_name"
                ][idx]
            )

            tracked_object = {

                "track_id": track_id,

                "bbox": bbox,

                "confidence": confidence,

                "class_name": class_name
            }

            tracked_objects.append(
                tracked_object
            )

        return tracked_objects

    def reset(self) -> None:
        """
        Reset ByteTrack state.
        """

        self.logger.info(
            "Resetting ByteTrack wrapper."
        )

        self.byte_tracker.reset()
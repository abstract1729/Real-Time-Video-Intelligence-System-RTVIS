"""
intrusion.py

Detects when an object enters a restricted polygon zone.

Author: RTVIS
"""

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.events.base_event import BaseEvent


class IntrusionEvent(BaseEvent):
    """
    Detects intrusion into a polygon zone.
    """

    def __init__(self, zone_points: List[Tuple[int, int]]):
        """
        Parameters
        ----------
        zone_points : List[Tuple[int, int]]
            Vertices of polygon zone.
        """

        self.zone = np.array(zone_points, dtype=np.int32)
        # Prevent repeated alerts
        self.active_tracks = set()

    def process(self, track: Dict) -> List[Dict]:
        """
        Process a single track.

        Parameters
        ----------
        track : Dict
            Track dictionary from TrackManager.

        Returns
        -------
        List[Dict]
            Generated intrusion events.
        """

        events = []
        track_id = track["track_id"]

        # Event already generated
        if track_id in self.active_tracks:
            return events

        history = track["history"]

        if len(history) == 0:
            return events

        center = history[-1]

        if self._is_inside_zone(center):

            event = {
                "event_type": "intrusion",
                "track_id": track_id,
                "timestamp": time.time()
            }

            events.append(event)
            self.active_tracks.add(track_id)

        return events

    def _is_inside_zone(self, point: Tuple[int, int]) -> bool:
        """
        Check whether a point lies inside the polygon.
        """
        result = cv2.pointPolygonTest(self.zone, point, False)
        return result >= 0
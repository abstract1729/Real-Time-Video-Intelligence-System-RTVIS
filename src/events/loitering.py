"""
loitering.py

Detects loitering behavior based on object dwell time.

Author: RTVIS
"""

import time
from typing import Dict, List

from src.events.base_event import BaseEvent


class LoiteringEvent(BaseEvent):
    """
    Detects objects that remain visible longer than
    a specified duration threshold.
    """

    def __init__(
        self,
        duration_threshold: float = 30.0
    ):
        """
        Parameters
        ----------
        duration_threshold : float
            Minimum dwell time (seconds) required to
            trigger a loitering event.
        """

        self.duration_threshold = duration_threshold

        # Prevent repeated event generation
        self.reported_tracks = set()

    def process(self, track: Dict) -> List[Dict]:
        """
        Process a single track.

        Parameters
        ----------
        track : Dict
            Track dictionary obtained from TrackManager.

        Returns
        -------
        List[Dict]
            Generated loitering events.
        """

        events = []

        track_id = track["track_id"]

        # Event already reported
        if track_id in self.reported_tracks:
            return events

        first_seen = track.get("first_seen")
        last_seen = track.get("last_seen")

        # Safety check
        if first_seen is None or last_seen is None:
            return events

        duration = last_seen - first_seen

        if duration >= self.duration_threshold:

            event = {
                "event_type": "loitering",
                "track_id": track_id,
                "timestamp": time.time(),
                "duration": duration
            }

            events.append(event)

            self.reported_tracks.add(track_id)

        return events
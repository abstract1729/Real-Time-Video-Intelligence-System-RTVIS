"""
line_crossing.py

Detects when an object trajectory crosses a virtual line.

Author: RTVIS
"""

import time
from typing import Dict, List, Tuple
from src.events.base_event import BaseEvent


class LineCrossingEvent(BaseEvent):
    """
    Event detector for line crossing.
    """

    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """
        Parameters
        ----------
        line_start : Tuple[int, int]
            Start point of the virtual line.

        line_end : Tuple[int, int]
            End point of the virtual line.
        """

        self.line_start = line_start
        self.line_end = line_end

        # Prevent repeated event generation
        self.crossed_tracks = set()

    def process(self, track: Dict) -> List[Dict]:
        """
        Process a single track.

        Parameters
        ----------
        track : Dict
            Track information from TrackManager.

        Returns
        -------
        List[Dict]
            List of generated events.
        """

        events = []
        track_id = track["track_id"]

        # Event already generated for this track
        if track_id in self.crossed_tracks:
            return events

        history = track["history"]

        # Need at least two points to form a segment
        if len(history) < 2:
            return events

        p1 = history[-2]
        p2 = history[-1]

        if self._intersects(p1,p2,self.line_start,self.line_end):
            event = {
                "event_type": "line_crossing",
                "track_id": track_id,
                "timestamp": time.time()
            }

            events.append(event)
            self.crossed_tracks.add(track_id)

        return events

    @staticmethod
    def _orientation(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> int:
        """
        Returns orientation of ordered triplet (p, q, r).

        0 -> Collinear
        1 -> Clockwise
        2 -> Counterclockwise
        """

        value = (
            (q[1] - p[1]) * (r[0] - q[0])
            - (q[0] - p[0]) * (r[1] - q[1])
        )

        if value == 0:
            return 0

        return 1 if value > 0 else 2

    def _intersects( self, p1: Tuple[int, int], p2: Tuple[int, int],
                    q1: Tuple[int, int], q2: Tuple[int, int]) -> bool:
        """
        Check whether segments p1-p2 and q1-q2 intersect.
        """

        o1 = self._orientation(p1, p2, q1)
        o2 = self._orientation(p1, p2, q2)

        o3 = self._orientation(q1, q2, p1)
        o4 = self._orientation(q1, q2, p2)

        return (o1 != o2) and (o3 != o4)
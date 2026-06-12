"""
event_engine.py
High-level orchestration layer for event detection.
Author: RTVIS
"""

from typing import Dict, List

from src.events.line_crossing import LineCrossingEvent
from src.events.intrusion import IntrusionEvent
from src.events.loitering import LoiteringEvent


class EventEngine:
    """
    High-level event orchestration layer.
    """

    def __init__(self, config: Dict):
        self.event_modules = []

        # -----------------------------------
        # Line Crossing
        # -----------------------------------

        line_crossing_config = config.get("line_crossing", {})

        if line_crossing_config.get("enabled", False):
            self.event_modules.append(
                LineCrossingEvent(
                    line_start=tuple(line_crossing_config["line_start"]),
                    line_end=tuple(line_crossing_config["line_end"])))

        # -----------------------------------
        # Intrusion
        # -----------------------------------

        intrusion_config = config.get("intrusion", {})

        if intrusion_config.get("enabled", False):
            self.event_modules.append(IntrusionEvent(zone_points=[tuple(point) for point in intrusion_config["zone_points"]]))

        # -----------------------------------
        # Loitering
        # -----------------------------------

        loitering_config = config.get("loitering", {})

        if loitering_config.get("enabled", False):
            self.event_modules.append(LoiteringEvent(duration_threshold=loitering_config.get("duration_threshold",30.0)))

    def process_tracks(self,tracks: Dict[int, Dict]) -> List[Dict]:
        events = []
        for track in tracks.values():
            for event_module in self.event_modules:
                detected_events = event_module.process(track)
                events.extend(detected_events)

        return events
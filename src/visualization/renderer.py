"""
Coordinate rendering pipeline.
Renderer is the orchestration layer. It calls Overlay internally to draw detections on frames.
These detections could be: tracks, events, metrics, etc. In the future, it will also handle track IDs and event polygons.

Responsibilities:
- Receive raw detections
- Draw bounding boxes
"""

from typing import Optional
import numpy as np
from src.visualization.overlay import OverlayDrawer
from collections import deque
from datetime import datetime

class FrameRenderer:
    """
    High-level rendering orchestration layer.

    Responsibilities:
    -----------------
    - Coordinate visualization
    - Render detections
    - Render future tracks
    - Render future events
    - Add system metrics

    Notes:
    -----
    OverlayDrawer performs actual drawing.

    FrameRenderer decides:
        WHAT gets drawn
        WHEN it gets drawn
    """

    def __init__(self,overlay_drawer: Optional[OverlayDrawer] = None):
        
        # Graphics Primitive Layer
        self.overlay = (overlay_drawer if overlay_drawer is not None
            else OverlayDrawer())

        
        # Runtime State
        self.last_rendered_frame = None
        self.total_render_calls = 0

        self.event_history = deque(maxlen=10)
        self.event_counts = {"line_crossing": 0,"intrusion": 0,"loitering": 0}
        self.seen_events = set()

    def render_detections(self,frame: np.ndarray,detections: list) -> np.ndarray:
        """
        Render object detections.
        Parameters
        ----------
        frame : np.ndarray
        detections : list
        Detection format:
        {
            bbox:
            confidence:
            class_name:
        }
        """

        rendered_frame = frame.copy()
        for detection in detections:
            rendered_frame = (
                self.overlay.draw_detection(
                    rendered_frame,
                    detection
                )
            )

        return rendered_frame

    def render_tracks(self, frame: np.ndarray, tracks: dict) -> np.ndarray:
        """
        Render tracked objects.
        Parameters
        ----------
        frame : np.ndarray
        tracks : dict
        Format
        ------
        {
            track_id:
            {
                track_data
            }
        }
        """

        rendered_frame = frame.copy()

        for _, track_data in tracks.items():
            rendered_frame = (
                self.overlay.draw_track(
                    rendered_frame,
                    track_data
                )
            )

        return rendered_frame

    def update_event_statistics(self,events: list)-> None:
        """
        Update cumulative event counts and history.
        """

        for event in events:
            event_key = (event["event_type"],event["track_id"])

            if event_key in self.seen_events:
                continue

            self.seen_events.add(event_key)

            event_type = event["event_type"]

            if event_type in self.event_counts:
                self.event_counts[event_type] += 1

            history_entry = (
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"{event_type.upper()} "
            f"ID:{event['track_id']}")

            self.event_history.append(history_entry)

    def render_virtual_line(self,frame: np.ndarray,line_start: tuple,line_end: tuple) -> np.ndarray:
        """
        Render line-crossing boundary.
        """

        rendered_frame = frame.copy()
        rendered_frame = (self.overlay.draw_virtual_line(rendered_frame,line_start,line_end))

        return rendered_frame

    def render_intrusion_zone(self,frame: np.ndarray,zone_points: list) -> np.ndarray:
        """
        Render intrusion polygon.
        """

        rendered_frame = frame.copy()
        rendered_frame = (self.overlay.draw_intrusion_zone(rendered_frame,zone_points))

        return rendered_frame

    def render_event_statistics(self,frame: np.ndarray) -> np.ndarray:
        rendered_frame = frame.copy()
        start_y = 180

        rendered_frame = self.overlay.draw_text(
            rendered_frame,
            f"Line Crossings: "
            f"{self.event_counts['line_crossing']}",
            position=(20, start_y)
        )

        start_y += 30

        rendered_frame = self.overlay.draw_text(
            rendered_frame,
            f"Intrusions: "
            f"{self.event_counts['intrusion']}",
            position=(20, start_y)
        )

        start_y += 30

        rendered_frame = self.overlay.draw_text(
            rendered_frame,
            f"Loitering: "
            f"{self.event_counts['loitering']}",
            position=(20, start_y)
        )

        return rendered_frame

    def render_event_history(self, frame: np.ndarray) -> np.ndarray:
        rendered_frame = frame.copy()
        start_y = 300

        rendered_frame = self.overlay.draw_text(rendered_frame,"Recent Events",position=(20, start_y))
        start_y += 30

        for event in reversed(self.event_history):
            rendered_frame = self.overlay.draw_text(rendered_frame,event,position=(20, start_y))
            start_y += 25

        return rendered_frame

    def render_events(self,frame: np.ndarray,events: list) -> np.ndarray:
        """
        Render event notifications.
        Parameters
        ----------
        frame : np.ndarray
        events : list
        Format
        ------
        [
            {
                "event_type": "...",
                "track_id": ...
            }
        ]
        """

        rendered_frame = frame.copy()
        start_y = 120

        for event in events:
            text = (
                f"{event['event_type'].upper()} "
                f"(ID:{event['track_id']})"
            )

            rendered_frame = (
                self.overlay.draw_event_label(
                    rendered_frame,
                    text,
                    position=(20, start_y)
                )
            )

            start_y += 30

        return rendered_frame

    def render_metrics( self, frame: np.ndarray,fps: Optional[float]=None, inference_time: Optional[float]=None,
                       detection_count: Optional[int]=None, track_count: Optional[int]=None)  -> np.ndarray:
        """
        Render system metrics.
        Future:
        -------
        FPS
        inference latency
        event count
        queue length
        tracking count
        """

        current_y = 30

        if fps is not None:
            frame = self.overlay.draw_text(frame,f"FPS: {fps:.2f}",position=(20, current_y))
            current_y += 30

        if inference_time is not None:
            frame = self.overlay.draw_text(frame, f"Inference: {inference_time*1000:.2f} ms",position=(20, current_y))
            current_y += 30

        if detection_count is not None:
            frame = self.overlay.draw_text(frame, f"Detections: {detection_count}", position=(20, current_y))
        
        if track_count is not None:
            current_y += 30
            frame = self.overlay.draw_text(frame, f"Tracks: {track_count}", position=(20, current_y))


        return frame

    def render_frame(self,frame: np.ndarray,detections=None,tracks=None,events=None,line_start=None,
                     line_end=None,intrusion_zone=None,fps=None,inference_time=None) -> np.ndarray:

        """
        Full rendering pipeline.
        Pipeline:
        Frame
          ↓
        Detections
          ↓
        Metrics
          ↓
        Output Frame
        """

        rendered_frame = frame.copy()
        detection_count = 0
        track_count = 0

        # -----------------------------
        # Detection Layer
        # -----------------------------

        if detections:
            rendered_frame = (self.render_detections(rendered_frame, detections))
            detection_count = len(detections)

        # -----------------------------
        # Track Layer
        # -----------------------------

        if tracks:
            rendered_frame = (self.render_tracks( rendered_frame,tracks))
            track_count = len(tracks)
        
        # -----------------------------
        # Region Layer
        # -----------------------------

        if line_start is not None and line_end is not None:
            rendered_frame = (self.render_virtual_line(rendered_frame, line_start,line_end))


        if intrusion_zone is not None:
            rendered_frame = (self.render_intrusion_zone(rendered_frame,intrusion_zone))

        
        # -----------------------------
        # Event Layer
        # -----------------------------

        if events:
            rendered_frame = (self.render_events(rendered_frame,events))
            self.update_event_statistics(events)
            

        # -----------------------------
        # Metrics Layer
        # -----------------------------
        rendered_frame = (self.render_metrics(rendered_frame, fps=fps, inference_time=inference_time, 
                                              detection_count=detection_count,track_count=track_count))
        rendered_frame = (self.render_event_statistics(rendered_frame))

        rendered_frame = (self.render_event_history(rendered_frame))
        
        self.total_render_calls += 1
        self.last_rendered_frame = (rendered_frame)

        return rendered_frame

    def get_render_statistics(self) -> dict:
        """
        Return renderer statistics.
        """
        return {"total_render_calls":self.total_render_calls}



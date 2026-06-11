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

    def render_frame(self, frame: np.ndarray, detections: Optional[list] = None, tracks: Optional[dict]=None,
                     fps: Optional[float] = None, inference_time: Optional[float] = None) -> np.ndarray:
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
        # Metrics Layer
        # -----------------------------

        rendered_frame = (self.render_metrics(rendered_frame, fps=fps, inference_time=inference_time, 
                                              detection_count=detection_count,track_count=track_count))
        self.total_render_calls += 1
        self.last_rendered_frame = (rendered_frame)

        return rendered_frame

    

    def get_render_statistics(self) -> dict:
        """
        Return renderer statistics.
        """
        return {"total_render_calls":self.total_render_calls}



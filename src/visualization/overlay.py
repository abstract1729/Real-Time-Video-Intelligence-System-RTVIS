"""
Overlay detections on video frames.

Responsibilities:
- Draw bounding boxes
- Annotate class labels
- Display confidence scores
- Draw track IDs (later)
- Draw event polygons (future)

This class is responsible for rendering the final output frame with detection information.
It should remain stateless and only modify frames.
"""

import cv2
import numpy as np


class OverlayDrawer:
    """
    Graphics primitive layer for RTVIS.

    Responsibilities:
    -----------------
    - Draw bounding boxes
    - Draw labels
    - Draw text
    - Draw polygons
    - Draw lines

    Notes:
    ------
    This class ONLY handles drawing.
    It does NOT know:
        - YOLO
        - Tracking
        - Events
        - Pipeline logic
    """

    def __init__(self,bbox_color=(0, 255, 0),text_color=(255, 255, 255),line_thickness=2,font_scale=0.5):

        self.bbox_color = bbox_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_track_color(self,track_id: int) -> tuple:
        """
        Generate deterministic color from track ID.
        """
        np.random.seed(track_id)
        color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
        return color

    def draw_bbox(self,frame: np.ndarray,bbox: list, color: tuple =(0, 0, 255)) -> np.ndarray:
        """
        Draw bounding box.

        Parameters
        ----------
        frame : np.ndarray
        bbox : [x1, y1, x2, y2]
        """

        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame,(x1, y1),(x2, y2),color,self.line_thickness)
        return frame

    def draw_label(self,frame: np.ndarray,bbox: list,label: str) -> np.ndarray:
        """
        Draw label above bounding box.
        """
        x1, y1, _, _ = bbox
        text_y = max(20, y1 - 10)
        cv2.putText(frame,label,(x1, text_y),self.font,self.font_scale,self.text_color,self.line_thickness)
        return frame

    def draw_detection(self,frame: np.ndarray,detection: dict) -> np.ndarray:
        """
        Draw complete detection.
        Detection format:
        {
            bbox:
            confidence:
            class_name:
        }
        """

        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = (f"{class_name}: "f"{confidence:.2f}")
        frame = self.draw_bbox(frame,bbox)
        frame = self.draw_label(frame,bbox,label)
        return frame

    def draw_text(self,frame: np.ndarray,text: str,position=(20, 30),color=(0, 0, 255) ) -> np.ndarray:
        """
        Draw generic text.
        """
        cv2.putText(frame,text,position,self.font,self.font_scale,color,self.line_thickness)
        return frame

    def draw_polygon(self, frame: np.ndarray,points: list,color=(255, 0, 0)) -> np.ndarray:
        """
        Draw polygon.

        Used later for:
        - intrusion zones
        - ROI regions
        """
        polygon = np.array(points,dtype=np.int32)
        cv2.polylines(frame,[polygon],isClosed=True,color=color,thickness=self.line_thickness)
        return frame

    def draw_line(self, frame: np.ndarray, start_point: tuple, end_point: tuple,color=(0, 0, 255)) -> np.ndarray:
        """
        Draw line.

        Used later for:
        - line crossing events
        """
        cv2.line(frame,start_point,end_point,color,self.line_thickness)
        return frame

    def draw_trajectory(self,frame: np.ndarray, history, color=(255, 0, 255)) -> np.ndarray:
        """
        Draw object trajectory.

        Parameters
        ----------
        history : iterable
            Sequence of center points:
            [(x1,y1), (x2,y2), ...]

        Used later for:
        - line crossing
        - loitering
        - path visualization
        """

        history = list(history)

        if len(history) < 2:
            return frame

        for i in range(1, len(history)):

            start_point = tuple(history[i - 1])
            end_point = tuple(history[i])

            cv2.line(frame, start_point, end_point, 
                     color,self.line_thickness)

        return frame

    def draw_track(self, frame: np.ndarray, track_data: dict) -> np.ndarray:
        """
        Draw tracked object.

        Expected format
        ----------------
        {
            "track_id": 1,
            "bbox": [...],
            "confidence": 0.91,
            "class_name": "person",
            "history": [...]
        }
        """

        bbox = track_data["bbox"]
        class_name = track_data["class_name"]
        confidence = track_data["confidence"]
        track_id = track_data["track_id"]
        history = track_data["history"]
        color = self._get_track_color(track_id)

        label = (
            f"ID:{track_id} "
            f"{class_name} "
            f"{confidence:.2f}"
        )

        frame = self.draw_bbox(frame, bbox, color=color)
        frame = self.draw_label(frame, bbox, label)
        frame = self.draw_trajectory(frame, history, color=color)

        return frame


if __name__ == "__main__":
    drawer = OverlayDrawer()
    frame = np.zeros((640,640,3),dtype=np.uint8)

    detection = {
        "bbox":[100,200,300,300],
        "confidence":0.91,
        "class_name":"person"}

    frame = drawer.draw_detection(frame,detection)
    cv2.imshow("Overlay Test",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
import time
from typing import Optional, Tuple

import numpy as np


class VideoCaptureManager:
    """
    Manages video ingestion from different source types.

    Supported Sources:
    ------------------
    - Webcam
    - Video File
    - RTSP Stream

    Responsibilities:
    -----------------
    - Open and validate video source
    - Read frames safely
    - Maintain stream metadata
    - Track runtime metrics
    - Handle resource cleanup

    Notes:
    ------
    This class is intentionally synchronous for the initial pipeline version.
    Threading / async ingestion can be introduced later.
    """

    VALID_SOURCE_TYPES = {"webcam", "file", "rtsp"}

    def __init__(self, src: str, src_type: str) -> None:
        """
        Initialize video source configuration.

        Parameters
        ----------
        src : str
            Video source path / stream URL / webcam identifier.

        src_type : str
            Type of source:
            - webcam
            - file
            - rtsp
        """

        src_type = src_type.lower()

        if src_type not in self.VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source type: {src_type}. "
                f"Supported types: {self.VALID_SOURCE_TYPES}"
            )

        self.src_type = src_type

        # Webcam sources are typically represented by integer device IDs.
        if self.src_type == "webcam":
            self.src = 0
        else:
            self.src = src

        # -----------------------------
        # OpenCV VideoCapture Handle
        # -----------------------------
        self.capture: Optional[cv2.VideoCapture] = None

        # -----------------------------
        # Stream Metadata
        # -----------------------------
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None

        # -----------------------------
        # Runtime State
        # -----------------------------
        self.is_opened: bool = False
        self.frame_count: int = 0
        self.current_frame: Optional[np.ndarray] = None

        # -----------------------------
        # Monitoring / Diagnostics
        # -----------------------------
        self.failed_reads: int = 0
        self.last_successful_frame_time: Optional[float] = None

    def open(self, fallback_fps: int = 30) -> None:
        """
        Open the video source and initialize stream metadata.

        Parameters
        ----------
        fallback_fps : int
            FPS value used if source FPS cannot be determined.
        """

        self.capture = cv2.VideoCapture(self.src)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open video source: {self.src}"
            )

        # Retrieve stream metadata
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detected_fps = self.capture.get(cv2.CAP_PROP_FPS)

        # Some webcams return 0 FPS.
        if detected_fps <= 0:
            self.fps = fallback_fps
        else:
            self.fps = detected_fps

        self.is_opened = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.

        Returns
        -------
        Tuple[bool, Optional[np.ndarray]]
            (success flag, frame)

        Notes
        -----
        - Does not raise exception for normal EOF/frame failure.
        - Caller decides how to handle failures.
        """

        if not self.is_opened or self.capture is None:
            raise RuntimeError(
                "Video source is not opened. Call open() first."
            )

        success, frame = self.capture.read()

        if not success:
            self.failed_reads += 1
            return False, None

        # Update runtime state
        self.current_frame = frame
        self.frame_count += 1
        self.last_successful_frame_time = time.time()

        return True, frame

    def get_stream_properties(self) -> dict:
        """
        Return stream metadata.
        """

        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def is_active(self) -> bool:
        """
        Check whether stream is active.
        """

        return self.is_opened

    def release(self) -> None:
        """
        Safely release video resources.
        """

        if self.capture is not None:
            self.capture.release()

        self.is_opened = False

    def reset_metrics(self) -> None:
        """
        Reset runtime monitoring metrics.
        """

        self.frame_count = 0
        self.failed_reads = 0
        self.last_successful_frame_time = None

    # -------------------------------------------------
    # Context Manager Support
    # Enables:
    #
    # with VideoCaptureManager(...) as cap:
    #     ...
    # -------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()







if __name__ == "__main__":

    video_manager = VideoCaptureManager( src="webcam", src_type="webcam")

    try:
        video_manager.open()
        print(video_manager.get_stream_properties())
        while video_manager.is_active():
            success, frame = video_manager.read_frame()

            if not success:
                print("Frame read failed.")
                break

            cv2.imshow("RTVIS Stream", frame)

            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        video_manager.release()
        cv2.destroyAllWindows()


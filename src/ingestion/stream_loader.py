"""

Contains
StreamLoader

Responsibilities:

read source config
choose correct capture backend
instantiate correct source manager

system.yaml
    ↓
StreamLoader
    ↓
VideoCaptureManager

"""
from src.ingestion.video_capture import VideoCaptureManager

class StreamLoader:
    """
    Responsible for loading the correct
    video stream source implementation.
    """

    def __init__(self, ingestion_config: dict) -> None:
        self.ingestion_config = ingestion_config

    def load_stream(self) -> VideoCaptureManager:

        source_type = self.ingestion_config.get("source_type")
        source = self.ingestion_config.get("source")

        capture_manager = VideoCaptureManager(
            src=source,
            src_type=source_type
        )

        return capture_manager
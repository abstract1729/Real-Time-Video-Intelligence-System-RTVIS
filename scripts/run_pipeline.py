import sys
from pathlib import Path

import cv2


# Add Project Root to Python Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# Internal Imports
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger

from src.ingestion.stream_loader import StreamLoader

# Logger
logger = setup_logger(__name__)


def main():

    try:

        # Load System Configuration
        config = ConfigLoader.load_yaml(ROOT_DIR / "configs/system.yaml")
        ingestion_config = config["system"]["ingestion"]
        logger.info("System configuration loaded.")

        
        # Initialize Stream Loader
        stream_loader = StreamLoader(ingestion_config)
        capture_manager = stream_loader.load_stream()
        logger.info("Video source initialized.")

        # Open Stream
        fallback_fps = ingestion_config.get("fallback_fps",30)

        capture_manager.open(fallback_fps=fallback_fps)

        logger.info(
            f"Stream opened successfully | "
            f"Resolution: "
            f"{capture_manager.width}x{capture_manager.height} | "
            f"FPS: {capture_manager.fps}"
        )

        
        # Read / Display Frames
        while capture_manager.is_active():
            success, frame = capture_manager.read_frame()
            if not success:
                logger.warning("Frame read failed.")
                break

            cv2.imshow("RTVIS Stream", frame)

            # ESC Key
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("Exit requested by user.")
                break

    except Exception as error:

        logger.error(f"Pipeline execution failed: {error}")

    finally:

        # Cleanup
        if "capture_manager" in locals():
            capture_manager.release()

        cv2.destroyAllWindows()

        logger.info("Resources released successfully.")


if __name__ == "__main__":
    main()
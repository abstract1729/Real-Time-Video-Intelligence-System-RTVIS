import sys
from pathlib import Path
import cv2

# # --------------------------------------------------
# # Add Project Root to Python Path
# # --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# Internal Imports
# --------------------------------------------------

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger

from src.ingestion.stream_loader import StreamLoader

from src.detection.model_loader import ModelLoader
from src.detection.yolo_inference import YOLOInference
from src.detection.postprocess import DetectionPostProcessor
from src.tracking.tracker import ObjectTracker
from src.events.event_engine import EventEngine

from src.visualization.renderer import FrameRenderer

# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = setup_logger(__name__)


def main():

    capture_manager = None

    try:

        # ------------------------------------------
        # Load Configurations
        # ------------------------------------------

        system_config = ConfigLoader.load_yaml(ROOT_DIR / "configs/system.yaml")
        detection_config = ConfigLoader.load_yaml(ROOT_DIR / "configs/detection.yaml")
        tracking_config = ConfigLoader.load_yaml(ROOT_DIR / "configs/tracking.yaml")
        events_config = ConfigLoader.load_yaml(ROOT_DIR / "configs/events.yaml")
        ingestion_config = (system_config["system"]["ingestion"])
        detection_settings = (detection_config["detection"])
        tracking_settings = (tracking_config["tracking"])
        line_crossing_config = events_config["line_crossing"]
        line_start = tuple(line_crossing_config["line_start"])
        line_end = tuple(line_crossing_config["line_end"])
        intrusion_zone = [tuple(point) for point in events_config["intrusion"]["zone_points"]]

        logger.info("Configurations loaded successfully.")

        # ------------------------------------------
        # Stream Initialization
        # ------------------------------------------

        stream_loader = StreamLoader(ingestion_config)
        capture_manager = (stream_loader.load_stream())
        capture_manager.open(fallback_fps=ingestion_config.get( "fallback_fps",30))

        logger.info("Video stream initialized.")

        # ------------------------------------------
        # Detection Pipeline
        # ------------------------------------------

        model_loader = ModelLoader(model_name=detection_settings.get( "model_name","yolov8n"),
                                   weights_path= detection_settings.get( "weights_path", None),
                                   device= detection_settings.get("device","cpu"))

        model_loader.load_model()
        logger.info("YOLO model loaded.")

        inference_engine = YOLOInference(model_loader=model_loader,
                                         confidence_threshold=detection_settings.get("confidence_threshold",0.5))
        inference_engine.warmup()
        logger.info("Inference engine initialized.")
        postprocessor = (DetectionPostProcessor(confidence_threshold=detection_settings.get("confidence_threshold",0.5),
                                                target_classes=detection_settings.get( "target_classes",None)))

        logger.info("Postprocessor initialized.")

        # ------------------------------------------
        # Tracking Pipeline
        # ------------------------------------------

        tracker = ObjectTracker(tracking_settings)
        logger.info("Tracker initialized.")

        # ------------------------------------------
        # Event Pipeline
        # ------------------------------------------

        event_engine = EventEngine(events_config)
        logger.info("Event engine initialized.")

        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        renderer = FrameRenderer()
        logger.info("Renderer initialized.")

        # ------------------------------------------
        # Main Pipeline Loop
        # ------------------------------------------

        while capture_manager.is_active():

            success, frame = (capture_manager.read_frame())
            if not success:
                logger.warning( "Frame read failed.")
                break

            # -----------------------------
            # YOLO Inference
            # -----------------------------
            results = (inference_engine.run_inference(frame))

            # -----------------------------
            # Postprocess
            # -----------------------------
            detections = (postprocessor.process(results))

            # -----------------------------
            # Tracking
            # -----------------------------
            tracks = (tracker.update_tracks(detections))

            # -----------------------------
            # Event Processing
            # -----------------------------
            events = event_engine.process_tracks(tracks)

            # Log events
            for event in events:
                logger.info(
                    f"Event detected: "
                    f"{event['event_type']} "
                    f"(track_id={event['track_id']})"
                )

            # -----------------------------
            # Runtime Metrics
            # -----------------------------
            runtime_stats = (inference_engine.get_runtime_statistics())
            tracking_stats = (tracker.get_stats())
            inference_time = (runtime_stats["last_inference_sec"])

            fps = None

            if inference_time is not None and inference_time > 0:
                fps = (1 / inference_time)

            # -----------------------------
            # Render
            # -----------------------------
            rendered_frame = renderer.render_frame(frame, tracks=tracks, events=events, line_start=line_start,
                                                   line_end=line_end, intrusion_zone=intrusion_zone, fps=fps, inference_time=inference_time)

            cv2.imshow("RTVIS",rendered_frame)

            if (cv2.waitKey(1) & 0xFF== 27):
                logger.info("Exit requested.")
                break

    except Exception as error:
        logger.error(f"Pipeline failed: {error}")

    finally:

        if capture_manager is not None:
            capture_manager.release()

        if "tracker" in locals():
            tracker.reset()

        cv2.destroyAllWindows()

        logger.info("Resources released.")


if __name__ == "__main__":
    main()
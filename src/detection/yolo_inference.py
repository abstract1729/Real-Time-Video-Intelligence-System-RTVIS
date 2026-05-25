'''
Run inference on incoming frames
Responsibilities:

- Input: OpenCV frame
- Run: YOLO inference
- Return: Raw detections
Additionally, measure inference time for performance monitoring.
'''

from typing import Optional
import time
import numpy as np
# from model_loader import ModelLoader
from src.detection.model_loader import ModelLoader

from ultralytics.engine.results import Results


class YOLOInference:
    """
    Handles YOLO execution on incoming frames.

    Responsibilities:
    -----------------
    - Receive OpenCV frames
    - Run YOLO inference
    - Measure inference latency
    - Update runtime metrics
    - Return raw detection output

    Notes:
    ------
    This class DOES NOT:
        - load model weights
        - postprocess detections
        - render bounding boxes
    """

    def __init__(self, model_loader: ModelLoader, confidence_threshold: float = 0.5,verbose: bool = False) -> None:

        # Model Lifecycle Manager
        self.model_loader = model_loader
        self.model = self.model_loader.get_model()

        # Inference Configuration
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        # Runtime State
        self.last_result: Optional[list[Results]] = None
        self.last_frame = None

    def run_inference(self,frame: np.ndarray) -> list[Results]:
        """
        Run YOLO inference on frame.
        Parameters
        ----------
        frame : np.ndarray
        Returns
        -------
        list[Results]

        Notes
        -----
        Output remains RAW.
        Postprocessing happens separately.
        """

        if frame is None: 
            raise ValueError("Input frame cannot be None.")
        
        start_time = time.perf_counter()
        results = self.model.predict(source=frame,conf=self.confidence_threshold,verbose=self.verbose)
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        # Update Runtime Metrics
        self.model_loader.update_inference_metrics(inference_time)

        # Store Runtime State
        self.last_result = results
        self.last_frame = frame

        return results

    def warmup(self,frame_shape=(640, 640, 3)) -> None:
        """
        Perform warmup inference.
        Purpose:
        --------
        Remove first inference latency spike.
        """

        dummy_frame = np.zeros(frame_shape,dtype=np.uint8)
        self.run_inference(dummy_frame)

    def get_last_result(self) -> Optional[list[Results]]:
        """
        Return latest inference result.
        """
        return self.last_result

    def get_runtime_statistics(self) -> dict:
        """
        Return runtime inference stats.
        """
        model_metrics = (self.model_loader.get_runtime_metrics())
        return {
            "confidence_threshold":self.confidence_threshold,
            "last_inference_sec": model_metrics["last_inference_sec"],
            "avg_inference_sec": model_metrics["avg_inference_sec"],
            "total_calls": model_metrics["total_calls"]
        }

    def reset_statistics(self) -> None:
        """
        Reset inference metrics.
        """
        self.model_loader.reset_metrics()


if __name__ == "__main__":
    # Basic test to verify inference execution
    loader = ModelLoader(model_name="yolov8n",device="cpu")
    loader.load_model()
    inference_engine = YOLOInference(model_loader=loader,confidence_threshold=0.5,verbose=False)
    inference_engine.warmup()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = inference_engine.run_inference(dummy_frame)
    print("Inference Results:", results[0].speed)
    print("Inference executed successfully.")
    print("Runtime Statistics:", inference_engine.get_runtime_statistics())
'''
Model lifecycle management.

Responsibilities:
- Load model weights and Track loading status.
- Device handling (CPU/GPU/MPS)
- Configure inference backend
- Store model instance for use in detection pipeline.
'''

import os
from pathlib import Path
from typing import Optional
import time

import torch
from ultralytics import YOLO


class ModelLoader:
    """
    Handles lifecycle management for detection models.

    Responsibilities:
    -----------------
    - Load model weights
    - Manage model state
    - Device selection
    - Runtime monitoring
    - Inference metrics storage

    Notes:
    ------
    This class DOES NOT run inference.
    Use yolo_inference.py for execution.
    """

    VALID_MODELS = {
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x"
    }

    def __init__(self,model_name: str = "yolov8n",weights_path: Optional[str] = None, device="auto") -> None:
        
        # Configuration
        self.model_name = model_name
        self.weights_path = weights_path
        self.device_preference = device

        # Model State
        self.model: Optional[YOLO] = None
        self.is_loaded = False
        self.device = self._select_device()
        

        # Runtime Metrics
        self.model_load_time = None
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
        self.total_inference_calls = 0
        self.last_inference_time = None

    def _select_device(self) -> str:
        """
        Automatically select computation device.
        """
        if self.device_preference != "auto":
            return self.device_preference

        elif torch.cuda.is_available():
            return "cuda"

        elif torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def load_model(self) -> None:
        """
        Load YOLO model weights.
        """

        start_time = time.perf_counter()
        if self.weights_path is not None:
            weight_file = Path(self.weights_path)
            if not weight_file.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_file}")

            self.model = YOLO(str(weight_file))

        else:
            if self.model_name not in self.VALID_MODELS:
                raise ValueError(f"Unsupported model: {self.model_name}")

            model_weight = f"{self.model_name}.pt"
            self.model = YOLO(model_weight)

        self.model.to(self.device)
        end_time = time.perf_counter()
        self.model_load_time = end_time - start_time
        self.is_loaded = True

    def get_model(self) -> YOLO:
        """
        Return initialized model instance.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def update_inference_metrics(self,inference_time: float) -> None:
        """
        Update runtime inference statistics.
        """

        self.last_inference_time = inference_time
        self.total_inference_calls += 1
        self.total_inference_time += inference_time
        self.avg_inference_time = (self.total_inference_time /self.total_inference_calls)

    def get_runtime_metrics(self) -> dict:
        """
        Return runtime model statistics.
        """

        return {
            "device": self.device,
            "loaded": self.is_loaded,
            "load_time_sec": self.model_load_time,
            "last_inference_sec": self.last_inference_time,
            "avg_inference_sec": self.avg_inference_time,
            "total_calls": self.total_inference_calls }

    def reset_metrics(self) -> None:
        """
        Reset inference statistics.
        """

        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
        self.total_inference_calls = 0
        self.last_inference_time = None


if __name__ == "__main__":
    # Basic test to verify model loading
    loader = ModelLoader(model_name="yolov8n",device="cpu")
    loader.load_model()
    model = loader.get_model()
    print(loader.get_runtime_metrics())
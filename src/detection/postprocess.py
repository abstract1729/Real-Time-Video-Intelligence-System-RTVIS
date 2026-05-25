'''
Convert raw model output into usable objects.

Postprocessing converts:
tensor output into:
{
   bbox:
   confidence:
   class_name:
}

Responsibilities:
- Confidence filtering
- Bounding box extraction

This converts YOLO output into a format usable by:
- tracking
- event engine
- visualization
'''


from typing import List, Optional
from ultralytics.engine.results import Results
from model_loader import ModelLoader
from yolo_inference import YOLOInference


class DetectionPostProcessor:
    """
    Converts raw YOLO detections into structured objects.

    Responsibilities:
    -----------------
    - Confidence filtering
    - Class filtering
    - Bounding box extraction
    - Coordinate conversion

    Notes:
    ------
    Output format is intentionally tracker-friendly.
    """

    def __init__(self,confidence_threshold: float = 0.5,target_classes: Optional[list] = None) -> None:

        # Configuration
        self.confidence_threshold = (confidence_threshold)
        self.target_classes = target_classes

        
        # Runtime State
        self.last_detection_count = 0

    def process(self,results: List[Results]) -> list[dict]:
        """
        Convert raw YOLO output to structured detections.
        Parameters
        ----------
        results : List[Results]

        Returns
        -------
        list[dict]

        Detection format:

        {
            bbox:
            confidence:
            class_id:
            class_name:
        }
        """

        processed_detections = []

        if results is None:
            return processed_detections

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            names = result.names
            for box in boxes:
                confidence = float(box.conf.item())

                
                # Confidence Filtering
                if (confidence < self.confidence_threshold):
                    continue

                class_id = int(box.cls.item())
                class_name = names[class_id]

                
                # Class Filtering
                if (self.target_classes is not None):
                    if (class_name not in self.target_classes):
                        continue

                
                # Bounding Box Extraction
                x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy().astype(int))

                detection = {"bbox": [x1,y1,x2,y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name }

                processed_detections.append(detection)

        self.last_detection_count = (len(processed_detections))
        return processed_detections

    def get_detection_count(self) -> int:
        """
        Return latest detection count.
        """
        return self.last_detection_count


if __name__ == "__main__":
      import numpy as np
      loader = ModelLoader(model_name="yolov8n",device="cpu")
      loader.load_model()

      engine = YOLOInference(model_loader=loader)
      processor = DetectionPostProcessor(confidence_threshold=0.5)
      dummy = np.zeros((640,640,3),dtype=np.uint8)

      results = engine.run_inference(dummy)
      print(results[0].boxes is None)
      detections = processor.process(results)
      print(detections)
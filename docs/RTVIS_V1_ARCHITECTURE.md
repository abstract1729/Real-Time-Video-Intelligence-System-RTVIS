# RTVIS V1 Architecture

## Overview

RTVIS V1 implements the first end-to-end computer vision pipeline:

```text
Video Source
      ↓
Ingestion
      ↓
YOLO Detection
      ↓
Postprocessing
      ↓
Rendering
      ↓
Display
```

---

## Implemented Modules

### Ingestion Layer

Files:

```text
src/ingestion/

video_capture.py
stream_loader.py
```

Components:

- `VideoCaptureManager`
  - Handles webcam / video file / RTSP ingestion
  - Stream metadata extraction
  - Runtime monitoring

- `StreamLoader`
  - Config-driven source initialization

---

### Detection Layer

Files:

```text
src/detection/

model_loader.py
yolo_inference.py
postprocess.py
```

Components:

- `ModelLoader`
  - Model lifecycle management
  - Device selection
  - Runtime metrics

- `YOLOInference`
  - Frame-level inference execution

- `DetectionPostProcessor`
  - Bounding box extraction
  - Confidence filtering
  - Class filtering

Detection output:

```python
{
    "bbox": [x1,y1,x2,y2],
    "confidence": 0.91,
    "class_name": "person"
}
```

---

### Visualization Layer

Files:

```text
src/visualization/

overlay.py
renderer.py
```

Components:

- `OverlayDrawer`
  - Bounding boxes
  - Labels
  - Metrics overlays

- `FrameRenderer`
  - Rendering orchestration

---

## Current Capabilities

Implemented:

- Real-time video ingestion
- YOLOv8 inference
- Detection rendering
- FPS display
- Inference latency display

Pending:

- Tracking
- Events
- Storage
- APIs
- Dashboard
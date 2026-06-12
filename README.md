# Real-Time-Video-Intelligence-System-RTVIS-
RTVIS (Real-Time Video Intelligence System) is a modular real-time computer vision system designed to process video streams and convert them into actionable intelligence, which can be used to detect, track, and analyze object behavior. The system transforms raw video into actionable intelligence through event detection (intrusion, loitering, line crossing), and exposes APIs and dashboards for monitoring and control.<br>

The system is being developed as an end-to-end video analytics pipeline supporting:

- Object detection
- Multi-object tracking
- Event intelligence
- Storage and APIs
- Monitoring dashboards


# Real-Time-Video-Intelligence-System (RTVIS)

High Level Architecture:

```text
Video
   ↓
Ingestion
   ↓
Preprocessing
   ↓
Detection
   ↓
Tracking
   ↓
Event Engine
   ↓
Storage / API / UI
```

---

## Current Status

Current implementation: **RTVIS V1**

Implemented features:

- Webcam / video / RTSP ingestion
- YOLOv8 inference
- Detection rendering
- FPS visualization
- Runtime metrics

---

## Repository Structure

```text
src/

ingestion/
detection/
tracking/
events/
pipeline/
storage/
api/
visualization/
utils/
core/
```

Detailed architecture documentation is available in:

```text
docs/
RTVIS_V1_ARCHITECTURE.md
ROADMAP.md
DESIGN_NOTES.md
```

---

## Installation

Create environment:

```bash
pip install -r requirements.txt
```

Install additional dependencies:

```bash
pip install ultralytics opencv-python pyyaml torch torchvision
```

---

## Run Pipeline

Execute:

```bash
python scripts/run_pipeline.py
```

Current output:

- Live video stream
- Bounding boxes
- Labels
- Confidence scores
- FPS
- Inference latency

---

## Roadmap

* V1 — Detection Pipeline ✅
* V2 — Tracking (DeepSORT / ByteTrack) ✅
* V3 — Event Intelligence ✅
* V4 — Storage Layer
* V5 — FastAPI Layer
* V6 — Dashboard Layer

# RTVIS Design Notes

## Design Principles

- Modular architecture
- Config-driven design
- Separation of concerns
- Real-time constraints
- Scalable pipeline design

---

## Ingestion Decisions

### Single `VideoCaptureManager`

Decision:

Single class handles:

- Webcam
- Video file
- RTSP

Reason:

All currently use:

```python
cv2.VideoCapture()
```

Separate classes deferred until behavior diverges.

---

### Postponed `IngestionManager`

Reason:

Current system is synchronous.

Future trigger:

- Threading
- Buffering
- Multi-streams
- Queue systems

---

## Visualization Decisions

### Overlay vs Renderer Separation

`OverlayDrawer`

Responsibility:

```text
HOW to draw
```

Examples:

- bbox
- text
- polygons

---

`FrameRenderer`

Responsibility:

```text
WHAT to draw
WHEN to draw
```

Examples:

- detections
- future tracking
- future events

---

## Detection Decisions

`ModelLoader`

Purpose:

Model lifecycle management.

---

`YOLOInference`

Purpose:

Inference execution only.

---

`DetectionPostProcessor`

Purpose:

Raw output → structured detections.

---

## Future Optimization Targets

- Producer-consumer queues
- Multithreading
- Async pipeline
- Frame buffering
- Multi-stream support
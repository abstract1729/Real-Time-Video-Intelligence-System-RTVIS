# RTVIS Roadmap

## V1 — Detection Pipeline (May 25, 2026)

Pipeline:

```text
Video
   ↓
Detection
   ↓
Visualization
```

Features:

- Video ingestion
- YOLO detection
- Bounding box rendering

---

## V2 — Tracking Layer

Pipeline:

```text
Detection
    ↓
Tracking
```

Planned:

- DeepSORT integration
- ByteTrack integration
- Track persistence
- Trajectory handling

---

## V3 — Event Engine

Pipeline:

```text
Tracking
    ↓
Event Intelligence
```

Planned events:

- Restricted zone intrusion
- Loitering detection
- Line crossing

---

## V4 — Storage Layer

Planned:

- PostgreSQL integration
- Event persistence
- Repository pattern

---

## V5 — API Layer

Planned:

- FastAPI endpoints
- Health monitoring
- Event retrieval APIs

---

## V6 — Dashboard

Planned:

- Streamlit UI
- Analytics dashboard
- Event monitoring
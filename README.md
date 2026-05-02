# SafeGuard Edge — Worker Safety Monitoring System

On-device ML inference only. No cloud APIs. No data egress.

```
safeguard-edge/
├── backend/
│   ├── main.py          ← FastAPI server + WebSocket + ML engine
│   └── requirements.txt
└── frontend/
    └── index.html       ← Full dashboard (zero build step)
```

---

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### 2. Frontend

Just open the HTML file in your browser:

```bash
# Option A: direct open
open frontend/index.html

# Option B: serve with Python (avoids CORS issues)
cd frontend
python3 -m http.server 3000
# then visit http://localhost:3000
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health + inference policy |
| GET | `/api/models` | All on-device models with scores |
| GET | `/api/models/{id}` | Single model details |
| POST | `/api/infer/{model}/{zone}` | Trigger manual inference |
| GET | `/api/zones` | All zones with live readings |
| GET | `/api/workers` | All workers + locations |
| GET | `/api/alerts` | Active alerts |
| POST | `/api/alerts/{zone}/acknowledge` | Acknowledge an alert |
| GET | `/api/events?limit=50` | Inference event log |
| GET | `/api/stats` | System stats + constraint violations |
| GET | `/api/constraints` | Enforcement policy |
| WS | `/ws` | Live state stream (1.5s cadence) |

---

## On-Device ML Models

All models enforced: size < 50MB, latency < 100ms, cloud_api = False

| Model | Format | Size | Latency | Power | Use Case |
|-------|--------|------|---------|-------|----------|
| Gas Anomaly Detector | TFLite INT8 | 18.4 MB | ~34ms | 0.4W | H₂S/CO detection on sensor MCU |
| Fall Detection CV | ONNX INT4 | 32.1 MB | ~88ms | 1.2W | IMU + edge camera fall detection |
| Vitals Classifier | Core ML FP16 | 9.7 MB | ~21ms | 0.2W | PPG/ECG wristband classification |
| Thermal Hotspot Seg | TFLite INT8 | 41.3 MB | ~67ms | 1.8W | U-Net on thermal camera feed |

## Power Efficiency Score Formula

```
score = (latency_headroom × 50) + (size_headroom × 30) + (power_headroom × 20)

where:
  latency_headroom = (100ms - actual_latency) / 100ms
  size_headroom    = (50MB  - model_size)     / 50MB
  power_headroom   = (3W    - power_draw)     / 3W
```

## Enforcement Architecture

```
OnDeviceInferenceEngine
  ├── _validate_all_models()     ← runs at startup, blocks cloud_api=True
  ├── infer()
  │   ├── BLOCK if cloud_api=True
  │   ├── MEASURE latency
  │   ├── ENFORCE latency < 100ms (flag violations)
  │   ├── COMPUTE power_score
  │   └── RUN _classify() → label + severity + action
  └── violation_log[]
```

## WebSocket Protocol

Connect to `ws://localhost:8000/ws`

Receive every 1.5 seconds:
```json
{
  "type": "state_update",
  "zones": { ... },
  "alerts": [ ... ],
  "stats": { "total": 1482, "per_minute": 247, "cloud_blocked": 0 },
  "recent_events": [ ... ],
  "timestamp": "2026-05-02T14:23:07.123"
}
```

Send ping:
```json
{ "type": "ping" }
```

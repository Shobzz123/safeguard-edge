"""
SafeGuard Edge — Worker Safety Monitoring Backend
On-device inference only. No cloud API calls. All ML runs locally.

Run: uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import math
import random
import time
from collections import deque
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="SafeGuard Edge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ON-DEVICE ML MODEL REGISTRY
# Each model enforces: size < 50MB, latency < 100ms
# No cloud API calls allowed — all inference is local
# ─────────────────────────────────────────────

ML_MODELS = {
    "gas_anomaly": {
        "id": "gas_anomaly",
        "name": "Gas Anomaly Detector",
        "format": "TFLite INT8",
        "size_mb": 18.4,
        "target_latency_ms": 100,
        "base_latency_ms": 34,
        "power_watts": 0.4,
        "accuracy": 97.2,
        "cloud_api": False,  # enforced: always False
        "description": "Quantized LSTM on sensor MCU for H2S/CO anomaly detection",
        "input": "gas_sensor_array",
        "hardware": "NPU",
    },
    "fall_detection": {
        "id": "fall_detection",
        "name": "Fall Detection CV",
        "format": "ONNX INT4",
        "size_mb": 32.1,
        "target_latency_ms": 100,
        "base_latency_ms": 88,
        "power_watts": 1.2,
        "accuracy": 93.8,
        "cloud_api": False,
        "description": "4-bit MobileNetV3 on wearable accelerometer + edge camera",
        "input": "imu_camera_fusion",
        "hardware": "GPU",
    },
    "vitals_classifier": {
        "id": "vitals_classifier",
        "name": "Vitals Classifier",
        "format": "Core ML FP16",
        "size_mb": 9.7,
        "target_latency_ms": 100,
        "base_latency_ms": 21,
        "power_watts": 0.2,
        "accuracy": 95.1,
        "cloud_api": False,
        "description": "Apple Neural Engine PPG/ECG signal classifier on wristband",
        "input": "ppg_ecg_wristband",
        "hardware": "ANE",
    },
    "thermal_segmentation": {
        "id": "thermal_segmentation",
        "name": "Thermal Hotspot Seg.",
        "format": "TFLite INT8",
        "size_mb": 41.3,
        "target_latency_ms": 100,
        "base_latency_ms": 67,
        "power_watts": 1.8,
        "accuracy": 91.5,
        "cloud_api": False,
        "description": "U-Net segmentation on thermal camera for hotspot detection",
        "input": "thermal_camera",
        "hardware": "NPU",
    },
}

# ─────────────────────────────────────────────
# WORKER & ZONE STATE
# ─────────────────────────────────────────────

ZONES = {
    "B3": {"id": "B3", "name": "Chemical Storage", "type": "chemical", "model": "gas_anomaly", "unit": "ppm H₂S"},
    "A1": {"id": "A1", "name": "Welding Bay",       "type": "welding",   "model": "gas_anomaly", "unit": "ppm CO"},
    "C2": {"id": "C2", "name": "Assembly Floor",    "type": "assembly",  "model": "fall_detection", "unit": "fall_risk"},
    "D1": {"id": "D1", "name": "Boiler Room",       "type": "thermal",   "model": "thermal_segmentation", "unit": "°C surface"},
    "E4": {"id": "E4", "name": "Cold Storage",      "type": "cold",      "model": "vitals_classifier", "unit": "°C ambient"},
    "F2": {"id": "F2", "name": "Loading Dock",      "type": "loading",   "model": "vitals_classifier", "unit": "HR bpm"},
}

WORKERS = {
    f"W{i:02d}": {
        "id": f"W{i:02d}",
        "name": f"Worker {i:02d}",
        "zone": list(ZONES.keys())[i % len(ZONES)],
        "status": "safe",
        "heart_rate": random.randint(62, 85),
        "temperature": round(36.5 + random.random() * 0.8, 1),
        "last_seen": time.time(),
    }
    for i in range(1, 25)
}

# Assign specific workers to zones
zone_keys = list(ZONES.keys())
for i, wid in enumerate(WORKERS):
    WORKERS[wid]["zone"] = zone_keys[i % len(zone_keys)]

# Sensor readings per zone
ZONE_READINGS = {
    "B3": {"value": 142.0, "min": 0, "max": 200, "critical": 100, "warning": 50},
    "A1": {"value": 78.0,  "min": 0, "max": 150, "critical": 100, "warning": 50},
    "C2": {"value": 0.12,  "min": 0, "max": 1,   "critical": 0.7, "warning": 0.4},
    "D1": {"value": 310.0, "min": 200, "max": 600, "critical": 500, "warning": 400},
    "E4": {"value": -18.0, "min": -30, "max": 10, "critical": -25, "warning": -22},
    "F2": {"value": 68.0,  "min": 50, "max": 120, "critical": 110, "warning": 95},
}

# Event log — circular buffer
event_log = deque(maxlen=200)

# Active alerts
active_alerts = {}

# Inference stats
inference_stats = {
    "total": 0,
    "per_minute": 0,
    "cloud_blocked": 0,
    "latency_violations": 0,
}

# ─────────────────────────────────────────────
# SIMULATED ON-DEVICE INFERENCE ENGINE
# ─────────────────────────────────────────────

class OnDeviceInferenceEngine:
    """
    Simulates on-device ML inference.
    Enforces: no cloud API, size < 50MB, latency < 100ms.
    """

    MAX_MODEL_SIZE_MB = 50.0
    MAX_LATENCY_MS = 100.0

    def __init__(self):
        self.violation_log = []
        self._validate_all_models()

    def _validate_all_models(self):
        for mid, model in ML_MODELS.items():
            if model["size_mb"] > self.MAX_MODEL_SIZE_MB:
                raise ValueError(f"Model {mid} exceeds size limit: {model['size_mb']}MB > {self.MAX_MODEL_SIZE_MB}MB")
            if model["cloud_api"]:
                raise ValueError(f"Model {mid} uses cloud API — POLICY VIOLATION")

    def infer(self, model_id: str, sensor_value: float, zone_id: str) -> dict:
        model = ML_MODELS.get(model_id)
        if not model:
            raise HTTPException(404, f"Model {model_id} not found")

        # ENFORCE: no cloud
        if model["cloud_api"]:
            inference_stats["cloud_blocked"] += 1
            raise ValueError("Cloud API inference BLOCKED by policy")

        # Simulate inference latency with jitter
        jitter = random.gauss(0, 4)
        latency = max(10, model["base_latency_ms"] + jitter)

        # ENFORCE: latency target
        latency_ok = latency <= self.MAX_LATENCY_MS
        if not latency_ok:
            inference_stats["latency_violations"] += 1

        # Compute power efficiency score (0-100)
        # Formula: (latency_headroom * 0.5) + (size_headroom * 0.3) + (power_headroom * 0.2)
        latency_headroom = max(0, (self.MAX_LATENCY_MS - latency) / self.MAX_LATENCY_MS)
        size_headroom = max(0, (self.MAX_MODEL_SIZE_MB - model["size_mb"]) / self.MAX_MODEL_SIZE_MB)
        power_headroom = max(0, (3.0 - model["power_watts"]) / 3.0)
        power_score = round((latency_headroom * 50 + size_headroom * 30 + power_headroom * 20))

        # Determine anomaly / classification result
        result = self._classify(model_id, sensor_value, zone_id)
        confidence = round(random.uniform(88.0, 99.0), 1)

        inference_stats["total"] += 1

        return {
            "model_id": model_id,
            "model_name": model["name"],
            "zone_id": zone_id,
            "latency_ms": round(latency, 1),
            "latency_ok": latency_ok,
            "latency_target_ms": self.MAX_LATENCY_MS,
            "model_size_mb": model["size_mb"],
            "size_limit_mb": self.MAX_MODEL_SIZE_MB,
            "power_watts": model["power_watts"],
            "power_score": power_score,
            "cloud_api_used": False,
            "result": result,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "hardware": model["hardware"],
            "format": model["format"],
        }

    def _classify(self, model_id: str, value: float, zone_id: str) -> dict:
        thresholds = ZONE_READINGS.get(zone_id, {})
        crit = thresholds.get("critical", 100)
        warn = thresholds.get("warning", 50)

        if model_id == "gas_anomaly":
            if value >= crit:
                return {"label": "CRITICAL_GAS", "severity": "critical", "action": "evacuate"}
            elif value >= warn:
                return {"label": "WARNING_GAS", "severity": "warning", "action": "alert_supervisor"}
            return {"label": "NORMAL", "severity": "safe", "action": "none"}

        elif model_id == "fall_detection":
            if value >= crit:
                return {"label": "FALL_DETECTED", "severity": "critical", "action": "dispatch_medic"}
            elif value >= warn:
                return {"label": "INSTABILITY", "severity": "warning", "action": "check_worker"}
            return {"label": "NORMAL_POSTURE", "severity": "safe", "action": "none"}

        elif model_id == "vitals_classifier":
            if value >= crit or value <= thresholds.get("min", -999) + 2:
                return {"label": "VITALS_CRITICAL", "severity": "critical", "action": "emergency"}
            elif value >= warn:
                return {"label": "VITALS_ELEVATED", "severity": "warning", "action": "rest_break"}
            return {"label": "VITALS_NORMAL", "severity": "safe", "action": "none"}

        elif model_id == "thermal_segmentation":
            if value >= crit:
                return {"label": "HOTSPOT_CRITICAL", "severity": "critical", "action": "shutdown_zone"}
            elif value >= warn:
                return {"label": "HOTSPOT_WARNING", "severity": "warning", "action": "increase_cooling"}
            return {"label": "THERMAL_NORMAL", "severity": "safe", "action": "none"}

        return {"label": "UNKNOWN", "severity": "safe", "action": "none"}


engine = OnDeviceInferenceEngine()

# ─────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────

def simulate_sensor_tick():
    """Update sensor readings with realistic drift + occasional spikes."""
    for zone_id, reading in ZONE_READINGS.items():
        zone = ZONES[zone_id]
        model_id = zone["model"]
        model = ML_MODELS[model_id]

        # Drift with mean-reversion
        drift = random.gauss(0, model["base_latency_ms"] * 0.02)
        reading["value"] += drift

        # Occasional spike events
        if random.random() < 0.02:
            spike = (reading["max"] - reading["min"]) * random.uniform(0.1, 0.3)
            reading["value"] += spike * (1 if random.random() > 0.5 else -1)

        # Clamp
        reading["value"] = max(reading["min"], min(reading["max"], reading["value"]))

        # Run on-device inference
        infer_result = engine.infer(model_id, reading["value"], zone_id)
        severity = infer_result["result"]["severity"]
        label = infer_result["result"]["label"]

        # Emit event if noteworthy
        if severity != "safe" or random.random() < 0.05:
            event = {
                "id": f"evt_{int(time.time()*1000)}_{zone_id}",
                "timestamp": datetime.now().isoformat(),
                "zone_id": zone_id,
                "zone_name": ZONES[zone_id]["name"],
                "model": infer_result["model_name"],
                "model_format": infer_result["format"],
                "latency_ms": infer_result["latency_ms"],
                "severity": severity,
                "label": label,
                "value": round(reading["value"], 2),
                "unit": ZONES[zone_id]["unit"],
                "confidence": infer_result["confidence"],
                "cloud_api_used": False,
                "power_score": infer_result["power_score"],
                "action": infer_result["result"]["action"],
            }
            event_log.appendleft(event)

            # Manage active alerts
            if severity == "critical":
                active_alerts[zone_id] = event
            elif zone_id in active_alerts and severity == "safe":
                del active_alerts[zone_id]


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)


manager = ConnectionManager()

# ─────────────────────────────────────────────
# BACKGROUND TASK: sensor loop + broadcast
# ─────────────────────────────────────────────

async def sensor_loop():
    start = time.time()
    while True:
        simulate_sensor_tick()

        # Update inference per-minute stat
        elapsed = (time.time() - start) / 60
        inference_stats["per_minute"] = int(inference_stats["total"] / max(elapsed, 0.001))

        payload = {
            "type": "state_update",
            "zones": {
                zid: {
                    **ZONES[zid],
                    "value": round(ZONE_READINGS[zid]["value"], 2),
                    "unit": ZONES[zid]["unit"],
                    "status": _zone_status(zid),
                    "workers": [w for w, d in WORKERS.items() if d["zone"] == zid],
                }
                for zid in ZONES
            },
            "alerts": list(active_alerts.values()),
            "stats": {
                **inference_stats,
                "active_workers": len(WORKERS),
                "total_zones": len(ZONES),
                "cloud_blocked": True,
            },
            "recent_events": list(event_log)[:10],
            "timestamp": datetime.now().isoformat(),
        }
        await manager.broadcast(payload)
        await asyncio.sleep(1.5)


def _zone_status(zone_id: str) -> str:
    r = ZONE_READINGS[zone_id]
    v = r["value"]
    if v >= r["critical"] or v <= r.get("min", -999) + 2:
        return "critical"
    elif v >= r["warning"]:
        return "warning"
    return "safe"


@app.on_event("startup")
async def startup():
    asyncio.create_task(sensor_loop())

# ─────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "cloud_api": False, "inference_engine": "on_device", "version": "1.0.0"}


@app.get("/api/models")
def get_models():
    enriched = {}
    for mid, m in ML_MODELS.items():
        score_data = engine.infer(mid, ZONE_READINGS.get(
            next((z for z, d in ZONES.items() if d["model"] == mid), "B3"), {}).get("value", 50), "B3")
        enriched[mid] = {**m, "power_score": score_data["power_score"], "last_latency_ms": score_data["latency_ms"]}
    return enriched


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    if model_id not in ML_MODELS:
        raise HTTPException(404, "Model not found")
    return ML_MODELS[model_id]


@app.post("/api/infer/{model_id}/{zone_id}")
def trigger_inference(model_id: str, zone_id: str, value: Optional[float] = None):
    if zone_id not in ZONE_READINGS:
        raise HTTPException(404, "Zone not found")
    v = value if value is not None else ZONE_READINGS[zone_id]["value"]
    result = engine.infer(model_id, v, zone_id)
    return result


@app.get("/api/zones")
def get_zones():
    return {
        zid: {
            **ZONES[zid],
            "value": round(ZONE_READINGS[zid]["value"], 2),
            "status": _zone_status(zid),
            "workers": [w for w, d in WORKERS.items() if d["zone"] == zid],
            "thresholds": {k: v for k, v in ZONE_READINGS[zid].items() if k != "value"},
        }
        for zid in ZONES
    }


@app.get("/api/workers")
def get_workers():
    return WORKERS


@app.get("/api/alerts")
def get_alerts():
    return list(active_alerts.values())


@app.post("/api/alerts/{zone_id}/acknowledge")
def ack_alert(zone_id: str):
    if zone_id in active_alerts:
        alert = active_alerts.pop(zone_id)
        event_log.appendleft({
            **alert,
            "id": f"ack_{int(time.time()*1000)}",
            "label": "ALERT_ACKNOWLEDGED",
            "severity": "info",
            "timestamp": datetime.now().isoformat(),
        })
        return {"acknowledged": True, "zone_id": zone_id}
    raise HTTPException(404, "No active alert for zone")


@app.get("/api/events")
def get_events(limit: int = 50):
    return list(event_log)[:limit]


@app.get("/api/stats")
def get_stats():
    return {
        **inference_stats,
        "models": len(ML_MODELS),
        "zones": len(ZONES),
        "workers": len(WORKERS),
        "active_alerts": len(active_alerts),
        "cloud_api_enforced": True,
        "max_model_size_mb": engine.MAX_MODEL_SIZE_MB,
        "max_latency_ms": engine.MAX_LATENCY_MS,
        "constraint_violations": inference_stats["latency_violations"],
    }


@app.get("/api/constraints")
def get_constraints():
    """Returns the enforcement policy — on-device only."""
    return {
        "cloud_api_allowed": False,
        "max_model_size_mb": 50.0,
        "max_latency_ms": 100.0,
        "power_efficiency_scoring": True,
        "data_egress_blocked": True,
        "models_compliant": all(
            m["size_mb"] <= 50 and not m["cloud_api"]
            for m in ML_MODELS.values()
        ),
    }


# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# VoiceShield Streaming WebSocket Protocol Specification (v1.0)

## 1. Overview & Operational Contract

The VoiceShield streaming protocol provides real-time, low-latency synthetic speech detection over WebSocket connections (`ws://` / `wss://`). It couples stateful polyphase audio preprocessing, Voice Activity Detection (VAD), and acoustic deepfake classification with a temporal decision policy that tracks anomalies over continuous sliding windows.

### Strict Scope & Compliance Disclaimers
- **Advisory Likelihood Only**: Outputs reflect synthetic acoustic indicators. They do **not** verify speaker identity and do **not** compute fraud probability.
- **Local Audio Only**: Designed for microphone or local audio feeds; does not intercept remote carrier cellular telephony.
- **No Raw Audio Retention**: Audio is resampled and buffered strictly in volatile memory. No audio frames are written to disk.

---

## 2. Protocol State Machine

```
   [ Client Connects ]
           |
           v
  +------------------+
  | HANDSHAKE_WAIT   | ---- (Timeout / Malformed / Unsupported) ----> [ Close 1008/1003 ]
  +------------------+
           |
   Valid 'start' JSON
           |
           v
  +------------------+
  | STREAMING_ACTIVE | <=========================+
  +------------------+                           |
           |                                     | Binary PCM Frame
           |--- Binary Frame --------------------+ (VAD, Resampling, Windowing)
           |--- Eligible Window -----------------> [ Asynchronous Model Inference ]
           |                                                |
           |<-- 'window_result' Event ----------------------+
           |
           |--- 'stop' JSON ---------------------> [ 'session_stopped' ACK ] ---> [ Normal Close 1000 ]
           |--- Disconnect ----------------------> [ Resource Cleanup ] ---------> [ Terminated ]
           |--- Idle Timeout / Oversized Frame --> [ 'error' Event ] -----------> [ Close 1008/1009 ]
```

---

## 3. WebSocket Endpoint & Origin Policy

- **Path**: `/api/v1/stream`
- **Origin Enforcement**: Validated against `VOICESHIELD_CORS_ORIGINS` / `app.state.allowed_origins`. Connections with unauthorized `Origin` headers are immediately rejected with close code `1008 (Policy Violation)`.
- **Concurrency Guard**: Default limit of 10 concurrent active streaming sessions to prevent memory pressure on Apple Silicon unified memory. Excess connections are rejected with `CONCURRENCY_LIMIT_REACHED` and close code `1008`.

---

## 4. Message Specifications

### 4.1 Client Handshake (`start`)
Immediately upon establishing the WebSocket connection, the client must transmit a JSON text message declaring stream parameters:

```json
{
  "type": "start",
  "protocol_version": "1.0",
  "sample_rate": 48000,
  "channels": 1,
  "pcm_encoding": "int16",
  "metadata": {
    "client_id": "web-browser-mic",
    "idle_timeout_seconds": 30.0,
    "max_frame_bytes": 131072
  }
}
```

#### Fields
| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | Yes | Must be `"start"`. |
| `protocol_version` | string | Yes | Supported version: `"1.0"`. |
| `sample_rate` | integer | Yes | Declared source sample rate in Hz ($8,000$ to $192,000$). |
| `channels` | integer | Yes | Number of channels ($1$ for mono, $2$ for stereo). Stereo is averaged to mono. |
| `pcm_encoding` | string | Yes | Supported formats: `"int16"`, `"int32"`, or `"float32"`. |
| `metadata` | object | No | Optional key-value pairs (e.g. `idle_timeout_seconds`, `max_frame_bytes`). |

---

### 4.2 Server Handshake Acknowledgement (`session_started`)
Upon successful validation, the server generates a unique UUID session ID, initializes an isolated `StreamingSession`, and responds:

```json
{
  "type": "session_started",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "protocol_version": "1.0",
  "canonical_sample_rate": 16000,
  "canonical_channels": 1,
  "window_seconds": 4.0,
  "hop_seconds": 1.0,
  "min_speech_seconds": 0.8,
  "max_frame_bytes": 131072,
  "max_stream_duration_seconds": 3600.0,
  "idle_timeout_seconds": 30.0,
  "disclaimers": {
    "scope_limitations": "Advisory only. Does not prove identity or calculate fraud probability.",
    "model_name": "VoiceShield-AASIST-L"
  }
}
```

---

### 4.3 Client Binary Audio Frames
After receiving `session_started`, the client streams raw binary audio frames matching the declared `pcm_encoding`, `channels`, and `sample_rate`.

#### Operational Invariants
- **Chunk-Boundary Invariance**: Audio is statefully resampled to canonical 16 kHz mono. Filter state is preserved across chunks, producing seam-free waveforms.
- **Maximum Frame Size**: Default 128 KB (131,072 bytes). Frames exceeding this limit trigger a `PAYLOAD_TOO_LARGE` error and close code `1009`.
- **Idle Timeout**: If no frames or text messages are received within `idle_timeout_seconds` (default 30.0s), the server sends an `IDLE_TIMEOUT` error and closes cleanly.
- **Stream Duration**: Maximum 3600.0s (1 hour) continuous stream duration before graceful termination.

---

### 4.4 Server Quality Event (`quality_update`)
Emitted when signal quality anomalies are detected on incoming frames:

```json
{
  "type": "quality_update",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "timestamp": 12.35,
  "is_clipped": true,
  "is_silent": false,
  "has_timing_gap": false,
  "clipping_count": 3,
  "gap_count": 0
}
```

---

### 4.5 Server Window Result Event (`window_result`)
Emitted every 1.0-second hop once a 4.0-second window is evaluated:

#### Eligible Window (Active Speech $\ge 0.8$s)
```json
{
  "type": "window_result",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "window_index": 4,
  "window_start_sec": 4.0,
  "window_end_sec": 8.0,
  "speech_seconds": 3.75,
  "is_eligible": true,
  "synthetic_score": 0.82,
  "uncalibrated_synthetic_score": 0.82,
  "score_is_calibrated": false,
  "operating_threshold": 0.65,
  "operating_threshold_source": "unvalidated_heuristic",
  "raw_score": 0.82,
  "calibrated_score": null,
  "decision": {
    "state": "high",
    "synthetic_score": 0.82,
    "score_is_calibrated": false,
    "calibrated_score": null,
    "smoothed_score": 0.79,
    "reason": "sustained_anomaly_elevation",
    "reason_details": "Elevated to HIGH: 3 of last 3 windows exceeded high threshold (0.65, unvalidated heuristic).",
    "window_index": 4,
    "history_scores": [0.71, 0.76, 0.79]
  }
}
```

> [!NOTE]
> `calibrated_score` is strictly `null` and `score_is_calibrated` is `false` until an empirical calibrator is fitted and loaded. The threshold `0.65` is an unvalidated heuristic. When scores are low, the decision reason details report: *"Low acoustic anomaly score — does not authenticate speech."*

#### Abstained Window (Active Speech $< 0.8$s)
```json
{
  "type": "window_result",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "window_index": 5,
  "window_start_sec": 5.0,
  "window_end_sec": 9.0,
  "speech_seconds": 0.32,
  "is_eligible": false,
  "synthetic_score": null,
  "uncalibrated_synthetic_score": null,
  "score_is_calibrated": false,
  "raw_score": null,
  "calibrated_score": null,
  "decision": {
    "state": "insufficient_evidence",
    "synthetic_score": null,
    "score_is_calibrated": false,
    "calibrated_score": null,
    "smoothed_score": 0.79,
    "reason": "insufficient_speech_duration",
    "reason_details": "Active speech (0.32s) is below required minimum threshold (0.80s). Decision abstained.",
    "window_index": 5,
    "history_scores": [0.71, 0.76, 0.79]
  }
}
```

---

### 4.6 Client Graceful Termination (`stop`)
The client can terminate the stream gracefully at any time by sending:

```json
{
  "type": "stop"
}
```

Server replies with confirmation and closes the connection with code `1000 (Normal Closure)`:

```json
{
  "type": "session_stopped",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "total_duration_seconds": 24.5,
  "total_windows_evaluated": 21
}
```

---

### 4.7 Server Error Event (`error`)
Emitted prior to closing the connection when protocol or operational constraints are violated:

```json
{
  "type": "error",
  "error_code": "PAYLOAD_TOO_LARGE",
  "message": "Audio frame size (150000 bytes) exceeds maximum permitted limit (131072 bytes).",
  "details": {}
}
```

#### Machine-Readable Error Codes
- `HANDSHAKE_TIMEOUT`: Initial handshake message not received within timeout.
- `INVALID_JSON`: Non-JSON text received during handshake.
- `INVALID_HANDSHAKE`: First message type was not `"start"`.
- `INVALID_STREAM_CONFIG`: Field validation failed on start message.
- `UNSUPPORTED_PROTOCOL_VERSION`: Version other than `"1.0"`.
- `UNSUPPORTED_PCM_ENCODING`: Format other than `"int16"`, `"int32"`, `"float32"`.
- `PAYLOAD_TOO_LARGE`: Frame bytes exceeded `max_frame_bytes`.
- `STREAM_DURATION_EXCEEDED`: Stream exceeded `max_stream_duration_seconds`.
- `IDLE_TIMEOUT`: Inactivity exceeded `idle_timeout_seconds`.
- `CONCURRENCY_LIMIT_REACHED`: Server-wide active session limit reached.
- `MODEL_INFERENCE_ERROR`: Model inference threw an unexpected runtime exception.

---

## 5. Health vs Readiness Endpoints

The API distinguishes process liveness from neural model readiness:

### 1. Liveness Probe (`GET /health`)
- Returns `200 OK` confirming the FastAPI process is alive, reporting hardware acceleration status and disclaimers.

### 2. Readiness Probe (`GET /ready`)
- Returns `200 OK` if the deepfake detector model is initialized and ready to serve inference.
- Returns `503 Service Unavailable` if model weights failed to load or are not initialized.

---

## 6. Backpressure & Concurrency Guarantees

1. **Non-blocking Event Loop**: Ingestion executes only synchronous DSP and ring-buffering. Model forward passes are dispatched via `asyncio.to_thread` with an `asyncio.Semaphore(1)` hardware lock.
2. **At Most One In-Flight Inference**: At most one inference task runs per session at any time.
3. **Drop / Replace Stale Pending Windows**: If an eligible window is generated while inference is already running, the session replaces any pending window with the newest one, preventing unbounded memory queues.
4. **Session Cleanup**: On disconnect, `StreamingSession.reset()` wipes all ring buffers, pointers, resamplers, and temporal histories.

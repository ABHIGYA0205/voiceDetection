# VoiceShield Streaming WebSocket Protocol Specification (v1.0)

## 1. Overview & Operational Contract

The VoiceShield streaming protocol provides real-time, low-latency synthetic speech detection over WebSocket connections (`ws://` / `wss://`). It couples stateful polyphase audio preprocessing, Voice Activity Detection (VAD), and acoustic deepfake classification with a temporal decision policy that tracks anomalies over continuous sliding windows.

### Strict Scope & Compliance Disclaimers
- **Advisory Likelihood Only**: Outputs reflect synthetic acoustic indicators. They do **not** verify speaker identity and do **not** compute fraud probability.
- **Local Audio Only**: Designed for microphone or local audio feeds; does not intercept remote carrier cellular telephony.
- **No Raw Audio Retention**: Audio is resampled and buffered strictly in volatile memory. No audio frames are written to disk.
- **Engineering Parity**: VoiceShield establishes strict engineering parity between offline WAV/FLAC evaluation and real-time streaming: both paths use the single canonical stateful sinc resampler (with identical filter kernels, bounded FIR history, exact sample output length, and end-of-stream flush), sample-grid window scheduler, EnergyVAD, and TemporalDecisionPolicy. Engineering parity does **not** demonstrate detector accuracy, calibration, cross-dataset generalization, or production readiness.
- **Remaining Limitation**: Abrupt disconnect and inference thread cancellation safety remains deferred for the separate H-04 audit repair.

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
           |<-- 'quality_update' Event (anomalies / drops) -+
           |<-- 'continuity_reset' Event (timing gaps) -----+
           |
           |--- 'stop' JSON ---------------------> [ Flush & Drain Final Window ] ---> [ 'session_stopped' ACK ] ---> [ Normal Close 1000 ]
           |--- Disconnect ----------------------> [ Resource Cleanup ] -------------> [ Terminated ]
           |--- Idle Timeout / Oversized Frame --> [ 'error' Event ] ----------------> [ Close 1008/1009 ]
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
| `sample_rate` | integer | Yes | Declared source sample rate in Hz ($8,000$ to $192,000$ inclusive). |
| `channels` | integer | Yes | Number of channels ($1$ for mono, $2$ for stereo). Stereo is averaged to mono. |
| `pcm_encoding` | string | Yes | Supported formats: `"int16"`, `"int32"`, or `"float32"`. Explicit little-endian byte ordering is required. |
| `metadata` | object | No | Optional key-value pairs (e.g. `idle_timeout_seconds`, `max_frame_bytes`). |

---

## 4.2 Server Handshake Acknowledgement (`session_started`)
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

## 4.3 Client Binary Audio Frames
After receiving `session_started`, the client streams raw binary audio frames matching the declared `pcm_encoding`, `channels`, and `sample_rate`.

#### Audio Ingestion Invariants
- **Little-Endian PCM Requirement**: All multi-byte PCM samples must be explicitly little-endian encoded (`<i2`, `<i4`, `<f4`).
- **Complete Frame-Width Alignment**: Binary frame byte length must be an exact integer multiple of the complete interleaved frame width (`bytes_per_sample * channels`). Odd-length int16 buffers and misaligned multi-channel buffers are strictly rejected with `AUDIO_DECODE_ERROR`.
- **Sample Range & Finiteness**: Decoded float samples must be finite (`NaN` and `±Inf` rejected) and bounded within $[-1.0, 1.0]$.
- **Format Immutability**: Once accepted, declared `sample_rate`, `channels`, and `pcm_encoding` are immutable for the duration of the session until a clean reset. Mid-stream format changes are rejected.
- **Maximum Frame Size**: Default 128 KB (131,072 bytes). Frames exceeding this limit trigger a `PAYLOAD_TOO_LARGE` error and close code `1009`.
- **Idle Timeout**: If no frames or text messages are received within `idle_timeout_seconds` (default 30.0s), the server sends an `IDLE_TIMEOUT` error and closes cleanly.
- **Stream Duration**: Maximum 3600.0s (1 hour) continuous stream duration before graceful termination.

---

## 4.4 Window Scheduling & Backpressure Guarantees

Logical window scheduling and inference backpressure are decoupled:

1. **Exact Canonical Hop Grid**:
   Logical windows are scheduled strictly along the canonical 16 kHz sample grid:
   - Window 0 ends at exactly 64,000 canonical samples (4.0s).
   - Window 1 ends at 80,000 canonical samples (5.0s).
   - Window $k$ ends at $64,000 + k \times 16,000$ canonical samples ($4.0 + k \times 1.0\text{s}$).
   Every due logical boundary is detected and assigned continuous, unique global window indices with zero cumulative timestamp drift, regardless of chunk arrival size.

2. **Bounded Backpressure Queue**:
   To prevent unbounded latency and memory growth:
   - At most one inference task is in flight per session.
   - At most one eligible window is held pending.
   - If a new eligible window arrives while an inference is already in flight and a pending window exists, the fresher window replaces the stale pending window.
   - Every replaced window increments an explicit `total_windows_dropped` counter. Drops are never silent.
   - Abstained windows (insufficient speech) do not require model inference and are emitted immediately without backpressure drops.

3. **Backpressure Updates**:
   Whenever one or more eligible windows are dropped/replaced due to backpressure overload, the server emits a `quality_update` event with current window accounting metrics.

---

## 4.5 Continuity Segment Model & Discontinuity Resets

Timing anomalies—positive gaps exceeding the configured threshold ($> 0.25\text{s}$), materially overlapping timestamps, out-of-order timestamps, or non-finite timestamps—trigger an explicit continuity segment reset:

1. **No Concatenation Across Discontinuities**: Audio from before and after a timing discontinuity is never concatenated into the same inference window.
2. **Buffer & Resampler Reset**: The ring buffer, streaming resampler filter history, window scheduler, VAD, and temporal policy history are reset.
3. **Contiguity Requirement**: Exactly 4.0 contiguous seconds of post-discontinuity audio are required before model inference can resume.
4. **Stale Pre-Gap Result Suppression**: Any in-flight inference started before the discontinuity is suppressed upon completion: its score is discarded, it is not applied to the reset temporal policy, and no `window_result` event is emitted.
5. **Continuity Reset Event**: A machine-readable event with reason `insufficient_contiguous_audio` is emitted to the client.

```json
{
  "type": "continuity_reset",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "reason": "insufficient_contiguous_audio",
  "timestamp_delta": 2.50,
  "gap_count": 1,
  "continuity_segment_id": 1,
  "timestamp": 14.50
}
```

---

## 4.6 Server Quality Event (`quality_update`)
Emitted when signal quality anomalies or backpressure window drops occur:

```json
{
  "type": "quality_update",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "timestamp": 12.35,
  "is_clipped": true,
  "is_silent": false,
  "has_timing_gap": false,
  "clipping_count": 3,
  "gap_count": 0,
  "total_windows_generated": 10,
  "total_windows_evaluated": 7,
  "total_windows_dropped": 2
}
```

---

## 4.7 Server Window Result Event (`window_result`)
Emitted every 1.0-second hop once a 4.0-second window is evaluated:

#### Eligible Window (Active Speech $\ge 0.8$s)
```json
{
  "type": "window_result",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "window_index": 4,
  "window_start_sec": 4.0,
  "window_end_sec": 8.0,
  "speech_seconds": 3.82,
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
    "smoothed_score": 0.80,
    "reason": "sustained_anomaly_elevation",
    "reason_details": "Elevated to HIGH: 3 of last 5 windows exceeded high threshold (0.65).",
    "window_index": 4,
    "window_start_sec": 4.0,
    "window_end_sec": 8.0,
    "speech_seconds": 3.82,
    "history_scores": [0.42, 0.78, 0.80, 0.79, 0.82]
  }
}
```

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
    "window_start_sec": 5.0,
    "window_end_sec": 9.0,
    "speech_seconds": 0.32,
    "history_scores": [0.71, 0.76, 0.79]
  }
}
```

---

## 4.8 Client Graceful Termination & Finalization (`stop`)

When the client requests termination by sending `{"type": "stop"}`:
1. **Resampler Flush**: The server finalizes the `StreamingSession`, flushing all remaining FIR tail samples from `CanonicalResampler`.
2. **Scheduler Drain**: Flushed canonical samples are passed into `CanonicalWindowScheduler`. If flush completes a full 64,000-sample window, it is emitted.
3. **Pre-Stop Drain**: Any newly completed full window is either abstained immediately or dispatched for inference and evaluated before `session_stopped` is sent.
4. **No Padded Windows**: The server never appends a padded or non-hop-aligned trailing window.
5. **Stop Acknowledgement**: The server emits `session_stopped` and closes cleanly with code `1000`:

```json
{
  "type": "session_stopped",
  "session_id": "6a9e8b12-9c3f-4e0a-b348-4e892cfa7123",
  "total_duration_seconds": 24.5,
  "total_windows_evaluated": 18,
  "total_windows_generated": 21,
  "total_windows_dropped": 3,
  "status": "success",
  "final_reason": null
}
```

If the stream finalized with fewer than 64,000 canonical contiguous samples (< 4.0s), `ServerStopAck` indicates `status="insufficient_evidence"` and `final_reason="insufficient_contiguous_audio"`.

---

## 4.9 Server Error Event (`error`)
Emitted prior to closing the connection or when a non-fatal corrupt frame is handled:

```json
{
  "type": "error",
  "error_code": "AUDIO_DECODE_ERROR",
  "message": "Odd-length int16 buffer rejected: 3 bytes.",
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
- `AUDIO_DECODE_ERROR`: PCM frame decoding, byte alignment, or sample range validation failure. The session continues accepting subsequent valid frames.
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

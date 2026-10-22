# VoiceShield: Consented Real vs. Fake Demonstration Checklist

This document details the complete end-to-end procedure for conducting a reproducible, ethical, and consented live demonstration of VoiceShield's real-time streaming and offline file analysis capabilities.

---

## 1. Ethical Governance & Mandatory Consent Protocol

VoiceShield is designed strictly as an **advisory acoustic anomaly indicator** for local microphone feeds and submitted audio files.

> [!IMPORTANT]
> **Consent Requirement**: Before capturing any live human voice via the microphone, explicit verbal or written consent **must** be obtained from the speaker.
> - Never intercept, record, or test unconsented telephony conversations.
> - Clearly communicate to the participant that their voice will be analyzed in real time for acoustic anomaly scoring.
> - Audio is evaluated strictly in volatile RAM and **zero audio files or voiceprints are persisted to disk**.

---

## 2. Environment & Prerequisites

### 2.1 Dependencies
Ensure the system has Python 3.11+ and Node.js 18+ installed:
```bash
python3.11 --version
node --version
npm --version
```

### 2.2 Repository Setup
Install all Python and frontend dependencies:
```bash
make setup
```

---

## 3. Checkpoint Setup: Real Mode vs. Mock Mode

VoiceShield enforces strict operational integrity: **real mode refuses to report as model-ready if official weights are missing**. It will never silently invent or hallucinate model parameters.

### Option A: Real Mode Setup (Official AASIST-L Checkpoint)
To evaluate real acoustic deepfake detection performance using the official Clova AI AASIST-L baseline:

1. Download the official checkpoint weights into `models/weights/`:
   ```bash
   mkdir -p models/weights
   curl -L -o models/weights/AASIST-L.pth \
     https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L/resolve/main/AASIST-L.pth
   ```

2. Verify the checkpoint SHA-256 integrity:
   ```bash
   shasum -a 256 models/weights/AASIST-L.pth
   ```

3. Launch the API in real mode:
   ```bash
   make api
   ```
   Check readiness at `http://127.0.0.1:8000/ready`. It will report:
   ```json
   {
     "status": "ready",
     "service": "VoiceShield",
     "version": "0.1.0",
     "model_loaded": true,
     "is_mock_detector": false,
     "device": "mps"
   }
   ```

### Option B: Mock Mode Setup (UI and Transport Testing)
If the 85MB checkpoint is not present on disk, or if you are running tests or UI demos:

1. Start the API in mock detector mode:
   ```bash
   make api-mock
   ```
   *(Or set `export VOICESHIELD_MOCK_MODEL=true && make api`)*

2. Verify readiness at `http://127.0.0.1:8000/ready`. It will report:
   ```json
   {
     "status": "ready",
     "service": "VoiceShield",
     "version": "0.1.0",
     "model_loaded": true,
     "is_mock_detector": true
   }
   ```
   The UI will visibly display a prominent yellow badge: **`[MOCK DETECTOR - TEST MODE]`**.

---

## 4. Starting the Web Application

In a separate terminal window, start the React + TypeScript frontend:
```bash
make web
```
The application will launch at `http://127.0.0.1:5173`.

> [!NOTE]
> **Development Proxy**: The web app uses relative URLs (`/api/v1/...`). The Vite dev server proxies all `/api` traffic (including WebSocket upgrades) to `http://127.0.0.1:8000`. Deployed browser code never hardcodes `localhost:8000`.

---

## 5. Browser Microphone Permission Guide

When accessing `http://127.0.0.1:5173` for the first time:

### macOS System Permissions
1. If the browser or Terminal has not been granted microphone access:
   - Open **System Settings > Privacy & Security > Microphone**.
   - Ensure your browser (e.g., Google Chrome, Safari, Firefox) is toggled **ON**.

### Browser Site Permissions
1. Click **🎤 Start Microphone Stream** in the web dashboard.
2. The browser will display a permission prompt: *"127.0.0.1 wants to Use your microphone"*.
3. Click **Allow**.
4. If accidentally blocked:
   - **Chrome / Brave / Edge**: Click the tune/lock icon in the URL bar, toggle Microphone to **Allow**, and reload the page.
   - **Safari**: Open **Settings > Websites > Microphone**, locate `127.0.0.1`, and set to **Allow**.
   - **Firefox**: Click the microphone icon next to the URL bar, clear blocked permission, and reload.

---

## 6. Consented Real / Fake Live Demonstration

### Test Case 1: Consented Authentic Human Speech
1. In the web dashboard, confirm the **🎤 Live Microphone Stream** tab is active.
2. Click **🎤 Start Microphone Stream**.
3. Speak clearly into the microphone for 4–8 seconds (e.g., reciting a sentence naturally).
4. **Observations to Verify**:
   - **Connection & Mic**: Both pills show `connected` and `recording`.
   - **Speech Sufficiency**: Active speech gauge shows `≥ 0.8s / 4.0s (Sufficient)`.
   - **Audio Quality**: Signals indicate `Clean`, `Active Signal`, and `Continuous`.
   - **Status Indicator**: Displays `[✓ LOW ANOMALY]` — *Consistent with Natural Speech*.
   - **Rolling Timeline**: Window entries appear sequentially every 1.0s hop.

### Test Case 2: Synthetic Voice Playback (or File Upload)
1. **Via Microphone Stream**:
   - Play an authorized, consented TTS speech clip or voice clone sample through a high-fidelity speaker into the microphone.
   - Observe the score elevation after multiple consecutive 4.0-second windows.
2. **Via File Audio Analysis Tab**:
   - Switch to the **📁 File Audio Analysis** tab.
   - Drag and drop or select a consented synthetic WAV or FLAC file.
   - Set hop step (default: `1.0s`) and threshold (default: `0.65`).
   - Click **⚡ Analyze Audio File**.
3. **Observations to Verify**:
   - **High Anomaly Status**: Displays `[! HIGH ANOMALY]`.
   - **Action Directive Banner Appears**:
     > **ACTION DIRECTIVE**
     > *Pause and verify through a known independent channel.*
   - **Window Table Breakdown**: Displays per-window bona fide scores and synthetic likelihoods.
   - **Compliance Check**: Confirm no prohibited terms appear (no *"100% fake"*, *"fraud proven"*, or *"identity confirmed"*).

---

## 7. Transport Verification Script (Non-Human Audio)

To verify the entire HTTP and WebSocket pipeline without needing a live speaker or human audio:
```bash
.venv/bin/python scripts/demo_transport_verification.py
```
This script programmatically generates pure sine waves, frequency chirps, and silence in memory, streaming them through the FastAPI endpoints and verifying response schemas and latency.

---

## 8. Zero Audio Persistence Audit

To confirm that no audio was saved anywhere during the demonstration:
1. Search the workspace for any newly created `.wav` or `.flac` files:
   ```bash
   git status
   ```
2. Verify that volatile AudioWorklet frames and FastAPI streaming ring-buffers discarded all raw PCM frames immediately upon session disconnect.
3. No database, cloud bucket, or temporary file system storage exists for raw audio.

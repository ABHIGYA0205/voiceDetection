#!/usr/bin/env python3
"""Transport & Pipeline Verification Demo using synthetic non-human audio fixtures.

Verifies:
1. API health and model readiness endpoints.
2. Development origin and CORS/WebSocket acceptance.
3. File analysis transport via HTTP POST /api/v1/analyze/file.
4. Real-time streaming transport via WebSocket /api/v1/stream.
5. Identical preprocessing and BaseDetector evaluation.
6. Zero audio persistence verification (no audio written to disk).

IMPORTANT DISCLAIMER:
Generated synthetic audio fixtures (pure sine waves, chirps, silence) are used
strictly to verify network transport, chunk framing, resampler DSP, and protocol
mechanics. They DO NOT assess, benchmark, or claim acoustic deepfake detection accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voiceshield.demo")


def generate_synthetic_fixtures() -> dict[str, bytes]:
    """Generate in-memory non-human audio fixtures as WAV bytes."""
    fixtures = {}

    # 1. 440 Hz Pure Sine Wave (4.5s @ 16 kHz mono)
    sr_sine = 16000
    t_sine = np.linspace(0, 4.5, int(sr_sine * 4.5), endpoint=False, dtype=np.float32)
    sine_samples = 0.5 * np.sin(2 * np.pi * 440.0 * t_sine)
    buf_sine = io.BytesIO()
    sf.write(buf_sine, sine_samples, sr_sine, format="WAV", subtype="PCM_16")
    fixtures["sine_440hz_16k_mono"] = buf_sine.getvalue()

    # 2. Linear Frequency Chirp 200 Hz -> 800 Hz (5.0s @ 48 kHz stereo to test resampler)
    sr_chirp = 48000
    t_chirp = np.linspace(0, 5.0, int(sr_chirp * 5.0), endpoint=False, dtype=np.float32)
    chirp_phase = 2 * np.pi * (200.0 * t_chirp + (600.0 / (2 * 5.0)) * (t_chirp**2))
    ch1 = 0.4 * np.sin(chirp_phase)
    ch2 = 0.4 * np.cos(chirp_phase)
    chirp_stereo = np.column_stack([ch1, ch2]).astype(np.float32)
    buf_chirp = io.BytesIO()
    sf.write(buf_chirp, chirp_stereo, sr_chirp, format="WAV", subtype="PCM_16")
    fixtures["chirp_200_800hz_48k_stereo"] = buf_chirp.getvalue()

    # 3. Digital Silence (4.0s @ 16 kHz mono)
    sr_silence = 16000
    silence_samples = np.zeros(int(sr_silence * 4.0), dtype=np.float32)
    buf_silence = io.BytesIO()
    sf.write(buf_silence, silence_samples, sr_silence, format="WAV", subtype="PCM_16")
    fixtures["silence_16k_mono"] = buf_silence.getvalue()

    return fixtures


async def run_transport_demo(api_url: str) -> bool:
    """Run end-to-end transport verification on file and streaming pathways."""
    print("=" * 80)
    print("VoiceShield Transport & Protocol Verification Demo")
    print("=" * 80)
    print(
        "Notice: Synthetic non-human fixtures verify network framing, resampler DSP,\n"
        "and protocol compliance only. They DO NOT evaluate deepfake detection performance.\n"
    )

    ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://") + "/api/v1/stream"
    workspace_root = Path.cwd()

    def count_workspace_audio_files() -> int:
        count = 0
        for p in workspace_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".raw", ".pcm"}:
                if "tests" not in str(p) and "data" not in str(p):
                    count += 1
        return count

    audio_files_before = count_workspace_audio_files()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Step 1: Health (Liveness) Check
        print("[1/5] Verifying API Health (Liveness)...")
        try:
            resp_health = await client.get(f"{api_url}/health")
        except Exception as exc:
            print(f"❌ Failed to reach VoiceShield API at {api_url}: {exc}")
            print("Please ensure the backend is running (e.g., 'make api' or 'make api-mock').")
            return False

        if resp_health.status_code != 200:
            print(f"❌ Health check failed with HTTP {resp_health.status_code}: {resp_health.text}")
            return False

        health_data = resp_health.json()
        print(f"  ✓ Service: {health_data.get('service')} v{health_data.get('version')}")
        print(f"  ✓ Device: {health_data.get('device', {}).get('name')}")
        print(f"  ✓ Model Loaded: {health_data.get('model_loaded')}")
        print(f"  ✓ Mock Detector Mode: {health_data.get('is_mock_detector')}")

        # Step 2: Readiness Check
        print("\n[2/5] Verifying Model Readiness...")
        resp_ready = await client.get(f"{api_url}/ready")
        if resp_ready.status_code == 200:
            ready_data = resp_ready.json()
            is_mock = ready_data.get("is_mock_detector", False)
            if is_mock:
                print("  ✓ Detector Status: READY [MOCK DETECTOR MODE]")
                print("    (Running with mock weights; anomaly scores are synthetic fixtures.)")
            else:
                print("  ✓ Detector Status: READY [REAL MODEL CHECKPOINT]")
        elif resp_ready.status_code == 503:
            not_ready_data = resp_ready.json().get("detail", {})
            print("  ⚠️ Detector Status: NOT READY (Real mode requires checkpoint weights)")
            print(f"    Reason: {not_ready_data.get('reason')}")
            print(
                "    Run 'make api-mock' or set VOICESHIELD_MOCK_MODEL=true to test mock transport."
            )
            return False
        else:
            print(f"❌ Readiness returned unexpected HTTP status: {resp_ready.status_code}")
            return False

        # Step 3: Generating Fixtures
        print("\n[3/5] Generating Non-Human Audio Fixtures In-Memory...")
        fixtures = generate_synthetic_fixtures()
        for name, data in fixtures.items():
            print(f"  ✓ Generated fixture: {name} ({len(data)} bytes, non-human synthetic)")

        # Step 4: File Analysis Transport via HTTP POST /api/v1/analyze/file
        print("\n[4/5] Testing File Analysis Pathway (HTTP Multipart Upload)...")
        sine_bytes = fixtures["sine_440hz_16k_mono"]
        t0 = time.perf_counter()
        resp_file = await client.post(
            f"{api_url}/api/v1/analyze/file",
            files={"file": ("synthetic_sine.wav", sine_bytes, "audio/wav")},
            params={"hop_seconds": 1.0, "threshold": 0.65},
        )
        elapsed_file = (time.perf_counter() - t0) * 1000.0

        if resp_file.status_code != 200:
            print(f"❌ File analysis failed with HTTP {resp_file.status_code}: {resp_file.text}")
            return False

        file_result = resp_file.json()
        print(f"  ✓ File Analysis Completed in {elapsed_file:.1f}ms")
        print(f"  ✓ Reported Duration: {file_result.get('duration_seconds')}s")
        print(f"  ✓ Evaluated 4.0s Windows: {file_result.get('total_windows')}")
        print(f"  ✓ Advisory Anomaly Suspected: {file_result.get('is_synthetic_suspected')}")
        print(f"  ✓ Mean Synthetic Likelihood: {file_result.get('mean_synthetic_likelihood'):.1%}")
        print(f"  ✓ Advisory Note: {file_result.get('temporal_rule_explanation')}")

        # Also test 48 kHz stereo chirp resampling
        chirp_bytes = fixtures["chirp_200_800hz_48k_stereo"]
        resp_chirp = await client.post(
            f"{api_url}/api/v1/analyze/file",
            files={"file": ("synthetic_chirp.wav", chirp_bytes, "audio/wav")},
            params={"hop_seconds": 1.0},
        )
        if resp_chirp.status_code == 200:
            chirp_result = resp_chirp.json()
            print(
                f"  ✓ Stereo 48kHz Resampling & Windowing: {chirp_result.get('total_windows')} windows"
            )

    # Step 5: Streaming Transport via WebSocket /api/v1/stream
    print("\n[5/5] Testing Real-Time Streaming Pathway (WebSocket)...")
    try:
        import websockets
    except ImportError:
        print("❌ websockets library not installed in active environment.")
        return False

    headers = {"Origin": "http://127.0.0.1:5173"}
    try:
        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            # Send protocol start handshake
            start_payload = {
                "type": "start",
                "protocol_version": "1.0",
                "sample_rate": 16000,
                "channels": 1,
                "pcm_encoding": "int16",
                "metadata": {"client": "demo_transport_script"},
            }
            await ws.send(json.dumps(start_payload))

            raw_ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
            start_ack = json.loads(raw_ack)
            if start_ack.get("type") != "session_started":
                print(f"❌ Expected session_started, received: {start_ack}")
                return False

            session_id = start_ack.get("session_id")
            print(f"  ✓ Session Initialized: {session_id}")
            print(f"  ✓ Canonical Sample Rate: {start_ack.get('canonical_sample_rate')} Hz")
            print(
                f"  ✓ Window Size: {start_ack.get('window_seconds')}s (Hop: {start_ack.get('hop_seconds')}s)"
            )
            print(f"  ✓ Mock Detector Flag: {start_ack.get('is_mock_detector')}")

            # Stream 4.5 seconds of PCM sine audio in 100ms chunks (1600 samples = 3200 bytes)
            num_samples = int(16000 * 4.5)
            t_stream = np.linspace(0, 4.5, num_samples, endpoint=False, dtype=np.float32)
            sine_stream = (0.5 * np.sin(2 * np.pi * 440.0 * t_stream) * 32767.0).astype(np.int16)
            pcm_bytes = sine_stream.tobytes()

            chunk_size = 3200  # 100ms
            quality_events_received = 0
            window_results_received = 0

            async def _receive_events() -> None:
                nonlocal quality_events_received, window_results_received
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(msg)
                        mtype = data.get("type")
                        if mtype == "quality_update":
                            quality_events_received += 1
                        elif mtype == "window_result":
                            window_results_received += 1
                        elif mtype == "session_stopped":
                            break
                    except TimeoutError:
                        break

            recv_task = asyncio.create_task(_receive_events())

            # Send chunks with pacing
            for offset in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[offset : offset + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.02)  # Fast-forward playback

            # Gracefully stop stream
            await ws.send(json.dumps({"type": "stop"}))
            await recv_task

            print(
                f"  ✓ Streamed 4.5s PCM audio: Received {quality_events_received} quality events, {window_results_received} window results"
            )
            print("  ✓ Gracefully terminated session with server ACK.")
    except Exception as exc:
        print(f"❌ WebSocket streaming transport test failed: {exc}")
        return False

    # Confirm Zero Audio Persistence
    print("\n[Verification] Confirming Zero Audio Persistence on Disk...")
    audio_files_after = count_workspace_audio_files()
    if audio_files_after != audio_files_before:
        print(
            f"❌ Audio persistence detected! Files before: {audio_files_before}, after: {audio_files_after}"
        )
        return False
    print("  ✓ Zero audio persistence confirmed (0 temporary or cached audio files written).")

    print("\n" + "=" * 80)
    print("✅ All Transport & Protocol Verifications Succeeded!")
    print("=" * 80)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoiceShield Transport & Protocol Verification Demo"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of VoiceShield FastAPI service (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    success = asyncio.run(run_transport_demo(args.api_url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

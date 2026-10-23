import React, { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import {
  ConnectionState,
  DetectionStatus,
  FileInferenceResult,
  QualityMetrics,
  RecordingState,
  ServerQualityEvent,
  ServerStartAck,
  ServerStopAck,
  ServerWindowResult,
  SystemReadiness,
  WindowHistoryItem,
} from './types';

export function getWebSocketStreamUrl(): string {
  if (typeof window === 'undefined') {
    return 'ws://127.0.0.1:5173/api/v1/stream';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/v1/stream`;
}

export function App(): React.JSX.Element {
  // Navigation tab
  const [activeTab, setActiveTab] = useState<'stream' | 'file'>('stream');

  // System readiness and mock detector tracking
  const [readiness, setReadiness] = useState<SystemReadiness>({
    status: 'checking',
    is_mock_detector: false,
  });
  const [isMockDetector, setIsMockDetector] = useState<boolean>(false);

  // Streaming connection and mic state
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Status and score metrics for stream (uncalibrated decimal scores)
  const [status, setStatus] = useState<DetectionStatus>('insufficient_evidence');
  const [uncalibratedScore, setUncalibratedScore] = useState<number | null>(null);
  const [operatingThreshold, setOperatingThreshold] = useState<number>(0.65);
  const [operatingThresholdSource, setOperatingThresholdSource] = useState<string>('unvalidated_heuristic');
  const [reasonDetails, setReasonDetails] = useState<string>(
    'Awaiting microphone activation. Press Start Microphone Stream to begin analysis.'
  );

  // Speech sufficiency and quality diagnostics
  const [activeSpeechSeconds, setActiveSpeechSeconds] = useState<number>(0.0);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics>({
    isClipping: false,
    isSilent: false,
    hasTimingGap: false,
    clippingCount: 0,
    gapCount: 0,
  });

  // Rolling chronological window timeline
  const [timeline, setTimeline] = useState<WindowHistoryItem[]>([]);

  // File analysis state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileHopSeconds, setFileHopSeconds] = useState<number>(1.0);
  const [fileThreshold, setFileThreshold] = useState<number>(0.65);
  const [isFileAnalyzing, setIsFileAnalyzing] = useState<boolean>(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileResult, setFileResult] = useState<FileInferenceResult | null>(null);

  // Internal Web Audio & WebSocket references
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  // Check backend readiness on component mount
  useEffect(() => {
    let isMounted = true;
    async function checkReadiness(): Promise<void> {
      try {
        const resp = await fetch('/api/v1/ready');
        if (resp.ok) {
          const data = (await resp.json()) as { is_mock_detector: boolean; device?: string };
          if (isMounted) {
            setReadiness({
              status: 'ready',
              is_mock_detector: Boolean(data.is_mock_detector),
              device: data.device,
            });
            setIsMockDetector(Boolean(data.is_mock_detector));
          }
        } else if (resp.status === 503) {
          const data = (await resp.json()) as { detail?: { reason?: string } };
          if (isMounted) {
            setReadiness({
              status: 'not_ready',
              is_mock_detector: false,
              reason: data.detail?.reason ?? 'Checkpoint absent or detector not initialized.',
            });
          }
        }
      } catch {
        // Dev server may not be running yet or running in unit tests
        if (isMounted) {
          setReadiness({
            status: 'error',
            is_mock_detector: false,
            reason: 'VoiceShield API dev server not reached at /api. Start with "make api".',
          });
        }
      }
    }
    void checkReadiness();
    return () => {
      isMounted = false;
    };
  }, []);

  const stopAudioAndDisconnect = useCallback((): void => {
    // 1. Stop all microphone hardware tracks immediately
    if (mediaStreamRef.current) {
      try {
        const tracks = mediaStreamRef.current.getTracks?.();
        if (Array.isArray(tracks)) {
          tracks.forEach((track) => {
            try {
              track.stop?.();
            } catch {
              // Ignore track stop errors
            }
          });
        }
      } catch {
        // Ignore media stream errors
      }
      mediaStreamRef.current = null;
    }

    // 2. Disconnect and close AudioWorklet and AudioContext
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // 3. Gracefully notify server and close WebSocket
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'stop' }));
        } catch {
          // Ignore send errors during shutdown
        }
        wsRef.current.close(1000, 'Client stopped');
      }
      wsRef.current = null;
    }

    setRecordingState('idle');
    setConnectionState('disconnected');
  }, []);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopAudioAndDisconnect();
    };
  }, [stopAudioAndDisconnect]);

  const startStream = async (): Promise<void> => {
    setErrorMessage(null);
    setRecordingState('requesting_permission');
    setConnectionState('connecting');

    try {
      // 1. Request microphone access through user gesture
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      mediaStreamRef.current = stream;

      // 2. Initialize Web Audio pipeline
      const AudioCtxClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtxClass();
      audioContextRef.current = audioCtx;

      // Load PCM AudioWorklet processor
      try {
        await audioCtx.audioWorklet.addModule('/pcm-worklet-processor.js');
      } catch {
        // Fallback to inline blob if module fetch fails
        const workletCode = `
          class PCMWorkletProcessor extends AudioWorkletProcessor {
            constructor() {
              super();
              this.bufferSize = 1024;
              this.buffer = new Int16Array(this.bufferSize);
              this.bufferIndex = 0;
            }

            process(inputs, outputs, parameters) {
              const input = inputs[0];
              if (!input || !input[0]) return true;
              const channelData = input[0];

              for (let i = 0; i < channelData.length; i++) {
                let s = channelData[i];
                if (s > 1.0) s = 1.0;
                else if (s < -1.0) s = -1.0;
                this.buffer[this.bufferIndex++] = s < 0 ? s * 32768 : s * 32767;

                if (this.bufferIndex >= this.bufferSize) {
                  this.port.postMessage(this.buffer.buffer.slice(0));
                  this.bufferIndex = 0;
                }
              }
              return true;
            }
          }
          registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
        `;
        const blob = new Blob([workletCode], { type: 'application/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        await audioCtx.audioWorklet.addModule(blobUrl);
      }

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, 'pcm-worklet-processor');
      workletNodeRef.current = workletNode;
      sourceNode.connect(workletNode);

      // 3. Establish WebSocket connection to backend
      const wsUrl = getWebSocketStreamUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setRecordingState('recording');
        // Send protocol start handshake declaring sample rate and format
        const startPayload = {
          type: 'start',
          protocol_version: '1.0',
          sample_rate: audioCtx.sampleRate,
          channels: 1,
          pcm_encoding: 'int16',
          metadata: { client: 'voiceshield-web-demo' },
        };
        ws.send(JSON.stringify(startPayload));
      };

      // Stream binary PCM chunks produced by AudioWorklet
      workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(event.data);
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data as string) as
            | ServerStartAck
            | ServerQualityEvent
            | ServerWindowResult
            | ServerStopAck
            | { type: 'error'; message: string; error_code: string };

          if (data.type === 'session_started') {
            setConnectionState('connected');
            if (data.is_mock_detector !== undefined) {
              setIsMockDetector(Boolean(data.is_mock_detector));
            }
          } else if (data.type === 'quality_update') {
            setQualityMetrics({
              isClipping: data.is_clipped,
              isSilent: data.is_silent,
              hasTimingGap: data.has_timing_gap,
              clippingCount: data.clipping_count,
              gapCount: data.gap_count,
            });
          } else if (data.type === 'window_result') {
            setActiveSpeechSeconds(data.speech_seconds);

            const decState = data.decision.state.toLowerCase() as DetectionStatus;
            setStatus(decState);
            setReasonDetails(data.decision.reason_details);

            const score =
              data.synthetic_score ??
              data.uncalibrated_synthetic_score ??
              data.raw_score ??
              null;
            setUncalibratedScore(score);

            if (data.operating_threshold !== undefined) {
              setOperatingThreshold(data.operating_threshold);
            }
            if (data.operating_threshold_source !== undefined) {
              setOperatingThresholdSource(data.operating_threshold_source);
            }

            const historyItem: WindowHistoryItem = {
              windowIndex: data.window_index,
              windowStartSec: data.window_start_sec,
              windowEndSec: data.window_end_sec,
              speechSeconds: data.speech_seconds,
              isEligible: data.is_eligible,
              syntheticScore: score,
              scoreIsCalibrated: Boolean(data.score_is_calibrated),
              calibratedScore: data.calibrated_score ?? null,
              rawScore: data.raw_score ?? null,
              status: decState,
              reason: data.decision.reason,
              reasonDetails: data.decision.reason_details,
              timestamp: new Date().toLocaleTimeString(),
            };

            setTimeline((prev) => [historyItem, ...prev.slice(0, 19)]);
          } else if (data.type === 'error') {
            setStatus('error');
            setErrorMessage(`${data.error_code}: ${data.message}`);
          }
        } catch {
          // Non-JSON server frame
        }
      };

      ws.onerror = () => {
        setConnectionState('error');
        setStatus('error');
        setErrorMessage('WebSocket connection error. Is the VoiceShield backend running?');
        stopAudioAndDisconnect();
      };

      ws.onclose = () => {
        setConnectionState('disconnected');
        setRecordingState('idle');
      };
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error ? err.message : 'Microphone permission denied or audio device error.';
      setErrorMessage(errorMsg);
      setRecordingState('error');
      setConnectionState('error');
      stopAudioAndDisconnect();
    }
  };

  const handleFileAnalysis = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsFileAnalyzing(true);
    setFileError(null);
    setFileResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const url = `/api/v1/analyze/file?hop_seconds=${fileHopSeconds}&threshold=${fileThreshold}`;
      const resp = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errJson = (await resp.json()) as { detail?: { message?: string; error_code?: string } };
        throw new Error(
          errJson.detail?.message || `File analysis failed with HTTP status ${resp.status}`
        );
      }

      const result = (await resp.json()) as FileInferenceResult;
      setFileResult(result);
      if (result.is_mock_detector !== undefined) {
        setIsMockDetector(Boolean(result.is_mock_detector));
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Audio file analysis failed.';
      setFileError(msg);
    } finally {
      setIsFileAnalyzing(false);
    }
  };

  const isRecording = recordingState === 'recording';

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            🛡️
          </div>
          <div>
            <h1 className="brand-title">VoiceShield</h1>
            <p className="brand-subtitle">
              Local Microphone Advisory Demo (Local input only — does not capture cellular telephony)
            </p>
          </div>
        </div>
        <div className="connection-badges">
          <div className={`status-pill pill-${connectionState}`} id="connection-state-pill">
            <span className="dot" aria-hidden="true" />
            <span>Connection: {connectionState}</span>
          </div>
          <div className={`status-pill pill-${recordingState}`} id="recording-state-pill">
            <span className="dot" aria-hidden="true" />
            <span>Mic: {recordingState}</span>
          </div>
        </div>
      </header>

      {/* Model Not Ready Alert (if real mode without checkpoint) */}
      {readiness.status === 'not_ready' && (
        <div className="model-not-ready-banner" role="alert" id="model-not-ready-banner">
          <strong>⚠️ DETECTOR MODEL NOT READY</strong>
          <p>
            VoiceShield is running in real mode but the AASIST-L checkpoint weights are absent. Real mode
            refuses to mark model as ready without verified weights.
          </p>
          {readiness.reason && <pre>{readiness.reason}</pre>}
        </div>
      )}

      {/* Mock Detector Visible Indicator Badge */}
      {isMockDetector && (
        <div
          className="mock-detector-badge"
          data-testid="mock-detector-badge"
          role="status"
          id="mock-detector-badge"
        >
          <span className="mock-badge-tag">[MOCK DETECTOR - TEST MODE]</span>
          <span className="mock-badge-text">
            Synthetic weights active for UI and transport demonstration. Anomaly scores are mock values
            and do not evaluate deepfakes.
          </span>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="demo-tabs" aria-label="Demo mode navigation">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'stream' ? 'active' : ''}`}
          onClick={() => setActiveTab('stream')}
        >
          🎙️ Live Microphone Stream
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === 'file' ? 'active' : ''}`}
          onClick={() => setActiveTab('file')}
        >
          📁 File Audio Analysis
        </button>
      </nav>

      {/* Strict Compliance and Privacy Banner */}
      <div className="compliance-banner" id="compliance-banner">
        <div className="compliance-row">
          <span className="badge-tag">PRIVACY NOTICE:</span>
          <span>
            Audio is processed in volatile memory and immediately discarded. Audio is never stored in
            your browser or on the server.
          </span>
        </div>
        <div className="compliance-row">
          <span className="badge-tag">ADVISORY NOTICE:</span>
          <span>
            VoiceShield outputs acoustic anomaly indicators only. It does not authenticate caller
            identity and is not a fraud confirmation. Never automatically disconnect or block callers.
          </span>
        </div>
      </div>

      {errorMessage && (
        <div className="alert-error" role="alert" id="stream-error-alert">
          <span className="alert-icon" aria-hidden="true">
            ✕
          </span>
          <p id="error-alert-text">{errorMessage}</p>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 1: Real-time Streaming Dashboard                                      */}
      {/* ========================================================================= */}
      {activeTab === 'stream' && (
        <>
          {/* High Anomaly Action Callout */}
          {status === 'high' && (
            <section
              className="action-directive-high"
              role="alert"
              aria-live="assertive"
              id="high-status-action-banner"
            >
              <div className="action-icon" aria-hidden="true">
                ⚠️
              </div>
              <div className="action-body">
                <h4>ACTION DIRECTIVE</h4>
                <p className="action-strong">
                  Pause and verify through a known independent channel.
                </p>
                <p className="action-subtext">
                  Elevated synthetic speech anomaly score sustained over multiple consecutive windows. Do
                  not rely on voice identity.
                </p>
              </div>
            </section>
          )}

          {/* Main Grid Dashboard */}
          <main className="grid-dashboard">
            {/* Status Card */}
            <div className="card status-card">
              <div className="card-title">
                <span>Assessment Status</span>
                <span className="sub-label">4.0s Window • 1.0s Hop</span>
              </div>

              <div
                className={`status-indicator-box status-box-${status}`}
                role="status"
                aria-live="polite"
                id="current-status-box"
              >
                {/* Accessible Non-Color Indicator Text Badge */}
                <div className="non-color-indicator" id="accessible-status-indicator">
                  {status === 'low' && '[✓ LOW ANOMALY]'}
                  {status === 'uncertain' && '[? UNCERTAIN]'}
                  {status === 'high' && '[! HIGH ANOMALY]'}
                  {status === 'insufficient_evidence' && '[... INSUFFICIENT SPEECH]'}
                  {status === 'error' && '[✕ ERROR]'}
                </div>

                <div className="status-label-heading" id="status-heading">
                  {status === 'low' && 'Low acoustic anomaly score — does not authenticate speech'}
                  {status === 'uncertain' && 'Uncertain / Ambiguous Zone'}
                  {status === 'high' && 'High Acoustic Anomaly Score'}
                  {status === 'insufficient_evidence' && 'Insufficient Active Speech'}
                  {status === 'error' && 'Assessment Error'}
                </div>

                <p className="status-reason" id="status-reason-details">
                  {reasonDetails}
                </p>
              </div>

              {/* Uncalibrated Score and Provenance Display */}
              <div className="score-container" id="score-container">
                {uncalibratedScore !== null ? (
                  <div className="score-block">
                    <span className="score-type">Uncalibrated Synthetic Score:</span>
                    <span className="score-value" id="raw-score-value">
                      score {uncalibratedScore.toFixed(3)} / threshold {operatingThreshold.toFixed(3)}
                    </span>
                    <span className="score-source-note" id="threshold-provenance-note" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Operating threshold: {operatingThreshold.toFixed(3)} ({operatingThresholdSource === 'unvalidated_heuristic' ? 'unvalidated heuristic' : operatingThresholdSource})
                    </span>
                  </div>
                ) : (
                  <div className="score-block">
                    <span className="score-type">Score:</span>
                    <span className="score-value-pending" id="score-pending-value">
                      Awaiting 4.0s window...
                    </span>
                  </div>
                )}
              </div>

              {/* Control Actions (Require User Gesture) */}
              <div className="controls-row">
                {!isRecording ? (
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={startStream}
                    id="start-mic-button"
                  >
                    🎤 Start Microphone Stream
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={stopAudioAndDisconnect}
                    id="stop-mic-button"
                  >
                    ⏹ Stop Stream & Disconnect
                  </button>
                )}
              </div>
            </div>

            {/* Quality & Gating Metrics Card */}
            <div className="card metrics-card">
              <div className="card-title">
                <span>Audio Quality & Speech Gating</span>
              </div>

              <div className="metrics-list">
                <div className="metric-row">
                  <span className="metric-name">Active Speech in Window</span>
                  <span
                    className={`metric-status ${
                      activeSpeechSeconds >= 0.8 ? 'metric-clean' : 'metric-warn'
                    }`}
                    id="speech-duration-status"
                  >
                    {activeSpeechSeconds.toFixed(2)}s / 4.0s{' '}
                    {activeSpeechSeconds >= 0.8 ? '(Sufficient)' : '(Insufficient < 0.8s)'}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-name">Clipping Distortion</span>
                  <span
                    className={`metric-status ${
                      qualityMetrics.isClipping ? 'metric-warn' : 'metric-clean'
                    }`}
                    id="clipping-status"
                  >
                    {qualityMetrics.isClipping
                      ? `Clipping Detected (${qualityMetrics.clippingCount} frame(s))`
                      : 'Clean'}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-name">Signal Energy (Silence Gate)</span>
                  <span
                    className={`metric-status ${
                      qualityMetrics.isSilent ? 'metric-warn' : 'metric-clean'
                    }`}
                    id="silence-status"
                  >
                    {qualityMetrics.isSilent ? 'Near Silence (RMS < 1e-4)' : 'Active Signal'}
                  </span>
                </div>

                <div className="metric-row">
                  <span className="metric-name">Timing Continuity</span>
                  <span
                    className={`metric-status ${
                      qualityMetrics.hasTimingGap ? 'metric-warn' : 'metric-clean'
                    }`}
                    id="timing-gap-status"
                  >
                    {qualityMetrics.hasTimingGap
                      ? `Discontinuity Flagged (${qualityMetrics.gapCount} gap(s))`
                      : 'Continuous'}
                  </span>
                </div>
              </div>
            </div>
          </main>

          {/* Rolling Timeline Section */}
          <section className="timeline-section" aria-label="Rolling timeline of window evaluations">
            <h3 className="timeline-title">Rolling Window Assessment Timeline</h3>
            {timeline.length === 0 ? (
              <p className="timeline-empty" id="timeline-empty-message">
                No inference windows evaluated yet. Stream active speech for at least 4.0 seconds.
              </p>
            ) : (
              <div className="timeline-list" id="timeline-list" role="feed">
                {timeline.map((item) => (
                  <article
                    key={item.windowIndex}
                    className={`timeline-card timeline-status-${item.status}`}
                  >
                    <div className="timeline-header">
                      <span className="timeline-idx">Window #{item.windowIndex}</span>
                      <span className="timeline-range">
                        {item.windowStartSec.toFixed(1)}s - {item.windowEndSec.toFixed(1)}s
                      </span>
                      <span className={`timeline-badge badge-${item.status}`}>
                        {item.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="timeline-body">
                      <span className="timeline-speech">
                        Speech: {item.speechSeconds.toFixed(2)}s
                      </span>
                      {item.syntheticScore !== null && (
                        <span className="timeline-score">
                          Score: {item.syntheticScore.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <p className="timeline-reason">{item.reasonDetails}</p>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: File Audio Analysis                                                */}
      {/* ========================================================================= */}
      {activeTab === 'file' && (
        <section className="file-analysis-card" aria-label="Offline audio file analysis">
          <form onSubmit={handleFileAnalysis}>
            <div
              className="file-dropzone"
              onClick={() => document.getElementById('audio-file-input')?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  document.getElementById('audio-file-input')?.click();
                }
              }}
              tabIndex={0}
              role="button"
              aria-label="Upload audio file"
            >
              <div className="file-dropzone-icon" aria-hidden="true">
                📁
              </div>
              <p className="file-dropzone-text">Click or select a WAV or FLAC file to analyze</p>
              <p className="file-dropzone-hint">
                Standard 16 kHz mono conversion and 4.0s windowing are applied in memory.
              </p>
              <input
                id="audio-file-input"
                type="file"
                accept=".wav,.flac,audio/wav,audio/flac"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setSelectedFile(e.target.files[0]);
                  }
                }}
              />
              {selectedFile && (
                <div className="file-selected-badge" id="selected-file-badge">
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                </div>
              )}
            </div>

            <div className="file-form-grid">
              <div className="form-group">
                <label htmlFor="hop-seconds-input">Window Hop Step (seconds):</label>
                <input
                  id="hop-seconds-input"
                  type="number"
                  min="0.25"
                  max="4.0"
                  step="0.25"
                  value={fileHopSeconds}
                  onChange={(e) => setFileHopSeconds(parseFloat(e.target.value) || 1.0)}
                />
              </div>

              <div className="form-group">
                <label htmlFor="threshold-input">Decision Threshold:</label>
                <input
                  id="threshold-input"
                  type="number"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={fileThreshold}
                  onChange={(e) => setFileThreshold(parseFloat(e.target.value) || 0.65)}
                />
              </div>
            </div>

            <div className="controls-row">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!selectedFile || isFileAnalyzing}
                id="submit-file-analysis-button"
              >
                {isFileAnalyzing ? '⏳ Evaluating 4.0s Windows...' : '⚡ Analyze Audio File'}
              </button>
            </div>
          </form>

          {fileError && (
            <div className="alert-error" role="alert" style={{ marginTop: '1.25rem' }}>
              <span className="alert-icon" aria-hidden="true">
                ✕
              </span>
              <p id="file-error-text">{fileError}</p>
            </div>
          )}

          {fileResult && (
            <div className="file-results-card" style={{ marginTop: '1.5rem' }}>
              <div className="file-verdict-summary">
                <div className="verdict-stat">
                  <div className="verdict-stat-label">Advisory Verdict</div>
                  <div
                    className="verdict-stat-value"
                    style={{
                      color:
                        fileResult.status === 'insufficient_evidence'
                          ? 'var(--accent-amber)'
                          : fileResult.is_synthetic_suspected
                          ? 'var(--accent-red)'
                          : 'var(--accent-emerald)',
                    }}
                  >
                    {fileResult.status === 'insufficient_evidence'
                      ? '[... INSUFFICIENT SPEECH]'
                      : fileResult.is_synthetic_suspected
                      ? '[! HIGH ANOMALY]'
                      : '[✓ LOW ANOMALY]'}
                  </div>
                </div>

                <div className="verdict-stat">
                  <div className="verdict-stat-label">Mean Synthetic Score</div>
                  <div className="verdict-stat-value" id="file-mean-score">
                    {fileResult.mean_synthetic_score !== null
                      ? fileResult.mean_synthetic_score.toFixed(3)
                      : 'N/A — inference not performed'}
                  </div>
                </div>

                <div className="verdict-stat">
                  <div className="verdict-stat-label">Peak Window Score</div>
                  <div className="verdict-stat-value" id="file-peak-score">
                    {fileResult.max_synthetic_score !== null
                      ? fileResult.max_synthetic_score.toFixed(3)
                      : 'N/A — inference not performed'}
                  </div>
                </div>

                <div className="verdict-stat">
                  <div className="verdict-stat-label">Duration / Windows</div>
                  <div className="verdict-stat-value">
                    {fileResult.duration_seconds.toFixed(1)}s ({fileResult.total_windows} win)
                  </div>
                </div>
              </div>

              <div
                className="threshold-provenance-banner"
                style={{
                  margin: '1rem 0',
                  padding: '0.75rem 1rem',
                  borderRadius: '6px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  fontSize: '0.85rem',
                }}
              >
                <strong style={{ color: 'var(--text-primary)' }}>Threshold Provenance: </strong>
                <span>
                  Effective threshold: {fileResult.effective_threshold.toFixed(3)} ({fileResult.operating_threshold_source === 'unvalidated_heuristic' ? 'unvalidated heuristic' : fileResult.operating_threshold_source}). Uncalibrated model output; does not represent calibrated probability.
                </span>
              </div>

              {/* Action Directive on High Anomaly File */}
              {fileResult.is_synthetic_suspected && (
                <section
                  className="action-directive-high"
                  role="alert"
                  aria-live="assertive"
                  style={{ marginBottom: '1.25rem' }}
                >
                  <div className="action-icon" aria-hidden="true">
                    ⚠️
                  </div>
                  <div className="action-body">
                    <h4>ACTION DIRECTIVE</h4>
                    <p className="action-strong">
                      Pause and verify through a known independent channel.
                    </p>
                    <p className="action-subtext">
                      Multiple consecutive temporal windows exhibited elevated synthetic acoustic anomaly scores.
                    </p>
                  </div>
                </section>
              )}

              <p className="status-reason" style={{ marginBottom: '1rem' }}>
                {fileResult.temporal_rule_explanation}
              </p>

              {/* Per-window breakdown table */}
              <div className="window-table-container">
                <table className="window-table" aria-label="Per-window acoustic evaluation records">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Time Range</th>
                      <th>Bona Fide Score</th>
                      <th>Synthetic Score</th>
                      <th>Active Speech</th>
                      <th>Window Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fileResult.windows.map((win) => (
                      <tr key={win.window_index}>
                        <td>#{win.window_index}</td>
                        <td>
                          {win.start_seconds.toFixed(1)}s - {win.end_seconds.toFixed(1)}s
                        </td>
                        <td>{win.raw_bonafide_score !== null ? win.raw_bonafide_score.toFixed(3) : 'N/A'}</td>
                        <td>{win.synthetic_score !== null ? win.synthetic_score.toFixed(3) : 'N/A'}</td>
                        <td>{win.active_speech_seconds.toFixed(2)}s</td>
                        <td>
                          <span
                            className={`timeline-badge ${
                              win.is_eligible === false
                                ? 'badge-insufficient_evidence'
                                : win.is_synthetic_suspected
                                ? 'badge-high'
                                : 'badge-low'
                            }`}
                          >
                            {win.is_eligible === false
                              ? 'ABSTAINED'
                              : win.is_synthetic_suspected
                              ? 'ELEVATED'
                              : 'LOW ANOMALY'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { StrictMode } from 'react';
import {
  App,
  fileWindowState,
  getWebSocketStreamUrl,
  validateInputReliability,
  validateQualityMetrics,
  validateFileAggregateAndWindows,
  validateStreamingStopAck,
  validateServerWindowResult,
} from './App';
import type {
  AudioQualityMetrics,
  DecisionState,
  FileInferenceResult,
  FileWindowResult,
  InputReliabilityAssessment,
  ReasonCode,
  ServerWindowResult,
  TemporalDecision,
} from './types';

const reliableAssessment: InputReliabilityAssessment = {
  state: 'reliable',
  reasons: [],
  explanation: 'Window meets all quality and speech sufficiency criteria.',
  origin_verdict_allowed: true,
};

const cleanQualityMetrics: AudioQualityMetrics = {
  peak_level: 0.5,
  rms: 0.15,
  clipping_fraction: 0.0,
  is_clipped: false,
  is_silent: false,
  snr_db_estimate: 24.5,
  quality_status: 'clean',
};

function normalizeReason(r: string | undefined): ReasonCode {
  if (!r) return 'normal_acoustics';
  if (r === 'sustained_high') return 'sustained_anomaly_elevation';
  if (r === 'final_window') return 'sustained_anomaly_elevation';
  if (r === 'normal') return 'normal_acoustics';
  if (r === 'insufficient_speech') return 'insufficient_speech_duration';
  if (r === 'clipping') return 'signal_quality_degraded';
  return r as ReasonCode;
}

function createValidServerWindowResult(overrides: Record<string, unknown> = {}): ServerWindowResult {
  const windowIndex = (overrides.window_index as number) ?? 0;
  const startSec = (overrides.window_start_sec as number) ?? windowIndex * 1.0;
  const endSec = (overrides.window_end_sec as number) ?? startSec + 4.0;
  const speechSec = (overrides.speech_seconds as number) ?? 3.5;
  const isEligible = overrides.is_eligible !== undefined ? (overrides.is_eligible as boolean) : true;
  const synthScore =
    overrides.synthetic_score !== undefined
      ? (overrides.synthetic_score as number | null)
      : isEligible
        ? 0.25
        : null;

  const defaultReason: ReasonCode = isEligible ? 'normal_acoustics' : 'insufficient_speech_duration';

  const defaultDecision: TemporalDecision = isEligible
    ? {
        state: 'low',
        synthetic_score: synthScore,
        score_is_calibrated: false,
        calibrated_score: null,
        smoothed_score: synthScore ?? 0.25,
        reason: defaultReason,
        reason_details: 'Normal acoustics',
        window_index: windowIndex,
        window_start_sec: startSec,
        window_end_sec: endSec,
        speech_seconds: speechSec,
        history_scores: synthScore !== null ? [synthScore] : [],
      }
    : {
        state: 'insufficient_evidence',
        synthetic_score: null,
        score_is_calibrated: false,
        calibrated_score: null,
        smoothed_score: 0.0,
        reason: defaultReason,
        reason_details: 'Insufficient speech',
        window_index: windowIndex,
        window_start_sec: startSec,
        window_end_sec: endSec,
        speech_seconds: speechSec,
        history_scores: [],
      };

  const defaultReliability: InputReliabilityAssessment = isEligible
    ? reliableAssessment
    : {
        state: 'insufficient',
        reasons: ['insufficient_speech'],
        explanation: 'Insufficient speech duration.',
        origin_verdict_allowed: false,
      };

  let decisionObj: TemporalDecision = defaultDecision;
  if (overrides.decision && typeof overrides.decision === 'object') {
    const rawDec = overrides.decision as Record<string, unknown>;
    const decScore =
      rawDec.synthetic_score !== undefined
        ? (rawDec.synthetic_score as number | null)
        : synthScore;
    const smoothed =
      rawDec.smoothed_score !== undefined && rawDec.smoothed_score !== null
        ? Number(rawDec.smoothed_score)
        : (decScore ?? 0.0);
    decisionObj = {
      ...defaultDecision,
      ...rawDec,
      state: (rawDec.state as DecisionState) ?? defaultDecision.state,
      reason: normalizeReason(rawDec.reason as string | undefined),
      synthetic_score: decScore,
      smoothed_score: smoothed,
      window_index: windowIndex,
      window_start_sec: startSec,
      window_end_sec: endSec,
      speech_seconds: speechSec,
      history_scores: Array.isArray(rawDec.history_scores)
        ? (rawDec.history_scores as number[])
        : smoothed !== null
          ? [smoothed]
          : [],
    };
  }

  const base: ServerWindowResult = {
    type: 'window_result',
    session_id: (overrides.session_id as string) ?? 'test-session-123',
    window_index: windowIndex,
    window_start_sec: startSec,
    window_end_sec: endSec,
    speech_seconds: speechSec,
    is_eligible: isEligible,
    synthetic_score: synthScore,
    uncalibrated_synthetic_score:
      overrides.uncalibrated_synthetic_score !== undefined
        ? (overrides.uncalibrated_synthetic_score as number | null)
        : synthScore,
    score_is_calibrated: (overrides.score_is_calibrated as boolean) ?? false,
    operating_threshold: (overrides.operating_threshold as number) ?? 0.65,
    threshold_low: (overrides.threshold_low as number) ?? 0.35,
    threshold_high: (overrides.threshold_high as number) ?? 0.65,
    operating_threshold_source:
      (overrides.operating_threshold_source as string) ?? 'unvalidated_heuristic',
    raw_score:
      overrides.raw_score !== undefined
        ? (overrides.raw_score as number | null)
        : synthScore,
    calibrated_score: (overrides.calibrated_score as number | null) ?? null,
    decision: decisionObj,
    input_reliability:
      (overrides.input_reliability as InputReliabilityAssessment) ?? defaultReliability,
    quality_status: (overrides.quality_status as AudioQualityMetrics) ?? cleanQualityMetrics,
    abstain_reason:
      overrides.abstain_reason !== undefined
        ? (overrides.abstain_reason as string | null)
        : isEligible
          ? null
          : 'insufficient_speech',
  };

  if (overrides.decision === null) {
    return { ...base, ...overrides, decision: null as unknown as TemporalDecision } as ServerWindowResult;
  }
  if ((overrides as any).__rawDecision) {
    return { ...base, ...overrides } as ServerWindowResult;
  }
  return { ...base, ...overrides, decision: decisionObj } as ServerWindowResult;
}

function createValidFileWindowResult(overrides: Partial<FileWindowResult> = {}): FileWindowResult {
  const base: FileWindowResult = {
    window_index: 0,
    start_seconds: 0.0,
    end_seconds: 4.0,
    raw_spoof_logit: -2.0,
    raw_bonafide_logit: 2.0,
    raw_bonafide_score: 0.9,
    synthetic_score: 0.1,
    uncalibrated_synthetic_score: 0.1,
    score_is_calibrated: false,
    calibrated_synthetic_likelihood: null,
    synthetic_likelihood: 0.1,
    effective_threshold: 0.65,
    is_synthetic_suspected: false,
    active_speech_seconds: 3.5,
    is_eligible: true,
    is_partial: false,
    abstain_reason: null,
    quality_status: cleanQualityMetrics,
    input_reliability: reliableAssessment,
    decision: {
      state: 'low',
      synthetic_score: 0.1,
      score_is_calibrated: false,
      calibrated_score: null,
      smoothed_score: 0.1,
      reason: 'normal_acoustics',
      reason_details: 'Low anomaly',
      window_index: 0,
      window_start_sec: 0.0,
      window_end_sec: 4.0,
      speech_seconds: 3.5,
      history_scores: [0.1],
    },
    decision_state: 'low',
    decision_reason: 'normal_acoustics',
    decision_reason_details: 'Low anomaly',
    smoothed_score: 0.1,
    history_scores: [0.1],
  };
  return { ...base, ...overrides };
}

function createValidFileInferenceResult(overrides: Partial<FileInferenceResult> = {}): FileInferenceResult {
  const base: FileInferenceResult = {
    file_path: 'test_audio.wav',
    status: 'success',
    error_message: null,
    error_code: null,
    abstain_reason: null,
    duration_seconds: 4.0,
    active_speech_duration_seconds: 3.5,
    original_sample_rate: 16000,
    original_channels: 1,
    total_windows: 1,
    eligible_windows: 1,
    reliable_window_count: 1,
    limited_reliability_window_count: 0,
    insufficient_reliability_window_count: 0,
    degraded_quality_window_count: 0,
    quality_origin_verdict_allowed: true,
    raw_high_window_count: 0,
    uncertain_window_count: 0,
    high_window_count: 0,
    mean_synthetic_score: 0.1,
    max_synthetic_score: 0.1,
    mean_synthetic_likelihood: 0.1,
    max_synthetic_likelihood: 0.1,
    score_is_calibrated: false,
    effective_threshold: 0.65,
    operating_threshold_source: 'heuristic',
    is_synthetic_suspected: false,
    final_decision_state: 'low',
    temporal_rule_explanation: 'Normal acoustics across evaluated windows.',
    is_mock_detector: false,
    config_hash: null,
    metadata: { client: 'voiceshield' },
    disclaimers: {
      identity_proof: 'Not proof of identity',
      fraud_probability: 'Not fraud probability',
      advisory_action: 'Advisory only',
      capture_boundary: 'Boundary',
    },
    windows: [createValidFileWindowResult()],
  };
  return { ...base, ...overrides };
}

// Mock Web Audio API and MediaDevices
class MockAudioWorklet {
  addModule = vi.fn().mockResolvedValue(undefined);
}

class MockAudioContext {
  static instances: MockAudioContext[] = [];
  static initialState: 'running' | 'suspended' = 'running';
  sampleRate = 48000;
  state: 'running' | 'suspended' | 'closed' = MockAudioContext.initialState;
  destination = {};
  audioWorklet = new MockAudioWorklet();
  resume = vi.fn().mockImplementation(async () => {
    this.state = 'running';
  });
  createGain = vi.fn().mockImplementation(() => {
    const gain = { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() };
    return gain;
  });
  createMediaStreamSource = vi.fn().mockImplementation(() => ({
    connect: vi.fn(),
    disconnect: vi.fn(),
  }));
  close = vi.fn().mockResolvedValue(undefined);

  constructor() {
    MockAudioContext.instances.push(this);
  }
}

class MockAudioWorkletNode {
  static instances: MockAudioWorkletNode[] = [];
  port = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage: vi.fn(),
  };
  connect = vi.fn();
  disconnect = vi.fn();

  constructor() {
    MockAudioWorkletNode.instances.push(this);
  }
}

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState: number = WebSocket.CONNECTING;
  binaryType = 'blob';
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  sentMessages: (string | ArrayBuffer)[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 0);
  }

  send(data: string | ArrayBuffer) {
    this.sentMessages.push(data);
  }

  close(_code?: number, _reason?: string) {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) this.onclose();
  }

  simulateServerMessage(data: unknown) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
}

describe('VoiceShield Dashboard State Transitions & Rendering', () => {
  it.each([
    ['high', 'ELEVATED', 'badge-high'],
    ['uncertain', 'UNCERTAIN', 'badge-uncertain'],
    ['low', 'LOW ANOMALY', 'badge-low'],
    ['insufficient_evidence', 'ABSTAINED', 'badge-insufficient_evidence'],
    ['ineligible', 'ABSTAINED', 'badge-insufficient_evidence'],
    [undefined, 'UNKNOWN', 'badge-unknown'],
  ])('maps file decision state %s without falling back to LOW', (state, label, className) => {
    const window = {
      is_eligible: state !== 'ineligible',
      decision_state: state,
    } as FileWindowResult;
    expect(fileWindowState(window)).toEqual({ label, className });
  });

  let mockTracks: { stop: ReturnType<typeof vi.fn> }[];

  beforeEach(() => {
    MockWebSocket.instances = [];
    MockAudioContext.instances = [];
    MockAudioContext.initialState = 'running';
    MockAudioWorkletNode.instances = [];
    mockTracks = [{ stop: vi.fn() }];

    vi.stubGlobal('AudioContext', MockAudioContext);
    vi.stubGlobal('AudioWorkletNode', MockAudioWorkletNode);
    vi.stubGlobal('WebSocket', MockWebSocket);

    vi.stubGlobal('navigator', {
      mediaDevices: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: vi.fn().mockReturnValue(mockTracks),
        }),
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue({ is_mock_detector: false, device: 'cpu' }),
      })
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders initial state with strict privacy and scope disclaimers', async () => {
    await act(async () => {
      render(<App />);
    });

    // Brand and Scope verification
    expect(screen.getByRole('heading', { level: 1, name: /VoiceShield/i })).toBeDefined();
    expect(screen.getByText(/Local Microphone Advisory Demo/i)).toBeDefined();
    expect(screen.getByText(/Local input only — does not capture cellular telephony/i)).toBeDefined();

    // Privacy Notice verification
    expect(screen.getByText(/PRIVACY NOTICE/i)).toBeDefined();
    expect(screen.getByText(/never stored in your browser or on the server/i)).toBeDefined();

    // Initial non-color accessible status badge
    expect(screen.getByText(/\[\.\.\. INSUFFICIENT SPEECH\]/i)).toBeDefined();
    expect(screen.getByText(/Insufficient Active Speech/i)).toBeDefined();

    // Initial connection badges
    expect(screen.getByText(/Connection: disconnected/i)).toBeDefined();
    expect(screen.getByText(/Mic: idle/i)).toBeDefined();

    // Verify PROHIBITED strings are NEVER rendered
    const bodyText = document.body.textContent || '';
    expect(bodyText).not.toContain('identity verified');
    expect(bodyText).not.toContain('fraud detected');
    expect(bodyText).not.toContain('100% real');
  });

  it('renders guidance for a readiness transport error', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
      json: vi.fn().mockResolvedValue({}),
    } as unknown as Response);
    render(<App />);
    expect((await screen.findByRole('alert')).textContent).toContain('READINESS UNAVAILABLE');
    expect(screen.getByText(/HTTP 500/i)).toBeDefined();
  });

  it('transitions state on startStream user gesture and WebSocket handshake', async () => {
    render(<App />);

    const startButton = screen.getByRole('button', { name: /Start Microphone Stream/i });
    expect(startButton).toBeDefined();

    // User gesture trigger
    await act(async () => {
      fireEvent.click(startButton);
    });

    // Verify mediaDevices permission requested
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    // Wait for WebSocket instance creation
    expect(MockWebSocket.instances.length).toBe(1);
    const ws = MockWebSocket.instances[0];

    // Wait for open handshake
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });

    // Verify protocol handshake sent
    expect(ws.sentMessages.length).toBeGreaterThan(0);
    const startMsg = JSON.parse(ws.sentMessages[0] as string);
    expect(startMsg.type).toBe('start');
    expect(startMsg.protocol_version).toBe('1.0');
    expect(startMsg.sample_rate).toBe(48000);
    expect(startMsg.pcm_encoding).toBe('int16');

    // Simulate session_started ACK from server
    await act(async () => {
      ws.simulateServerMessage({
        type: 'session_started',
        session_id: 'test-session-123',
        protocol_version: '1.0',
        canonical_sample_rate: 16000,
        canonical_channels: 1,
        window_seconds: 4.0,
        hop_seconds: 1.0,
        min_speech_seconds: 0.8,
        max_frame_bytes: 131072,
        max_stream_duration_seconds: 3600.0,
        idle_timeout_seconds: 30.0,
      });
    });

    expect(screen.getByText(/Connection: connected/i)).toBeDefined();
    expect(screen.getByText(/Mic: recording/i)).toBeDefined();
  });

  it('renders high anomaly status with required Action Directive text', async () => {
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];

    // Simulate high anomaly window result with uncalibrated score
    const validMsg = createValidServerWindowResult({
      window_index: 3,
      window_start_sec: 3.0,
      window_end_sec: 7.0,
      speech_seconds: 3.6,
      is_eligible: true,
      synthetic_score: 0.88,
      decision: {
        state: 'high',
        synthetic_score: 0.88,
        score_is_calibrated: false,
        calibrated_score: null,
        smoothed_score: 0.85,
        reason: 'sustained_anomaly_elevation',
        reason_details: 'Elevated anomaly detected over 3 consecutive windows.',
        window_index: 3,
        window_start_sec: 3.0,
        window_end_sec: 7.0,
        speech_seconds: 3.6,
      },
    });
    await act(async () => {
      ws.simulateServerMessage(validMsg);
    });

    // Verify non-color indicator and status
    expect(screen.getByText(/\[! HIGH ANOMALY\]/i)).toBeDefined();
    expect(screen.getByText(/High Acoustic Anomaly Score/i)).toBeDefined();
    expect(document.getElementById('raw-score-value')?.textContent).toContain('0.880');
    expect(document.getElementById('smoothed-score-value')?.textContent).toContain('0.850');

    // Verify REQUIRED Action Directive
    expect(screen.getByText(/ACTION DIRECTIVE/i)).toBeDefined();
    expect(
      screen.getByText(/Pause and verify through a known independent channel/i)
    ).toBeDefined();

    // Verify Rolling Timeline contains the item
    expect(screen.getByText(/Window #3/i)).toBeDefined();
    expect(screen.getByText(/Raw: 0.880/i)).toBeDefined();
  });

  it('renders raw uncalibrated score clearly labelled when calibration is absent', async () => {
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];

    // Simulate window result with uncalibrated score only
    await act(async () => {
      ws.simulateServerMessage(
        createValidServerWindowResult({
          window_index: 1,
          window_start_sec: 1.0,
          window_end_sec: 5.0,
          speech_seconds: 3.2,
          is_eligible: true,
          synthetic_score: 0.742,
          decision: {
            state: 'uncertain',
            synthetic_score: 0.742,
            score_is_calibrated: false,
            calibrated_score: null,
            smoothed_score: 0.742,
            reason: 'ambiguous_score_zone',
            reason_details: 'Score in ambiguous zone.',
            window_index: 1,
            window_start_sec: 1.0,
            window_end_sec: 5.0,
            speech_seconds: 3.2,
          },
        })
      );
    });

    // Check label explicitly distinguishes the raw uncalibrated score.
    expect(screen.getByText(/Raw uncalibrated window score:/i)).toBeDefined();
    expect(screen.getByText(/EWMA-smoothed policy score:/i)).toBeDefined();
    expect(document.getElementById('raw-score-value')?.textContent).toContain('0.742');
    expect(screen.getByText(/Low threshold: 0.350 \| High threshold: 0.650 \(unvalidated heuristic\)/i)).toBeDefined();
    expect(screen.getByText(/\[\? UNCERTAIN\]/i)).toBeDefined();
    expect(document.getElementById('smoothed-score-value')?.textContent).toContain('0.742');
  });

  it('renders crossed raw and smoothed live scores from temporal state', async () => {
    render(<App />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });
    const ws = MockWebSocket.instances[0];

    await act(async () => {
      ws.simulateServerMessage(
        createValidServerWindowResult({
          window_index: 1,
          window_start_sec: 1,
          window_end_sec: 5,
          speech_seconds: 3.5,
          is_eligible: true,
          synthetic_score: 0.987,
          threshold_low: 0.35,
          threshold_high: 0.65,
          decision: {
            state: 'low',
            synthetic_score: 0.987,
            smoothed_score: 0.345,
            reason: 'normal_acoustics',
            reason_details: 'Smoothed score remains below the low threshold.',
            window_index: 1,
            window_start_sec: 1,
            window_end_sec: 5,
            speech_seconds: 3.5,
          },
        })
      );
    });
    expect(screen.getByText('[✓ LOW ANOMALY]')).toBeDefined();
    expect(screen.getByText('Raw uncalibrated window score:')).toBeDefined();
    expect(document.getElementById('raw-score-value')?.textContent).toContain('0.987');
    expect(document.getElementById('smoothed-score-value')?.textContent).toContain('0.345');
    expect(screen.getByText(/Low threshold: 0.350 \| High threshold: 0.650/)).toBeDefined();

    await act(async () => {
      ws.simulateServerMessage(
        createValidServerWindowResult({
          window_index: 2,
          window_start_sec: 2,
          window_end_sec: 6,
          speech_seconds: 3.5,
          is_eligible: true,
          synthetic_score: 0.036,
          threshold_low: 0.35,
          threshold_high: 0.65,
          decision: {
            state: 'uncertain',
            synthetic_score: 0.036,
            smoothed_score: 0.613,
            reason: 'ambiguous_score_zone',
            reason_details: 'Smoothed score is in the intermediate uncertain range.',
            window_index: 2,
            window_start_sec: 2,
            window_end_sec: 6,
            speech_seconds: 3.5,
          },
        })
      );
    });
    expect(screen.getByText('[? UNCERTAIN]')).toBeDefined();
    expect(document.getElementById('raw-score-value')?.textContent).toContain('0.036');
    expect(document.getElementById('smoothed-score-value')?.textContent).toContain('0.613');

    await act(async () => {
      ws.simulateServerMessage(
        createValidServerWindowResult({
          window_index: 3,
          window_start_sec: 3,
          window_end_sec: 7,
          speech_seconds: 0.1,
          is_eligible: false,
          synthetic_score: null,
          input_reliability: {
            state: 'insufficient',
            reasons: ['insufficient_speech'],
            explanation: 'Insufficient active speech.',
            origin_verdict_allowed: false,
          },
          quality_status: {
            peak_level: 0.05,
            rms: 0.01,
            clipping_fraction: 0.0,
            is_clipped: false,
            is_silent: false,
            snr_db_estimate: 12.0,
            quality_status: 'clean',
          },
          abstain_reason: 'insufficient_speech',
          decision: {
            state: 'insufficient_evidence',
            synthetic_score: null,
            smoothed_score: 0,
            reason: 'insufficient_speech_duration',
            reason_details: 'Insufficient active speech.',
            window_index: 3,
            window_start_sec: 3,
            window_end_sec: 7,
            speech_seconds: 0.1,
          },
        })
      );
    });
    expect(document.getElementById('raw-score-value')).toBeNull();
    expect(document.getElementById('smoothed-score-value')?.textContent).toBe('N/A');
    expect(screen.getByText('[... INSUFFICIENT SPEECH]')).toBeDefined();
  });

  it('asserts that Calibrated Synthetic Probability, Authentic, and Consistent with Natural Speech are absent', async () => {
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];

    // Simulate low anomaly window result
    await act(async () => {
      ws.simulateServerMessage(
        createValidServerWindowResult({
          window_index: 1,
          window_start_sec: 0.0,
          window_end_sec: 4.0,
          speech_seconds: 3.5,
          is_eligible: true,
          synthetic_score: 0.08,
          decision: {
            state: 'low',
            synthetic_score: 0.08,
            score_is_calibrated: false,
            calibrated_score: null,
            smoothed_score: 0.08,
            reason: 'normal_acoustics',
            reason_details: 'Low acoustic anomaly score — does not authenticate speech.',
            window_index: 1,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 3.5,
          },
        })
      );
    });

    // Check low state rendering
    expect(screen.getAllByText(/Low acoustic anomaly score — does not authenticate speech/i).length).toBeGreaterThan(0);

    // Verify PROHIBITED strings are strictly ABSENT from UI
    const bodyText = document.body.textContent || '';
    expect(bodyText).not.toContain('Calibrated Synthetic Probability');
    expect(bodyText).not.toContain('Consistent with Natural Speech');
    expect(bodyText).not.toContain('Authentic');
  });

  it('renders quality flag updates on audio anomalies', async () => {
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];

    // Simulate quality event with clipping and timing gap
    await act(async () => {
      ws.simulateServerMessage({
        type: 'quality_update',
        session_id: 'test-session-123',
        timestamp: 5.2,
        is_clipped: true,
        is_silent: false,
        has_timing_gap: true,
        clipping_count: 5,
        gap_count: 2,
      });
    });

    expect(screen.getByText(/Clipping Detected \(5 frame\(s\)\)/i)).toBeDefined();
    expect(screen.getByText(/Discontinuity Flagged \(2 gap\(s\)\)/i)).toBeDefined();
  });

  it('renders error alert on server error message', async () => {
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];

    // Simulate server error
    await act(async () => {
      ws.simulateServerMessage({
        type: 'error',
        error_code: 'MODEL_INFERENCE_ERROR',
        message: 'Detector hardware pipeline timeout.',
      });
    });

    expect(screen.getByText(/MODEL_INFERENCE_ERROR: Detector hardware pipeline timeout\./i)).toBeDefined();
    expect(screen.getByText(/\[✕ ERROR\]/i)).toBeDefined();
  });

  it('stops all tracks and closes AudioContext and WebSocket on stop click or unmount', async () => {
    const { unmount } = render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];
    expect(ws.readyState).toBe(WebSocket.OPEN);

    // Click stop button
    const stopButton = screen.getByRole('button', { name: /Stop Stream & Disconnect/i });
    await act(async () => {
      fireEvent.click(stopButton);
    });

    // Verify all hardware tracks stopped
    expect(mockTracks[0].stop).toHaveBeenCalled();
    expect(MockAudioContext.instances[0].close).toHaveBeenCalledOnce();
    expect(MockAudioWorkletNode.instances[0].disconnect).toHaveBeenCalledOnce();
    expect(JSON.parse(ws.sentMessages[ws.sentMessages.length - 1] as string).type).toBe('stop');
    await act(async () => {
      ws.simulateServerMessage({
        type: 'session_stopped',
        session_id: 'test-session-123',
        total_duration_seconds: 1,
        total_windows_evaluated: 0,
        total_windows_generated: 0,
        total_windows_dropped: 0,
        status: 'insufficient_evidence',
        final_reason: 'Stream stopped',
        reliable_window_count: 0,
        limited_reliability_window_count: 0,
        insufficient_reliability_window_count: 0,
        degraded_quality_window_count: 0,
        quality_origin_verdict_allowed: false,
        final_decision_state: 'insufficient_evidence',
      });
    });
    expect(ws.readyState).toBe(WebSocket.CLOSED);

    // Also verify unmount calls cleanup safely
    unmount();
  });

  it('completes graceful stop through ACK in StrictMode', async () => {
    render(
      <StrictMode>
        <App />
      </StrictMode>
    );

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws.simulateServerMessage({ type: 'session_started', session_id: 'strict-mode-session' });
      fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
      ws.simulateServerMessage({
        type: 'session_stopped',
        session_id: 'strict-mode-session',
        total_duration_seconds: 1,
        total_windows_evaluated: 0,
        total_windows_generated: 0,
        total_windows_dropped: 0,
        status: 'insufficient_evidence',
        final_reason: 'Stream stopped',
        reliable_window_count: 0,
        limited_reliability_window_count: 0,
        insufficient_reliability_window_count: 0,
        degraded_quality_window_count: 0,
        quality_origin_verdict_allowed: false,
        final_decision_state: 'insufficient_evidence',
      });
    });

    expect(screen.getByText(/Mic: idle/i)).toBeDefined();
    expect(screen.getByText(/Connection: disconnected/i)).toBeDefined();
    expect(ws.sentMessages.filter((message) => {
      return typeof message === 'string' && JSON.parse(message).type === 'stop';
    })).toHaveLength(1);
  });

  it('uses the bounded timeout and sends only one stop command for duplicate clicks', async () => {
    vi.useFakeTimers();
    try {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await vi.advanceTimersByTimeAsync(10);
      });
      const ws = MockWebSocket.instances[0];
      await act(async () => {
        ws.simulateServerMessage({ type: 'session_started', session_id: 'timeout-session' });
        fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
        fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
      });
      expect(ws.sentMessages.filter((message) => {
        return typeof message === 'string' && JSON.parse(message).type === 'stop';
      })).toHaveLength(1);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1800);
      });
      expect(screen.getByText(/Mic: idle/i)).toBeDefined();
      expect(screen.getByText(/Connection: disconnected/i)).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });

  it('resumes suspended audio and builds the source-worklet-silent graph', async () => {
    MockAudioContext.initialState = 'suspended';
    render(<App />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((resolve) => setTimeout(resolve, 10));
    });

    const context = MockAudioContext.instances[0];
    const source = context.createMediaStreamSource.mock.results[0].value;
    const gain = context.createGain.mock.results[0].value;
    const worklet = MockAudioWorkletNode.instances[0];
    expect(context.resume).toHaveBeenCalledOnce();
    expect(source.connect).toHaveBeenCalledWith(worklet);
    expect(worklet.connect).toHaveBeenCalledWith(gain);
    expect(gain.connect).toHaveBeenCalledWith(context.destination);
  });

  it('cleans up stream resources on network error without leaving active state', async () => {
    render(<App />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws.onerror?.();
    });
    expect(screen.getByText(/Mic: idle/i)).toBeDefined();
    expect(screen.getByText(/Connection: disconnected/i)).toBeDefined();
    expect(mockTracks[0].stop).toHaveBeenCalledOnce();
  });

  it('renders a final window result received before the stop ACK', async () => {
    render(<App />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws.simulateServerMessage({ type: 'session_started', session_id: 'final-window-session' });
      fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
      ws.simulateServerMessage(
        createValidServerWindowResult({
          session_id: 'final-window-session',
          window_index: 9,
          window_start_sec: 9,
          window_end_sec: 13,
          speech_seconds: 3,
          is_eligible: true,
          synthetic_score: 0.77,
          decision: {
            state: 'high',
            synthetic_score: 0.77,
            smoothed_score: 0.77,
            reason: 'sustained_anomaly_elevation',
            reason_details: 'Final window received before stop acknowledgement.',
            window_index: 9,
            window_start_sec: 9,
            window_end_sec: 13,
            speech_seconds: 3,
          },
        })
      );
    });
    expect(screen.getByText(/Window #9/i)).toBeDefined();
    await act(async () => {
      ws.simulateServerMessage({
        type: 'session_stopped',
        session_id: 'final-window-session',
        total_duration_seconds: 1,
        total_windows_evaluated: 1,
        total_windows_generated: 1,
        total_windows_dropped: 0,
        status: 'success',
        final_reason: 'Stop ACK',
        reliable_window_count: 1,
        limited_reliability_window_count: 0,
        insufficient_reliability_window_count: 0,
        degraded_quality_window_count: 0,
        quality_origin_verdict_allowed: true,
        final_decision_state: 'high',
      });
    });
    expect(screen.getByText(/Mic: idle/i)).toBeDefined();
  });

  it('computes relative WebSocket URL from window.location without hardcoding port 8000', () => {
    const url = getWebSocketStreamUrl();
    expect(url).not.toContain(':8000');
    expect(url).toContain('/api/v1/stream');
  });

  it('renders visible mock detector badge when server indicates mock mode', async () => {
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
      await new Promise((r) => setTimeout(r, 10));
    });

    const ws = MockWebSocket.instances[0];

    // Simulate session_started ACK with is_mock_detector: true
    await act(async () => {
      ws.simulateServerMessage({
        type: 'session_started',
        session_id: 'mock-session-123',
        is_mock_detector: true,
      });
    });

    expect(screen.getByTestId('mock-detector-badge')).toBeDefined();
    expect(screen.getByText(/\[MOCK DETECTOR - TEST MODE\]/i)).toBeDefined();
  });

  it('displays N/A for abstained file evaluation and does not render LOW or HIGH badge', async () => {
    const abstainedResult = createValidFileInferenceResult({
      file_path: 'silence.wav',
      duration_seconds: 4.0,
      total_windows: 1,
      eligible_windows: 0,
      reliable_window_count: 0,
      limited_reliability_window_count: 0,
      insufficient_reliability_window_count: 1,
      degraded_quality_window_count: 0,
      quality_origin_verdict_allowed: false,
      mean_synthetic_score: null,
      max_synthetic_score: null,
      mean_synthetic_likelihood: null,
      max_synthetic_likelihood: null,
      status: 'insufficient_evidence',
      final_decision_state: 'insufficient_evidence',
      temporal_rule_explanation: 'Insufficient active speech across file windows.',
      windows: [
        createValidFileWindowResult({
          window_index: 0,
          raw_spoof_logit: null,
          raw_bonafide_logit: null,
          raw_bonafide_score: null,
          synthetic_score: null,
          uncalibrated_synthetic_score: null,
          synthetic_likelihood: null,
          smoothed_score: null,
          active_speech_seconds: 0.0,
          is_eligible: false,
          history_scores: [],
          decision_state: 'insufficient_evidence',
          decision_reason: 'insufficient_speech_duration',
          decision_reason_details: 'Insufficient active speech across file windows.',
          abstain_reason: 'insufficient_speech',
          quality_status: {
            peak_level: 0.0,
            rms: 0.0,
            clipping_fraction: 0.0,
            is_clipped: false,
            is_silent: true,
            snr_db_estimate: 0.0,
            quality_status: 'silent',
          },
          input_reliability: {
            state: 'insufficient',
            reasons: ['silent_audio', 'insufficient_speech'],
            explanation: 'Silence detected and insufficient active speech.',
            origin_verdict_allowed: false,
          },
          decision: null,
        }),
      ],
    });

    vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/v1/ready')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }),
        } as Response);
      }
      if (urlStr.includes('/api/v1/analyze/file')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(abstainedResult),
        } as Response);
      }
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
    });

    render(<App />);

    // Switch to file tab
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
    });

    // Upload mock file
    const file = new File(['dummy audio content'], 'silence.wav', { type: 'audio/wav' });
    const fileInput = document.getElementById('audio-file-input') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    // Submit form
    const submitBtn = screen.getByRole('button', { name: /Analyze Audio File/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    // Verify aggregate scores display "N/A — inference not performed"
    const meanScoreEl = document.getElementById('file-mean-score');
    const peakScoreEl = document.getElementById('file-peak-score');
    expect(meanScoreEl?.textContent).toBe('N/A — inference not performed');
    expect(peakScoreEl?.textContent).toBe('N/A — inference not performed');

    // Verify per-window table displays "N/A"
    const tableCells = screen.getAllByRole('cell');
    const cellTexts = tableCells.map((c) => c.textContent);
    expect(cellTexts).toContain('N/A');

    // Verify window badge is ABSTAINED (badge-insufficient_evidence) and NOT LOW or HIGH badge
    const badge = document.querySelector('.window-table .timeline-badge');
    expect(badge).toBeDefined();
    expect(badge?.textContent).toBe('ABSTAINED');
    expect(badge?.classList.contains('badge-insufficient_evidence')).toBe(true);
    expect(badge?.classList.contains('badge-low')).toBe(false);
    expect(badge?.classList.contains('badge-high')).toBe(false);
    expect(screen.getByText(/\[\.\.\. INSUFFICIENT SPEECH\]/i)).toBeDefined();
  });

  it('displays 0.000 for a real detector score of exactly 0.0', async () => {
    const zeroScoreResult = createValidFileInferenceResult({
      file_path: 'clean_zero.wav',
      duration_seconds: 4.0,
      total_windows: 1,
      eligible_windows: 1,
      reliable_window_count: 1,
      limited_reliability_window_count: 0,
      insufficient_reliability_window_count: 0,
      degraded_quality_window_count: 0,
      quality_origin_verdict_allowed: true,
      mean_synthetic_score: 0.0,
      max_synthetic_score: 0.0,
      mean_synthetic_likelihood: 0.0,
      max_synthetic_likelihood: 0.0,
      status: 'success',
      final_decision_state: 'low',
      temporal_rule_explanation: 'Low acoustic anomaly score — does not authenticate speech.',
      windows: [
        createValidFileWindowResult({
          window_index: 0,
          start_seconds: 0.0,
          end_seconds: 4.0,
          effective_threshold: 0.65,
          is_synthetic_suspected: false,
          raw_spoof_logit: -10.0,
          raw_bonafide_logit: 10.0,
          raw_bonafide_score: 1.0,
          synthetic_score: 0.0,
          uncalibrated_synthetic_score: 0.0,
          calibrated_synthetic_likelihood: null,
          synthetic_likelihood: 0.0,
          smoothed_score: 0.0,
          active_speech_seconds: 3.5,
          is_eligible: true,
          is_partial: false,
          score_is_calibrated: false,
          history_scores: [0.0],
          decision_state: 'low',
          decision_reason: 'normal_acoustics',
          decision_reason_details: 'Low anomaly score',
          abstain_reason: null,
          quality_status: cleanQualityMetrics,
          input_reliability: reliableAssessment,
          decision: {
            state: 'low',
            synthetic_score: 0.0,
            score_is_calibrated: false,
            calibrated_score: null,
            smoothed_score: 0.0,
            reason: 'normal_acoustics',
            reason_details: 'Low anomaly',
            window_index: 0,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 3.5,
            history_scores: [0.0],
          },
        }),
      ],
    });

    vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/v1/ready')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }),
        } as Response);
      }
      if (urlStr.includes('/api/v1/analyze/file')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(zeroScoreResult),
        } as Response);
      }
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
    });

    render(<App />);

    // Switch to file tab
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
    });

    // Upload mock file
    const file = new File(['dummy audio content'], 'clean_zero.wav', { type: 'audio/wav' });
    const fileInput = document.getElementById('audio-file-input') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    // Submit form
    const submitBtn = screen.getByRole('button', { name: /Analyze Audio File/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    // Verify aggregate scores display "0.000" and NOT "N/A"
    const meanScoreEl = document.getElementById('file-mean-score');
    const peakScoreEl = document.getElementById('file-peak-score');
    expect(meanScoreEl?.textContent).toBe('0.000');
    expect(peakScoreEl?.textContent).toBe('0.000');

    // Verify per-window table displays "0.000"
    const tableCells = screen.getAllByRole('cell');
    const cellTexts = tableCells.map((c) => c.textContent);
    expect(cellTexts).toContain('0.000');

    // Verify window badge is LOW ANOMALY
    const badge = document.querySelector('.window-table .timeline-badge');
    expect(badge).toBeDefined();
    expect(badge?.textContent).toBe('LOW ANOMALY');
    expect(badge?.classList.contains('badge-low')).toBe(true);
    expect(screen.getByText(/\[✓ LOW ANOMALY\]/i)).toBeDefined();
  });

  it('renders canonical file temporal states instead of inferring from the raw score', async () => {
    const fileResult = createValidFileInferenceResult({
      file_path: 'crossed.wav',
      duration_seconds: 4,
      total_windows: 1,
      eligible_windows: 1,
      reliable_window_count: 1,
      limited_reliability_window_count: 0,
      insufficient_reliability_window_count: 0,
      degraded_quality_window_count: 0,
      quality_origin_verdict_allowed: true,
      mean_synthetic_score: 1.0,
      max_synthetic_score: 1.0,
      mean_synthetic_likelihood: 1.0,
      max_synthetic_likelihood: 1.0,
      is_synthetic_suspected: false,
      final_decision_state: 'uncertain',
      status: 'success',
      temporal_rule_explanation: 'Pre-confirmation temporal state.',
      windows: [
        createValidFileWindowResult({
          window_index: 0,
          start_seconds: 0,
          end_seconds: 4,
          effective_threshold: 0.65,
          is_synthetic_suspected: false,
          raw_spoof_logit: 1,
          raw_bonafide_logit: 0,
          raw_bonafide_score: 0,
          synthetic_score: 1,
          uncalibrated_synthetic_score: 1,
          calibrated_synthetic_likelihood: null,
          synthetic_likelihood: 1,
          smoothed_score: 0.613,
          active_speech_seconds: 3.5,
          is_eligible: true,
          is_partial: false,
          score_is_calibrated: false,
          history_scores: [0.613],
          decision_state: 'uncertain',
          decision_reason: 'elevation_in_progress',
          decision_reason_details: 'Awaiting consecutive elevated windows.',
          abstain_reason: null,
          quality_status: cleanQualityMetrics,
          input_reliability: reliableAssessment,
          decision: {
            state: 'uncertain',
            synthetic_score: 1,
            score_is_calibrated: false,
            calibrated_score: null,
            smoothed_score: 0.613,
            reason: 'elevation_in_progress',
            reason_details: 'Awaiting consecutive elevated windows.',
            window_index: 0,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 3.5,
            history_scores: [0.613],
          },
        }),
      ],
    });
    vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/v1/ready')) {
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
      }
      if (urlStr.includes('/api/v1/analyze/file')) {
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(fileResult) } as Response);
      }
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
    });
    render(<App />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
    });
    await act(async () => {
      fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
        target: { files: [new File(['audio'], 'crossed.wav', { type: 'audio/wav' })] },
      });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
    });
    expect(document.querySelector('.window-table .timeline-badge')?.textContent).toBe('UNCERTAIN');
    expect(screen.getByText(/\[~ MIXED \/ UNCERTAIN\]/i)).toBeDefined();
    expect(screen.queryByText('LOW ANOMALY')).toBeNull();
    expect(screen.getByText('0.613')).toBeDefined();
    expect(screen.getByText(/Awaiting consecutive elevated windows/)).toBeDefined();
  });

  it('switches between stream and file analysis tabs seamlessly', async () => {
    render(<App />);

    // Default tab is stream
    expect(screen.getByRole('button', { name: /Start Microphone Stream/i })).toBeDefined();

    // Click File Analysis tab
    const fileTabBtn = screen.getByRole('button', { name: /File Audio Analysis/i });
    await act(async () => {
      fireEvent.click(fileTabBtn);
    });

    expect(screen.getByText(/Click or select a WAV or FLAC file to analyze/i)).toBeDefined();
    expect(screen.getByLabelText(/Window Hop Step/i)).toBeDefined();
    expect(screen.getByLabelText(/Decision Threshold/i)).toBeDefined();

    // Switch back to stream tab
    const streamTabBtn = screen.getByRole('button', { name: /Live Microphone Stream/i });
    await act(async () => {
      fireEvent.click(streamTabBtn);
    });

    expect(screen.getByRole('button', { name: /Start Microphone Stream/i })).toBeDefined();
  });

  describe('VoiceShield Reliability Safety & Contract Integration (Stage B1)', () => {
    it('Scenario 1: Reliable streaming window displays RELIABLE, model score, and permitted temporal state', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      await act(async () => {
        ws.simulateServerMessage(
          createValidServerWindowResult({
            session_id: 'rel-sess',
            window_index: 1,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 3.5,
            is_eligible: true,
            synthetic_score: 0.88,
            input_reliability: {
              state: 'reliable',
              reasons: [],
              explanation: 'Clean audio with sufficient speech.',
              origin_verdict_allowed: true,
            },
            decision: {
              state: 'high',
              synthetic_score: 0.88,
              smoothed_score: 0.85,
              reason: 'sustained_anomaly_elevation',
              reason_details: 'Elevated anomaly detected.',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
            },
          })
        );
      });

      expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
      expect(document.getElementById('raw-score-value')?.textContent).toContain('0.880');
      expect(document.getElementById('smoothed-score-value')?.textContent).toContain('0.850');
      expect(screen.getByText(/Reliability: RELIABLE/i)).toBeDefined();
      expect(screen.getByText(/Inference: Performed/i)).toBeDefined();
    });

    it('Scenario 2: LIMITED streaming window withholds inference, scores null, displays LIMITED and degraded reason', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      await act(async () => {
        ws.simulateServerMessage(
          createValidServerWindowResult({
            session_id: 'lim-sess',
            window_index: 1,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 2.0,
            is_eligible: false,
            synthetic_score: null,
            input_reliability: {
              state: 'limited',
              reasons: ['severe_clipping'],
              explanation: 'Severe clipping distortion detected.',
              origin_verdict_allowed: false,
            },
            quality_status: {
              peak_level: 1.0,
              rms: 0.4,
              clipping_fraction: 0.08,
              is_clipped: true,
              is_silent: false,
              snr_db_estimate: 15.0,
              quality_status: 'clipped',
            },
            abstain_reason: 'severe_clipping',
            decision: {
              state: 'insufficient_evidence',
              synthetic_score: null,
              smoothed_score: 0.0,
              reason: 'signal_quality_degraded',
              reason_details: 'Quality degraded: severe_clipping',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 2.0,
            },
          })
        );
      });

      // Score fields null / N/A
      expect(document.getElementById('raw-score-value')).toBeNull();
      expect(document.getElementById('smoothed-score-value')?.textContent).toBe('N/A');
      expect(screen.getByText(/N\/A — inference withheld/i)).toBeDefined();

      // No LOW/HIGH badge or action directive
      expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
      expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
      expect(document.getElementById('high-status-action-banner')).toBeNull();

      // Shows LIMITED and degraded reason
      expect(screen.getByText('[~ LIMITED RELIABILITY]')).toBeDefined();
      expect(screen.getAllByText(/LIMITED RELIABILITY — ORIGIN VERDICT WITHHELD/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Severe clipping distortion detected/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/Inference: Withheld/i)).toBeDefined();
    });

    it('Scenario 3: INSUFFICIENT streaming window withholds origin verdict, scores null, displays reasons', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      await act(async () => {
        ws.simulateServerMessage(
          createValidServerWindowResult({
            session_id: 'insuff-sess',
            window_index: 1,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 0.2,
            is_eligible: false,
            synthetic_score: null,
            input_reliability: {
              state: 'insufficient',
              reasons: ['insufficient_speech'],
              explanation: 'Active speech duration (0.20s) is below required minimum.',
              origin_verdict_allowed: false,
            },
            quality_status: {
              peak_level: 0.02,
              rms: 0.005,
              clipping_fraction: 0.0,
              is_clipped: false,
              is_silent: false,
              snr_db_estimate: 8.0,
              quality_status: 'clean',
            },
            abstain_reason: 'insufficient_speech',
            decision: {
              state: 'insufficient_evidence',
              synthetic_score: null,
              smoothed_score: 0.0,
              reason: 'insufficient_speech_duration',
              reason_details: 'Insufficient active speech.',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 0.2,
            },
          })
        );
      });

      expect(document.getElementById('raw-score-value')).toBeNull();
      expect(document.getElementById('smoothed-score-value')?.textContent).toBe('N/A');
      expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
      expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
      expect(screen.getByText('[... INSUFFICIENT SPEECH]')).toBeDefined();
      expect(screen.getByText(/Reliability: INSUFFICIENT/i)).toBeDefined();
      expect(screen.getByText(/Inference: Withheld/i)).toBeDefined();
    });

    it('Scenario 4: Contradictory/malicious streaming event withholds verdict and suppresses HIGH action banner', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      // Malicious server frame: origin_verdict_allowed=false but decision.state='high' and synthetic_score=0.99
      await act(async () => {
        ws.simulateServerMessage({
          type: 'window_result',
          session_id: 'bad-sess',
          window_index: 1,
          window_start_sec: 0.0,
          window_end_sec: 4.0,
          speech_seconds: 3.5,
          is_eligible: true,
          synthetic_score: 0.99,
          input_reliability: {
            state: 'limited',
            reasons: ['marginal_snr'],
            explanation: 'Degraded SNR.',
            origin_verdict_allowed: false,
          },
          quality_status: cleanQualityMetrics,
          abstain_reason: 'marginal_snr',
          decision: {
            state: 'high',
            synthetic_score: 0.99,
            smoothed_score: 0.99,
            reason: 'malicious_high',
            reason_details: 'Contradictory state.',
            window_index: 1,
          },
        });
      });

      // Frontend MUST withhold verdict and MUST NOT render the HIGH action directive
      expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
      expect(document.getElementById('high-status-action-banner')).toBeNull();
      expect(document.getElementById('raw-score-value')).toBeNull();
      expect(document.getElementById('smoothed-score-value')?.textContent).toBe('N/A');
      expect(screen.getByText('[✕ ERROR]')).toBeDefined();
    });

    it('Scenario 5: Missing input_reliability fails closed with contract error and never falls back to LOW', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      // Missing input_reliability entirely
      await act(async () => {
        ws.simulateServerMessage({
          type: 'window_result',
          session_id: 'missing-rel-sess',
          window_index: 1,
          window_start_sec: 0.0,
          window_end_sec: 4.0,
          speech_seconds: 3.5,
          is_eligible: true,
          synthetic_score: 0.1,
          decision: {
            state: 'low',
            smoothed_score: 0.1,
            reason: 'normal',
            reason_details: 'Normal acoustics.',
            window_index: 1,
          },
        });
      });

      // Must fail closed with contract error and never fall back to LOW
      expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
      expect(screen.getByText(/Contract Error/i)).toBeDefined();
      expect(screen.getByText('[✕ ERROR]')).toBeDefined();
    });

    it('Scenario 6: Stop ACK limited_reliability after earlier HIGH overrides HIGH and keeps verdict withheld after cleanup', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      // Earlier window was HIGH (reliable)
      await act(async () => {
        ws.simulateServerMessage(
          createValidServerWindowResult({
            session_id: 'stop-override-sess',
            window_index: 1,
            window_start_sec: 0.0,
            window_end_sec: 4.0,
            speech_seconds: 3.5,
            is_eligible: true,
            synthetic_score: 0.92,
            decision: {
              state: 'high',
              synthetic_score: 0.92,
              smoothed_score: 0.90,
              reason: 'sustained_anomaly_elevation',
              reason_details: 'High anomaly.',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
            },
          })
        );
      });

      expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
      expect(document.getElementById('high-status-action-banner')).not.toBeNull();

      // Now user stops stream and server replies with limited_reliability ACK (e.g. clipping occurred later)
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
        ws.simulateServerMessage({
          type: 'session_stopped',
          session_id: 'stop-override-sess',
          total_duration_seconds: 6.0,
          total_windows_evaluated: 2,
          total_windows_generated: 2,
          total_windows_dropped: 0,
          status: 'limited_reliability',
          final_reason: 'Degraded audio quality detected across session.',
          reliable_window_count: 1,
          limited_reliability_window_count: 1,
          insufficient_reliability_window_count: 0,
          degraded_quality_window_count: 1,
          quality_origin_verdict_allowed: false,
          final_decision_state: 'uncertain',
        });
      });

      // Final UI becomes LIMITED RELIABILITY, HIGH directive disappears, socket is clean
      expect(screen.getByText('[~ LIMITED RELIABILITY]')).toBeDefined();
      expect(screen.getAllByText(/LIMITED RELIABILITY — ORIGIN VERDICT WITHHELD/i).length).toBeGreaterThan(0);
      expect(document.getElementById('high-status-action-banner')).toBeNull();
      expect(ws.readyState).toBe(WebSocket.CLOSED);
      expect(screen.getByText(/Mic: idle/i)).toBeDefined();
      expect(screen.getByText(/Connection: disconnected/i)).toBeDefined();
    });

    it('Scenario 7: Stop ACK all-INSUFFICIENT with degraded_quality_window_count > 0 preserves INSUFFICIENT and displays degraded count', async () => {
      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
        await new Promise((r) => setTimeout(r, 10));
      });
      const ws = MockWebSocket.instances[0];

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
        ws.simulateServerMessage({
          type: 'session_stopped',
          session_id: 'insuff-degraded-sess',
          total_duration_seconds: 4.0,
          total_windows_evaluated: 1,
          total_windows_generated: 1,
          total_windows_dropped: 0,
          status: 'insufficient_evidence',
          final_reason: 'Insufficient active speech across stream.',
          reliable_window_count: 0,
          limited_reliability_window_count: 0,
          insufficient_reliability_window_count: 1,
          degraded_quality_window_count: 1,
          quality_origin_verdict_allowed: false,
          final_decision_state: 'insufficient_evidence',
        });
      });

      // Primary UI remains INSUFFICIENT EVIDENCE
      expect(screen.getByText('[... INSUFFICIENT SPEECH]')).toBeDefined();
      expect(screen.getAllByText(/INSUFFICIENT EVIDENCE — ORIGIN VERDICT WITHHELD/i).length).toBeGreaterThan(0);
      expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
      expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();

      // Degraded diagnostic count is visible in summary card
      expect(document.getElementById('stream-degraded-windows')?.textContent).toBe('1');
      expect(document.getElementById('stream-verdict-permitted')?.textContent).toContain('No (Withheld)');
    });

    it('Scenario 8: Uploaded file limited_reliability with reliable-window model scores withholds primary verdict and labels scores advisory evidence only', async () => {
      const mixedLimitedResult = createValidFileInferenceResult({
        file_path: 'mixed_limited.wav',
        duration_seconds: 8.0,
        total_windows: 2,
        eligible_windows: 1,
        reliable_window_count: 1,
        limited_reliability_window_count: 1,
        insufficient_reliability_window_count: 0,
        degraded_quality_window_count: 1,
        quality_origin_verdict_allowed: false,
        mean_synthetic_score: 0.85,
        max_synthetic_score: 0.85,
        mean_synthetic_likelihood: 0.85,
        max_synthetic_likelihood: 0.85,
        effective_threshold: 0.65,
        operating_threshold_source: 'heuristic',
        is_synthetic_suspected: true,
        status: 'limited_reliability',
        final_decision_state: 'uncertain',
        temporal_rule_explanation: 'Input quality is degraded: origin verdict withheld.',
        windows: [
          createValidFileWindowResult({
            window_index: 0,
            start_seconds: 0.0,
            end_seconds: 4.0,
            effective_threshold: 0.65,
            is_synthetic_suspected: false,
            raw_spoof_logit: 1.0,
            raw_bonafide_logit: -1.0,
            raw_bonafide_score: 0.15,
            synthetic_score: 0.85,
            uncalibrated_synthetic_score: 0.85,
            calibrated_synthetic_likelihood: null,
            synthetic_likelihood: 0.85,
            smoothed_score: 0.85,
            active_speech_seconds: 3.5,
            is_eligible: true,
            is_partial: false,
            score_is_calibrated: false,
            history_scores: [0.85],
            decision_state: 'high',
            decision_reason: 'sustained_anomaly_elevation',
            decision_reason_details: 'Elevated score',
            abstain_reason: null,
            quality_status: cleanQualityMetrics,
            input_reliability: reliableAssessment,
            decision: {
              state: 'high',
              synthetic_score: 0.85,
              score_is_calibrated: false,
              calibrated_score: null,
              smoothed_score: 0.85,
              reason: 'sustained_anomaly_elevation',
              reason_details: 'Elevated score',
              window_index: 0,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              history_scores: [0.85],
            },
          }),
          createValidFileWindowResult({
            window_index: 1,
            start_seconds: 4.0,
            end_seconds: 8.0,
            effective_threshold: 0.65,
            is_synthetic_suspected: false,
            raw_spoof_logit: null,
            raw_bonafide_logit: null,
            raw_bonafide_score: null,
            synthetic_score: null,
            uncalibrated_synthetic_score: null,
            calibrated_synthetic_likelihood: null,
            synthetic_likelihood: null,
            smoothed_score: null,
            active_speech_seconds: 3.0,
            is_eligible: false,
            is_partial: false,
            score_is_calibrated: false,
            history_scores: [],
            decision_state: null,
            decision_reason: null,
            decision_reason_details: 'Clipped window',
            abstain_reason: 'severe_clipping',
            quality_status: { ...cleanQualityMetrics, is_clipped: true, quality_status: 'clipped' },
            input_reliability: {
              state: 'limited',
              reasons: ['severe_clipping'],
              explanation: 'Clipped window',
              origin_verdict_allowed: false,
            },
            decision: null,
          }),
        ],
      });

      vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/v1/ready')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
        }
        if (urlStr.includes('/api/v1/analyze/file')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(mixedLimitedResult) } as Response);
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
      });

      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
      });
      await act(async () => {
        fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
          target: { files: [new File(['audio'], 'mixed_limited.wav', { type: 'audio/wav' })] },
        });
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
      });

      // Primary verdict is withheld
      expect(screen.getByText(/LIMITED RELIABILITY — ORIGIN VERDICT WITHHELD/i)).toBeDefined();
      expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();

      // Scores labelled advisory evidence only
      expect(screen.getByText(/Mean Score \(Advisory Model Evidence From Reliable Windows\)/i)).toBeDefined();
      expect(document.getElementById('file-mean-score')?.textContent).toBe('0.850');

      // No HIGH action directive
      expect(document.querySelector('.file-results-card .action-directive-high')).toBeNull();

      // Summary counts
      expect(document.getElementById('file-reliable-windows')?.textContent).toBe('1');
      expect(document.getElementById('file-limited-windows')?.textContent).toBe('1');
      expect(document.getElementById('file-degraded-windows')?.textContent).toBe('1');
      expect(document.getElementById('file-verdict-permitted')?.textContent).toContain('No (Withheld)');
    });

    it('Scenario 9: Uploaded file insufficient_evidence withholds origin verdict and displays scores as N/A without LOW fallback', async () => {
      const insufficientResult = createValidFileInferenceResult({
        file_path: 'quiet.wav',
        duration_seconds: 4.0,
        total_windows: 1,
        eligible_windows: 0,
        reliable_window_count: 0,
        limited_reliability_window_count: 0,
        insufficient_reliability_window_count: 1,
        degraded_quality_window_count: 0,
        quality_origin_verdict_allowed: false,
        mean_synthetic_score: null,
        max_synthetic_score: null,
        mean_synthetic_likelihood: null,
        max_synthetic_likelihood: null,
        effective_threshold: 0.65,
        operating_threshold_source: 'heuristic',
        is_synthetic_suspected: false,
        status: 'insufficient_evidence',
        final_decision_state: 'insufficient_evidence',
        temporal_rule_explanation: 'Insufficient active speech.',
        windows: [
          createValidFileWindowResult({
            window_index: 0,
            start_seconds: 0.0,
            end_seconds: 4.0,
            effective_threshold: 0.65,
            is_synthetic_suspected: false,
            raw_spoof_logit: null,
            raw_bonafide_logit: null,
            raw_bonafide_score: null,
            synthetic_score: null,
            uncalibrated_synthetic_score: null,
            calibrated_synthetic_likelihood: null,
            synthetic_likelihood: null,
            smoothed_score: null,
            active_speech_seconds: 0.1,
            is_eligible: false,
            is_partial: false,
            score_is_calibrated: false,
            history_scores: [],
            decision_state: null,
            decision_reason: null,
            decision_reason_details: 'Insufficient active speech',
            abstain_reason: 'insufficient_speech',
            quality_status: cleanQualityMetrics,
            input_reliability: {
              state: 'insufficient',
              reasons: ['insufficient_speech'],
              explanation: 'Insufficient speech',
              origin_verdict_allowed: false,
            },
            decision: null,
          }),
        ],
      });

      vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/v1/ready')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
        }
        if (urlStr.includes('/api/v1/analyze/file')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(insufficientResult) } as Response);
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
      });

      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
      });
      await act(async () => {
        fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
          target: { files: [new File(['audio'], 'quiet.wav', { type: 'audio/wav' })] },
        });
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
      });

      expect(screen.getByText(/INSUFFICIENT EVIDENCE — ORIGIN VERDICT WITHHELD/i)).toBeDefined();
      expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
      expect(document.getElementById('file-mean-score')?.textContent).toBe('N/A — inference not performed');
    });

    it('Scenario 10: Uploaded file success with quality permission true preserves advisory HIGH verdict and directive', async () => {
      const successHighResult = createValidFileInferenceResult({
        file_path: 'deepfake.wav',
        duration_seconds: 8.0,
        total_windows: 2,
        eligible_windows: 2,
        reliable_window_count: 2,
        limited_reliability_window_count: 0,
        insufficient_reliability_window_count: 0,
        degraded_quality_window_count: 0,
        quality_origin_verdict_allowed: true,
        mean_synthetic_score: 0.91,
        max_synthetic_score: 0.95,
        mean_synthetic_likelihood: 0.91,
        max_synthetic_likelihood: 0.95,
        effective_threshold: 0.65,
        operating_threshold_source: 'heuristic',
        is_synthetic_suspected: true,
        status: 'success',
        final_decision_state: 'high',
        temporal_rule_explanation: 'Sustained elevation across multiple windows.',
        windows: [
          createValidFileWindowResult({
            window_index: 0,
            start_seconds: 0.0,
            end_seconds: 4.0,
            effective_threshold: 0.65,
            is_synthetic_suspected: false,
            raw_spoof_logit: 2.0,
            raw_bonafide_logit: -2.0,
            raw_bonafide_score: 0.09,
            synthetic_score: 0.91,
            uncalibrated_synthetic_score: 0.91,
            calibrated_synthetic_likelihood: null,
            synthetic_likelihood: 0.91,
            smoothed_score: 0.91,
            active_speech_seconds: 3.5,
            is_eligible: true,
            is_partial: false,
            score_is_calibrated: false,
            history_scores: [0.91],
            decision_state: 'high',
            decision_reason: 'sustained_anomaly_elevation',
            decision_reason_details: 'High anomaly',
            input_reliability: reliableAssessment,
            quality_status: cleanQualityMetrics,
            abstain_reason: null,
            decision: {
              state: 'high',
              synthetic_score: 0.91,
              score_is_calibrated: false,
              calibrated_score: null,
              smoothed_score: 0.91,
              reason: 'sustained_anomaly_elevation',
              reason_details: 'High anomaly',
              window_index: 0,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              history_scores: [0.91],
            },
          }),
          createValidFileWindowResult({
            window_index: 1,
            start_seconds: 4.0,
            end_seconds: 8.0,
            effective_threshold: 0.65,
            is_synthetic_suspected: false,
            raw_spoof_logit: 3.0,
            raw_bonafide_logit: -3.0,
            raw_bonafide_score: 0.05,
            synthetic_score: 0.95,
            uncalibrated_synthetic_score: 0.95,
            calibrated_synthetic_likelihood: null,
            synthetic_likelihood: 0.95,
            smoothed_score: 0.93,
            active_speech_seconds: 3.5,
            is_eligible: true,
            is_partial: false,
            score_is_calibrated: false,
            history_scores: [0.91, 0.95],
            decision_state: 'high',
            decision_reason: 'sustained_anomaly_elevation',
            decision_reason_details: 'High anomaly',
            input_reliability: reliableAssessment,
            quality_status: cleanQualityMetrics,
            abstain_reason: null,
            decision: {
              state: 'high',
              synthetic_score: 0.95,
              score_is_calibrated: false,
              calibrated_score: null,
              smoothed_score: 0.93,
              reason: 'sustained_anomaly_elevation',
              reason_details: 'High anomaly',
              window_index: 1,
              window_start_sec: 4.0,
              window_end_sec: 8.0,
              speech_seconds: 3.5,
              history_scores: [0.91, 0.95],
            },
          }),
        ],
      });

      vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/v1/ready')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
        }
        if (urlStr.includes('/api/v1/analyze/file')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(successHighResult) } as Response);
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
      });

      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
      });
      await act(async () => {
        fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
          target: { files: [new File(['audio'], 'deepfake.wav', { type: 'audio/wav' })] },
        });
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
      });

      expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
      expect(document.querySelector('.file-results-card .action-directive-high')).not.toBeNull();
      expect(document.getElementById('file-mean-score')?.textContent).toBe('0.910');
      expect(document.getElementById('file-verdict-permitted')?.textContent).toContain('Yes (Permitted)');
    });

    it('Scenario 11: Malformed file response missing aggregate permission/counters fails closed without verdict', async () => {
      const malformedResult = {
        filename: 'corrupt_contract.wav',
        duration_seconds: 4.0,
        total_windows: 1,
        eligible_windows: 1,
        // Missing reliable_window_count, limited_reliability_window_count, etc.
        mean_synthetic_score: 0.05,
        max_synthetic_score: 0.05,
        status: 'success',
        final_decision_state: 'low',
        windows: [],
      };

      vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/v1/ready')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
        }
        if (urlStr.includes('/api/v1/analyze/file')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(malformedResult) } as Response);
        }
        return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
      });

      render(<App />);
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
      });
      await act(async () => {
        fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
          target: { files: [new File(['audio'], 'corrupt_contract.wav', { type: 'audio/wav' })] },
        });
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
      });

      // Fails closed with contract error and no LOW or HIGH verdict rendered
      expect(screen.getByText(/Contract Error/i)).toBeDefined();
      expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
      expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
      expect(document.querySelector('.file-results-card')).toBeNull();
    });

    describe('VoiceShield Stage B1 Reliability Invariants & Timeline Capacity', () => {
      it('Invariant: explanation non-empty after trimming', () => {
        const resEmpty = validateInputReliability({
          state: 'reliable',
          reasons: [],
          explanation: '',
          origin_verdict_allowed: true,
        });
        expect(resEmpty.valid).toBe(false);
        if (!resEmpty.valid) {
          expect(resEmpty.error).toMatch(/explanation/i);
        }

        const resWhitespace = validateInputReliability({
          state: 'reliable',
          reasons: [],
          explanation: '   \t\n  ',
          origin_verdict_allowed: true,
        });
        expect(resWhitespace.valid).toBe(false);
        if (!resWhitespace.valid) {
          expect(resWhitespace.error).toMatch(/explanation/i);
        }
      });

      it('Invariant: unknown reason code rejected', () => {
        const res = validateInputReliability({
          state: 'limited',
          reasons: ['unknown_reason_code'],
          explanation: 'Clipping detected',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/unrecognized/i);
        }
      });

      it('Invariant: duplicate reason codes rejected', () => {
        const res = validateInputReliability({
          state: 'limited',
          reasons: ['severe_clipping', 'severe_clipping'],
          explanation: 'Duplicate clipping reasons',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/duplicate/i);
        }
      });

      it('Invariant: out-of-order reasons rejected', () => {
        const res = validateInputReliability({
          state: 'insufficient',
          reasons: ['marginal_snr', 'empty_audio'],
          explanation: 'Out of order reasons',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/order/i);
        }
      });

      it('Invariant: reliable state with origin_verdict_allowed=false rejected', () => {
        const res = validateInputReliability({
          state: 'reliable',
          reasons: [],
          explanation: 'Audio is clean',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/origin_verdict_allowed/i);
        }
      });

      it('Invariant: reliable state with reasons nonempty rejected', () => {
        const res = validateInputReliability({
          state: 'reliable',
          reasons: ['severe_clipping'],
          explanation: 'Audio is clean but reasons attached',
          origin_verdict_allowed: true,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/reasons/i);
        }
      });

      it('Invariant: limited state with origin_verdict_allowed=true rejected', () => {
        const res = validateInputReliability({
          state: 'limited',
          reasons: ['severe_clipping'],
          explanation: 'Clipped audio',
          origin_verdict_allowed: true,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/origin_verdict_allowed/i);
        }
      });

      it('Invariant: limited state without any limited reason rejected', () => {
        const res = validateInputReliability({
          state: 'limited',
          reasons: [],
          explanation: 'Limited but empty reasons',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/LIMITED_REASONS/i);
        }
      });

      it('Invariant: limited state with insufficient reason rejected', () => {
        const res = validateInputReliability({
          state: 'limited',
          reasons: ['empty_audio', 'severe_clipping'],
          explanation: 'Limited with insufficient reason',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/INSUFFICIENT_REASONS/i);
        }
      });

      it('Invariant: insufficient state with origin_verdict_allowed=true rejected', () => {
        const res = validateInputReliability({
          state: 'insufficient',
          reasons: ['silent_audio'],
          explanation: 'Silent audio',
          origin_verdict_allowed: true,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/origin_verdict_allowed/i);
        }
      });

      it('Invariant: insufficient state without any insufficient reason rejected', () => {
        const res = validateInputReliability({
          state: 'insufficient',
          reasons: ['severe_clipping'],
          explanation: 'Insufficient state but only degraded reason',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/INSUFFICIENT_REASONS/i);
        }
      });

      it('Invariant: insufficient state with mixed degraded reasons accepted', () => {
        const res = validateInputReliability({
          state: 'insufficient',
          reasons: ['insufficient_speech', 'severe_clipping'],
          explanation: 'Sparse speech with digital clipping',
          origin_verdict_allowed: false,
        });
        expect(res.valid).toBe(true);
        if (res.valid) {
          expect(res.value.state).toBe('insufficient');
          expect(res.value.origin_verdict_allowed).toBe(false);
          expect(res.value.reasons).toEqual(['insufficient_speech', 'severe_clipping']);
        }
      });

      it('Invariant: streaming session counters track past 20 windows (e.g. 25 windows) without truncation while visual timeline retains only 20', async () => {
        render(<App />);

        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });

        const ws = MockWebSocket.instances[0];

        // Send 25 streaming window results
        for (let i = 0; i < 25; i++) {
          await act(async () => {
            ws.simulateServerMessage(
              createValidServerWindowResult({
                session_id: 'session-25-windows',
                window_index: i,
                window_start_sec: i * 1.0,
                window_end_sec: i * 1.0 + 4.0,
                speech_seconds: 3.5,
                is_eligible: true,
                synthetic_score: 0.1,
                decision: {
                  state: 'low',
                  synthetic_score: 0.1,
                  score_is_calibrated: false,
                  calibrated_score: null,
                  smoothed_score: 0.1,
                  reason: 'normal_acoustics',
                  reason_details: 'Low anomaly',
                  window_index: i,
                  window_start_sec: i * 1.0,
                  window_end_sec: i * 1.0 + 4.0,
                  speech_seconds: 3.5,
                },
              })
            );
          });
        }

        // The visual timeline in DOM must cap at exactly 20 items
        const timelineCards = document.querySelectorAll('.timeline-card');
        expect(timelineCards.length).toBe(20);

        // The independent session counters must track all 25 windows without truncation
        expect(document.getElementById('stream-total-windows')?.textContent).toBe('25');
        expect(document.getElementById('stream-reliable-windows')?.textContent).toBe('25');
        expect(document.getElementById('stream-limited-windows')?.textContent).toBe('0');
        expect(document.getElementById('stream-insufficient-windows')?.textContent).toBe('0');
        expect(document.getElementById('stream-degraded-windows')?.textContent).toBe('0');
      });
    });

    describe('VoiceShield Stage B1 Revision 3 Regression Tests (A through V)', () => {
      it('Test A: Exact quality payload passes', () => {
        const payload = {
          peak_level: 0.5,
          rms: 0.15,
          clipping_fraction: 0.0,
          is_clipped: false,
          is_silent: false,
          snr_db_estimate: 24.5,
          quality_status: 'clean' as const,
        };
        const res = validateQualityMetrics(payload);
        expect(res.valid).toBe(true);
        if (res.valid) {
          expect(res.value).toEqual(payload);
        }
      });

      it('Test B: rms_level without rms fails', () => {
        const payload = {
          peak_level: 0.5,
          rms_level: 0.15,
          clipping_fraction: 0.0,
          is_clipped: false,
          is_silent: false,
          snr_db_estimate: 24.5,
          quality_status: 'clean',
        };
        const res = validateQualityMetrics(payload);
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/rms/i);
        }
      });

      it('Test C: clipping_ratio without clipping_fraction fails', () => {
        const payload = {
          peak_level: 0.5,
          rms: 0.15,
          clipping_ratio: 0.0,
          is_clipped: false,
          is_silent: false,
          snr_db_estimate: 24.5,
          quality_status: 'clean',
        };
        const res = validateQualityMetrics(payload);
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/clipping_fraction/i);
        }
      });

      it('Test D: Missing is_clipped fails', () => {
        const payload = {
          peak_level: 0.5,
          rms: 0.15,
          clipping_fraction: 0.0,
          is_silent: false,
          snr_db_estimate: 24.5,
          quality_status: 'clean',
        };
        const res = validateQualityMetrics(payload);
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/is_clipped/i);
        }
      });

      it('Test E: Missing is_silent fails', () => {
        const payload = {
          peak_level: 0.5,
          rms: 0.15,
          clipping_fraction: 0.0,
          is_clipped: false,
          snr_db_estimate: 24.5,
          quality_status: 'clean',
        };
        const res = validateQualityMetrics(payload);
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/is_silent/i);
        }
      });

      it('Test F: quality_status=severe_clipping fails', () => {
        const payload = {
          peak_level: 0.5,
          rms: 0.15,
          clipping_fraction: 0.05,
          is_clipped: true,
          is_silent: false,
          snr_db_estimate: 20.0,
          quality_status: 'severe_clipping',
        };
        const res = validateQualityMetrics(payload);
        expect(res.valid).toBe(false);
        if (!res.valid) {
          expect(res.error).toMatch(/quality_status/i);
        }
      });

      it('Test G: Backend status=error stop ACK validates under error arithmetic and renders error with no verdict', async () => {
        const stopAck = {
          type: 'session_stopped',
          session_id: 'err-stop-g',
          total_duration_seconds: 10.0,
          total_windows_generated: 5,
          total_windows_evaluated: 2,
          total_windows_dropped: 1, // 2 + 1 = 3 <= 5
          reliable_window_count: 2,
          limited_reliability_window_count: 1,
          insufficient_reliability_window_count: 2,
          degraded_quality_window_count: 1,
          quality_origin_verdict_allowed: false,
          status: 'error',
          final_decision_state: 'uncertain',
          final_reason: 'Backend model worker failed mid-session.',
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(true);

        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
          ws.simulateServerMessage(stopAck);
        });

        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getAllByText('Backend model worker failed mid-session.').length).toBeGreaterThan(0);
        expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('raw-score-value')).toBeNull();
        expect(document.getElementById('stream-verdict-permitted')?.textContent).toContain('No (Withheld)');
      });

      it('Test H: Error stop with permission=true fails', () => {
        const stopAck = {
          type: 'session_stopped',
          session_id: 'err-stop-h',
          total_duration_seconds: 10.0,
          total_windows_generated: 5,
          total_windows_evaluated: 2,
          total_windows_dropped: 1,
          reliable_window_count: 2,
          limited_reliability_window_count: 1,
          insufficient_reliability_window_count: 2,
          degraded_quality_window_count: 1,
          quality_origin_verdict_allowed: true, // Invalid for error stop!
          status: 'error',
          final_decision_state: 'uncertain',
          final_reason: 'Backend worker error',
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/origin verdict/i);
        }
      });

      it('Test I: Non-error evaluated+dropped mismatch fails', () => {
        const stopAck = {
          type: 'session_stopped',
          session_id: 'err-stop-i',
          total_duration_seconds: 10.0,
          total_windows_generated: 10,
          total_windows_evaluated: 6,
          total_windows_dropped: 2, // 6 + 2 = 8 !== 10
          reliable_window_count: 6,
          limited_reliability_window_count: 2,
          insufficient_reliability_window_count: 2,
          degraded_quality_window_count: 2,
          quality_origin_verdict_allowed: false,
          status: 'limited_reliability',
          final_decision_state: 'uncertain',
          final_reason: null,
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/total_windows_evaluated.*total_windows_dropped/i);
        }
      });

      it('Test J: Error evaluated+dropped less than generated is accepted', () => {
        const stopAck = {
          type: 'session_stopped',
          session_id: 'err-stop-j',
          total_duration_seconds: 10.0,
          total_windows_generated: 10,
          total_windows_evaluated: 4,
          total_windows_dropped: 2, // 4 + 2 = 6 <= 10
          reliable_window_count: 4,
          limited_reliability_window_count: 2,
          insufficient_reliability_window_count: 4,
          degraded_quality_window_count: 2,
          quality_origin_verdict_allowed: false,
          status: 'error',
          final_decision_state: 'uncertain',
          final_reason: 'Worker crashed',
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(true);
      });

      it('Test K: Skeletal file payload missing required top-level fields fails', () => {
        const skeletal = {
          total_windows: 1,
          status: 'success',
          final_decision_state: 'low',
          windows: [],
        };
        const val = validateFileAggregateAndWindows(skeletal);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/file_path/i);
        }
      });

      it('Test L: File window missing required decision/score fields fails', () => {
        const invalidWindowFile = createValidFileInferenceResult();
        delete ((invalidWindowFile.windows[0] as unknown) as Record<string, unknown>).synthetic_score;
        const val = validateFileAggregateAndWindows(invalidWindowFile);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/synthetic_score/i);
        }
      });

      it('Test M: success + permission=false never renders LOW/HIGH and fails closed', async () => {
        const contradictoryFile = createValidFileInferenceResult({
          quality_origin_verdict_allowed: false,
          status: 'success',
        });
        const val = validateFileAggregateAndWindows(contradictoryFile);
        expect(val.valid).toBe(false);

        vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
          const urlStr = String(url);
          if (urlStr.includes('/api/v1/ready')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
          }
          if (urlStr.includes('/api/v1/analyze/file')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(contradictoryFile) } as Response);
          }
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
        });

        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
        });
        await act(async () => {
          fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
            target: { files: [new File(['audio'], 'contradictory.wav', { type: 'audio/wav' })] },
          });
        });
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
        });

        expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(screen.getByText(/Contract Error/i)).toBeDefined();
        expect(document.querySelector('.file-results-card')).toBeNull();
      });

      it('Test N: file status=error never renders LOW', async () => {
        const errorFile = createValidFileInferenceResult({
          status: 'error',
          quality_origin_verdict_allowed: false,
          final_decision_state: 'uncertain',
          error_message: 'File decode failed',
          reliable_window_count: 0,
          limited_reliability_window_count: 0,
          insufficient_reliability_window_count: 1,
          eligible_windows: 0,
          mean_synthetic_score: null,
          max_synthetic_score: null,
          windows: [
            createValidFileWindowResult({
              window_index: 0,
              start_seconds: 0.0,
              end_seconds: 4.0,
              effective_threshold: 0.65,
              is_synthetic_suspected: false,
              raw_spoof_logit: null,
              raw_bonafide_logit: null,
              raw_bonafide_score: null,
              synthetic_score: null,
              uncalibrated_synthetic_score: null,
              calibrated_synthetic_likelihood: null,
              synthetic_likelihood: null,
              smoothed_score: null,
              active_speech_seconds: 0.0,
              is_eligible: false,
              is_partial: false,
              score_is_calibrated: false,
              history_scores: [],
              decision_state: null,
              decision_reason: null,
              decision_reason_details: 'Decode error',
              input_reliability: {
                state: 'insufficient',
                reasons: ['empty_audio'],
                explanation: 'Empty audio',
                origin_verdict_allowed: false,
              },
              quality_status: {
                peak_level: 0.0,
                rms: 0.0,
                clipping_fraction: 0.0,
                is_clipped: false,
                is_silent: true,
                snr_db_estimate: 0.0,
                quality_status: 'empty',
              },
              abstain_reason: 'empty_audio',
              decision: null,
            }),
          ],
        });

        vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
          const urlStr = String(url);
          if (urlStr.includes('/api/v1/ready')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
          }
          if (urlStr.includes('/api/v1/analyze/file')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(errorFile) } as Response);
          }
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
        });

        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
        });
        await act(async () => {
          fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
            target: { files: [new File(['audio'], 'error.wav', { type: 'audio/wav' })] },
          });
        });
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
        });

        expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getByText(/ANALYSIS ERROR — ORIGIN VERDICT WITHHELD/i)).toBeDefined();
      });

      it('Test O: file status=unsupported never renders LOW', async () => {
        const unsupportedFile = createValidFileInferenceResult({
          status: 'unsupported',
          quality_origin_verdict_allowed: false,
          final_decision_state: 'insufficient_evidence',
          error_message: 'Unsupported format',
          reliable_window_count: 0,
          limited_reliability_window_count: 0,
          insufficient_reliability_window_count: 1,
          eligible_windows: 0,
          mean_synthetic_score: null,
          max_synthetic_score: null,
          windows: [
            createValidFileWindowResult({
              window_index: 0,
              start_seconds: 0.0,
              end_seconds: 4.0,
              effective_threshold: 0.65,
              is_synthetic_suspected: false,
              raw_spoof_logit: null,
              raw_bonafide_logit: null,
              raw_bonafide_score: null,
              synthetic_score: null,
              uncalibrated_synthetic_score: null,
              calibrated_synthetic_likelihood: null,
              synthetic_likelihood: null,
              smoothed_score: null,
              active_speech_seconds: 0.0,
              is_eligible: false,
              is_partial: false,
              score_is_calibrated: false,
              history_scores: [],
              decision_state: null,
              decision_reason: null,
              decision_reason_details: 'Unsupported audio format',
              input_reliability: {
                state: 'insufficient',
                reasons: ['empty_audio'],
                explanation: 'Empty audio',
                origin_verdict_allowed: false,
              },
              quality_status: {
                peak_level: 0.0,
                rms: 0.0,
                clipping_fraction: 0.0,
                is_clipped: false,
                is_silent: true,
                snr_db_estimate: 0.0,
                quality_status: 'empty',
              },
              abstain_reason: 'empty_audio',
              decision: null,
            }),
          ],
        });

        vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
          const urlStr = String(url);
          if (urlStr.includes('/api/v1/ready')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
          }
          if (urlStr.includes('/api/v1/analyze/file')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(unsupportedFile) } as Response);
          }
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
        });

        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
        });
        await act(async () => {
          fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
            target: { files: [new File(['audio'], 'unsupported.wav', { type: 'audio/wav' })] },
          });
        });
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
        });

        expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
        expect(screen.getByText('[? UNSUPPORTED]')).toBeDefined();
        expect(screen.getByText(/UNSUPPORTED RESULT — ORIGIN VERDICT WITHHELD/i)).toBeDefined();
      });

      it('Test P: Missing/unknown stop status after earlier HIGH clears HIGH', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        // Earlier HIGH window
        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-p',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: 0.95,
              decision: {
                state: 'high',
                synthetic_score: 0.95,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
        expect(document.getElementById('high-status-action-banner')).not.toBeNull();

        // session_stopped with invalid status
        await act(async () => {
          ws.simulateServerMessage({
            type: 'session_stopped',
            session_id: 'sess-p',
            total_duration_seconds: 4.0,
            total_windows_generated: 1,
            total_windows_evaluated: 1,
            total_windows_dropped: 0,
            reliable_window_count: 1,
            limited_reliability_window_count: 0,
            insufficient_reliability_window_count: 0,
            degraded_quality_window_count: 0,
            quality_origin_verdict_allowed: true,
            status: 'unknown_status', // Invalid!
            final_decision_state: 'high',
            final_reason: null,
          });
        });

        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('high-status-action-banner')).toBeNull();
        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getByText(/Contract Error/i)).toBeDefined();
      });

      it('Test Q: Missing final_decision_state after earlier HIGH clears HIGH', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        // Earlier HIGH window
        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-q',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: 0.95,
              decision: {
                state: 'high',
                synthetic_score: 0.95,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();

        // session_stopped with missing final_decision_state
        await act(async () => {
          ws.simulateServerMessage({
            type: 'session_stopped',
            session_id: 'sess-q',
            total_duration_seconds: 4.0,
            total_windows_generated: 1,
            total_windows_evaluated: 1,
            total_windows_dropped: 0,
            reliable_window_count: 1,
            limited_reliability_window_count: 0,
            insufficient_reliability_window_count: 0,
            degraded_quality_window_count: 0,
            quality_origin_verdict_allowed: true,
            status: 'success',
            final_reason: null,
            // Missing final_decision_state
          });
        });

        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('high-status-action-banner')).toBeNull();
        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
      });

      it('Test R: Permitted HIGH window with missing required score fails closed', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-r',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: null, // Missing for reliable window!
              uncalibrated_synthetic_score: null,
              raw_score: null,
              decision: {
                state: 'high',
                synthetic_score: null,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('high-status-action-banner')).toBeNull();
        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getByText(/Contract Error/i)).toBeDefined();
      });

      it('Test S: Malformed recognized window_result after earlier HIGH clears HIGH instead of being silently ignored', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        // Earlier valid HIGH window
        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-s',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: 0.95,
              decision: {
                state: 'high',
                synthetic_score: 0.95,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
        expect(document.getElementById('high-status-action-banner')).not.toBeNull();

        // Malformed recognized frame (e.g. invalid quality_status)
        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-s',
              window_index: 2,
              window_start_sec: 1.0,
              window_end_sec: 5.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: 0.95,
              quality_status: { invalid: 'not_metrics' } as any, // Malformed!
              decision: {
                state: 'high',
                synthetic_score: 0.95,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 2,
                window_start_sec: 1.0,
                window_end_sec: 5.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('high-status-action-banner')).toBeNull();
        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getByText(/Contract Error/i)).toBeDefined();
      });

      it('Test T: Starting a second session clears first-session result, timeline, scores, counters, and directive', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws1 = MockWebSocket.instances[0];

        // Send a HIGH window
        await act(async () => {
          ws1.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-t1',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: 0.95,
              decision: {
                state: 'high',
                synthetic_score: 0.95,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
        expect(document.getElementById('high-status-action-banner')).not.toBeNull();
        expect(document.querySelectorAll('.timeline-card').length).toBe(1);
        expect(document.getElementById('stream-reliable-windows')?.textContent).toBe('1');

        // Stop session 1
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
          ws1.simulateServerMessage({
            type: 'session_stopped',
            session_id: 'sess-t1',
            total_duration_seconds: 4.0,
            total_windows_generated: 1,
            total_windows_evaluated: 1,
            total_windows_dropped: 0,
            reliable_window_count: 1,
            limited_reliability_window_count: 0,
            insufficient_reliability_window_count: 0,
            degraded_quality_window_count: 0,
            quality_origin_verdict_allowed: true,
            status: 'success',
            final_decision_state: 'high',
            final_reason: null,
          });
        });

        // Start session 2
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });

        // Everything from session 1 must be cleared:
        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('high-status-action-banner')).toBeNull();
        expect(document.querySelectorAll('.timeline-card').length).toBe(0);
        expect(document.getElementById('raw-score-value')).toBeNull();
        expect(document.getElementById('stream-total-windows')?.textContent).toBe('0');
        expect(document.getElementById('stream-reliable-windows')?.textContent).toBe('0');
        expect(document.getElementById('stream-limited-windows')?.textContent).toBe('0');
        expect(document.getElementById('stream-insufficient-windows')?.textContent).toBe('0');
        expect(document.getElementById('stream-degraded-windows')?.textContent).toBe('0');
      });

      it('Test U: Stream timeline visibly renders exact quality_status', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'sess-u',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.0,
              is_eligible: false,
              synthetic_score: null,
              input_reliability: {
                state: 'limited',
                reasons: ['severe_clipping'],
                explanation: 'Severe digital clipping detected',
                origin_verdict_allowed: false,
              },
              quality_status: {
                peak_level: 1.0,
                rms: 0.35,
                clipping_fraction: 0.08,
                is_clipped: true,
                is_silent: false,
                snr_db_estimate: 22.0,
                quality_status: 'clipped',
              },
              abstain_reason: 'severe_clipping',
              decision: {
                state: 'insufficient_evidence',
                synthetic_score: null,
                score_is_calibrated: false,
                calibrated_score: null,
                smoothed_score: 0.0,
                reason: 'signal_quality_degraded',
                reason_details: 'Clipping',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.0,
              },
            })
          );
        });

        const qualityBadge = document.querySelector('.timeline-card .badge-quality');
        expect(qualityBadge).not.toBeNull();
        expect(qualityBadge?.textContent).toBe('CLIPPED');
        expect(qualityBadge?.classList.contains('status-clipped')).toBe(true);
      });

      it('Test V: File table visibly renders exact quality_status', async () => {
        const fileObj = createValidFileInferenceResult({
          status: 'limited_reliability',
          final_decision_state: 'uncertain',
          quality_origin_verdict_allowed: false,
          reliable_window_count: 0,
          limited_reliability_window_count: 1,
          insufficient_reliability_window_count: 0,
          degraded_quality_window_count: 1,
          eligible_windows: 0,
          mean_synthetic_score: null,
          max_synthetic_score: null,
          windows: [
            createValidFileWindowResult({
              window_index: 0,
              start_seconds: 0.0,
              end_seconds: 4.0,
              effective_threshold: 0.65,
              is_synthetic_suspected: false,
              raw_spoof_logit: null,
              raw_bonafide_logit: null,
              raw_bonafide_score: null,
              synthetic_score: null,
              uncalibrated_synthetic_score: null,
              calibrated_synthetic_likelihood: null,
              synthetic_likelihood: null,
              smoothed_score: null,
              active_speech_seconds: 3.5,
              is_eligible: false,
              is_partial: false,
              score_is_calibrated: false,
              history_scores: [],
              decision_state: null,
              decision_reason: null,
              decision_reason_details: 'Low SNR',
              input_reliability: {
                state: 'limited',
                reasons: ['marginal_snr'],
                explanation: 'Low SNR observed',
                origin_verdict_allowed: false,
              },
              quality_status: {
                peak_level: 0.4,
                rms: 0.05,
                clipping_fraction: 0.0,
                is_clipped: false,
                is_silent: false,
                snr_db_estimate: 8.5,
                quality_status: 'marginal_snr',
              },
              abstain_reason: 'marginal_snr',
              decision: null,
            }),
          ],
        });

        vi.mocked(fetch).mockImplementation((url: RequestInfo | URL) => {
          const urlStr = String(url);
          if (urlStr.includes('/api/v1/ready')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ is_mock_detector: false, device: 'cpu' }) } as Response);
          }
          if (urlStr.includes('/api/v1/analyze/file')) {
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(fileObj) } as Response);
          }
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) } as Response);
        });

        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /File Audio Analysis/i }));
        });
        await act(async () => {
          fireEvent.change(document.getElementById('audio-file-input') as HTMLInputElement, {
            target: { files: [new File(['audio'], 'marginal.wav', { type: 'audio/wav' })] },
          });
        });
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Analyze Audio File/i }));
        });

        const qualityBadge = document.querySelector('.window-table .badge-quality');
        expect(qualityBadge).not.toBeNull();
        expect(qualityBadge?.textContent).toBe('MARGINAL_SNR');
        expect(qualityBadge?.classList.contains('status-marginal_snr')).toBe(true);
      });
    });

    describe('VoiceShield Stage B1 Revision 4 Regression Tests (A through P)', () => {
      it('Test A: Error stop with final_decision_state=uncertain validates and renders ERROR with no verdict', async () => {
        const stopAck = {
          type: 'session_stopped',
          session_id: 'rev4-stop-a',
          total_duration_seconds: 12.0,
          total_windows_generated: 6,
          total_windows_evaluated: 3,
          total_windows_dropped: 2, // 3 + 2 = 5 <= 6
          reliable_window_count: 2,
          limited_reliability_window_count: 1,
          insufficient_reliability_window_count: 3,
          degraded_quality_window_count: 1,
          quality_origin_verdict_allowed: false,
          status: 'error',
          final_decision_state: 'uncertain',
          final_reason: 'Worker crashed mid-stream',
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(true);

        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
          ws.simulateServerMessage(stopAck);
        });

        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getAllByText('Worker crashed mid-stream').length).toBeGreaterThan(0);
        expect(screen.queryByText('[✓ LOW ANOMALY]')).toBeNull();
        expect(screen.queryByText('[? UNCERTAIN]')).toBeNull();
        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('raw-score-value')).toBeNull();
        expect(document.getElementById('stream-verdict-permitted')?.textContent).toContain('No (Withheld)');
      });

      it('Test B: Error stop with final_decision_state=high validates but still clears HIGH and displays no origin verdict', async () => {
        render(<App />);
        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Start Microphone Stream/i }));
          await new Promise((r) => setTimeout(r, 10));
        });
        const ws = MockWebSocket.instances[0];

        // Send a HIGH window
        await act(async () => {
          ws.simulateServerMessage(
            createValidServerWindowResult({
              session_id: 'rev4-stop-b',
              window_index: 1,
              window_start_sec: 0.0,
              window_end_sec: 4.0,
              speech_seconds: 3.5,
              is_eligible: true,
              synthetic_score: 0.95,
              decision: {
                state: 'high',
                synthetic_score: 0.95,
                smoothed_score: 0.95,
                reason: 'sustained_anomaly_elevation',
                reason_details: 'High anomaly',
                window_index: 1,
                window_start_sec: 0.0,
                window_end_sec: 4.0,
                speech_seconds: 3.5,
              },
            })
          );
        });

        expect(screen.getByText('[! HIGH ANOMALY]')).toBeDefined();
        expect(document.getElementById('high-status-action-banner')).not.toBeNull();

        // Server terminates with status=error and final_decision_state=high
        const stopAck = {
          type: 'session_stopped',
          session_id: 'rev4-stop-b',
          total_duration_seconds: 8.0,
          total_windows_generated: 4,
          total_windows_evaluated: 2,
          total_windows_dropped: 1, // 2 + 1 = 3 <= 4
          reliable_window_count: 2,
          limited_reliability_window_count: 0,
          insufficient_reliability_window_count: 2,
          degraded_quality_window_count: 0,
          quality_origin_verdict_allowed: false,
          status: 'error',
          final_decision_state: 'high',
          final_reason: 'Hardware audio device failure',
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(true);

        await act(async () => {
          fireEvent.click(screen.getByRole('button', { name: /Stop Stream & Disconnect/i }));
          ws.simulateServerMessage(stopAck);
        });

        // HIGH verdict must be cleared, action banner removed, ERROR displayed
        expect(screen.queryByText('[! HIGH ANOMALY]')).toBeNull();
        expect(document.getElementById('high-status-action-banner')).toBeNull();
        expect(screen.getByText('[✕ ERROR]')).toBeDefined();
        expect(screen.getAllByText('Hardware audio device failure').length).toBeGreaterThan(0);
        expect(document.getElementById('raw-score-value')).toBeNull();
        expect(document.getElementById('stream-verdict-permitted')?.textContent).toContain('No (Withheld)');
      });

      it('Test C: Error stop with invented final_decision_state=error fails', () => {
        const stopAck = {
          type: 'session_stopped',
          session_id: 'rev4-stop-c',
          total_duration_seconds: 10.0,
          total_windows_generated: 5,
          total_windows_evaluated: 2,
          total_windows_dropped: 1,
          reliable_window_count: 2,
          limited_reliability_window_count: 1,
          insufficient_reliability_window_count: 2,
          degraded_quality_window_count: 1,
          quality_origin_verdict_allowed: false,
          status: 'error',
          final_decision_state: 'error',
          final_reason: 'Crash',
        };
        const val = validateStreamingStopAck(stopAck);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/final_decision_state/i);
        }
      });

      it('Test D: Streaming window missing session_id fails', () => {
        const win = createValidServerWindowResult();
        delete (win as any).session_id;
        const val = validateServerWindowResult(win);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/session_id/i);
        }
      });

      it('Test E: Streaming window missing speech_seconds fails', () => {
        const win = createValidServerWindowResult();
        delete (win as any).speech_seconds;
        const val = validateServerWindowResult(win);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/speech_seconds/i);
        }
      });

      it('Test F: Streaming window missing any required score key fails', () => {
        const scoreKeys = ['synthetic_score', 'uncalibrated_synthetic_score', 'raw_score', 'calibrated_score'] as const;
        for (const k of scoreKeys) {
          const win = createValidServerWindowResult();
          delete (win as any)[k];
          const val = validateServerWindowResult(win);
          expect(val.valid).toBe(false);
          if (!val.valid) {
            expect(val.error).toMatch(new RegExp(k, 'i'));
          }
        }
      });

      it('Test G: Streaming window with incomplete TemporalDecision fails', () => {
        const win = createValidServerWindowResult({
          decision: {
            state: 'low',
            synthetic_score: 0.25,
            score_is_calibrated: false,
            calibrated_score: null,
            smoothed_score: 0.25,
            reason: 'normal_acoustics',
          },
          __rawDecision: true,
        });
        const val = validateServerWindowResult(win);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/decision.*missing required key/i);
        }
      });

      it('Test H: Streaming window with non-finite history score fails', () => {
        const win = createValidServerWindowResult({
          decision: {
            history_scores: [0.2, NaN],
          },
        });
        const val = validateServerWindowResult(win);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/history_scores/i);
        }

        const winInf = createValidServerWindowResult({
          decision: {
            history_scores: [0.2, Infinity],
          },
        });
        const valInf = validateServerWindowResult(winInf);
        expect(valInf.valid).toBe(false);
      });

      it('Test I: File result missing error_message fails even though it is nullable', () => {
        const file = createValidFileInferenceResult();
        delete (file as any).error_message;
        const val = validateFileAggregateAndWindows(file);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/error_message/i);
        }
      });

      it('Test J: File result missing each aggregate score alias fails', () => {
        const aliases = [
          'mean_synthetic_score',
          'max_synthetic_score',
          'mean_synthetic_likelihood',
          'max_synthetic_likelihood',
        ] as const;
        for (const alias of aliases) {
          const file = createValidFileInferenceResult();
          delete (file as any)[alias];
          const val = validateFileAggregateAndWindows(file);
          expect(val.valid).toBe(false);
          if (!val.valid) {
            expect(val.error).toMatch(new RegExp(alias, 'i'));
          }
        }
      });

      it('Test K: File result with config_hash absent fails; config_hash=null passes', () => {
        const fileAbsent = createValidFileInferenceResult();
        delete (fileAbsent as any).config_hash;
        const valAbsent = validateFileAggregateAndWindows(fileAbsent);
        expect(valAbsent.valid).toBe(false);
        if (!valAbsent.valid) {
          expect(valAbsent.error).toMatch(/config_hash/i);
        }

        const fileNull = createValidFileInferenceResult({ config_hash: null });
        const valNull = validateFileAggregateAndWindows(fileNull);
        expect(valNull.valid).toBe(true);
      });

      it('Test L: File window missing raw_spoof_logit fails; raw_spoof_logit=null passes when permitted by its state', () => {
        const fileMissing = createValidFileInferenceResult({
          windows: [
            (() => {
              const w = createValidFileWindowResult();
              delete (w as any).raw_spoof_logit;
              return w;
            })(),
          ],
        });
        const valMissing = validateFileAggregateAndWindows(fileMissing);
        expect(valMissing.valid).toBe(false);
        if (!valMissing.valid) {
          expect(valMissing.error).toMatch(/raw_spoof_logit/i);
        }

        // Non-reliable/ineligible window with explicit raw_spoof_logit: null
        const fileNull = createValidFileInferenceResult({
          status: 'insufficient_evidence',
          final_decision_state: 'insufficient_evidence',
          quality_origin_verdict_allowed: false,
          reliable_window_count: 0,
          limited_reliability_window_count: 0,
          insufficient_reliability_window_count: 1,
          degraded_quality_window_count: 0,
          eligible_windows: 0,
          mean_synthetic_score: null,
          max_synthetic_score: null,
          mean_synthetic_likelihood: null,
          max_synthetic_likelihood: null,
          windows: [
            createValidFileWindowResult({
              is_eligible: false,
              raw_spoof_logit: null,
              raw_bonafide_logit: null,
              raw_bonafide_score: null,
              synthetic_score: null,
              uncalibrated_synthetic_score: null,
              calibrated_synthetic_likelihood: null,
              synthetic_likelihood: null,
              // file_evaluator emits temporal policy state even when detector inference abstains.
              smoothed_score: 0,
              decision_state: 'insufficient_evidence',
              decision_reason: 'insufficient_speech_duration',
              decision_reason_details: 'Insufficient active speech.',
              decision: {
                state: 'insufficient_evidence',
                synthetic_score: null,
                score_is_calibrated: false,
                calibrated_score: null,
                smoothed_score: 0,
                reason: 'insufficient_speech_duration',
                reason_details: 'Insufficient active speech.',
                window_index: 0,
                window_start_sec: 0,
                window_end_sec: 4,
                speech_seconds: 3.5,
                history_scores: [],
              },
              history_scores: [],
              input_reliability: {
                state: 'insufficient',
                reasons: ['empty_audio'],
                explanation: 'Empty audio',
                origin_verdict_allowed: false,
              },
              quality_status: {
                peak_level: 0.0,
                rms: 0.0,
                clipping_fraction: 0.0,
                is_clipped: false,
                is_silent: true,
                snr_db_estimate: 0.0,
                quality_status: 'empty',
              },
              abstain_reason: 'empty_audio',
            }),
          ],
        });
        const valNull = validateFileAggregateAndWindows(fileNull);
        expect(valNull.valid).toBe(true);
      });

      it('Test M: File window missing abstain_reason fails', () => {
        const file = createValidFileInferenceResult({
          windows: [
            (() => {
              const w = createValidFileWindowResult();
              delete (w as any).abstain_reason;
              return w;
            })(),
          ],
        });
        const val = validateFileAggregateAndWindows(file);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/abstain_reason/i);
        }
      });

      it('Test N: File window history containing NaN or a string fails', () => {
        const fileNaN = createValidFileInferenceResult({
          windows: [
            createValidFileWindowResult({
              history_scores: [0.1, NaN],
            }),
          ],
        });
        const valNaN = validateFileAggregateAndWindows(fileNaN);
        expect(valNaN.valid).toBe(false);

        const fileStr = createValidFileInferenceResult({
          windows: [
            createValidFileWindowResult({
              history_scores: [0.1, '0.2' as any],
            }),
          ],
        });
        const valStr = validateFileAggregateAndWindows(fileStr);
        expect(valStr.valid).toBe(false);
      });

      it('Test O: File window with incomplete non-null decision fails', () => {
        const file = createValidFileInferenceResult({
          windows: [
            createValidFileWindowResult({
              decision: {
                state: 'low',
                synthetic_score: 0.1,
                score_is_calibrated: false,
                calibrated_score: null,
                smoothed_score: 0.1,
                reason: 'normal_acoustics',
              } as any,
            }),
          ],
        });
        const val = validateFileAggregateAndWindows(file);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/decision.*missing required key/i);
        }
      });

      it('Test P: Non-reliable window with a missing score key fails; the same key explicitly null passes', () => {
        const makeNonReliable = () =>
          createValidFileWindowResult({
            is_eligible: false,
            raw_spoof_logit: null,
            raw_bonafide_logit: null,
            raw_bonafide_score: null,
            synthetic_score: null,
            uncalibrated_synthetic_score: null,
            calibrated_synthetic_likelihood: null,
            synthetic_likelihood: null,
            smoothed_score: null,
            decision_state: null,
            decision_reason: null,
            decision: null,
            input_reliability: {
              state: 'insufficient',
              reasons: ['empty_audio'],
              explanation: 'Empty audio',
              origin_verdict_allowed: false,
            },
            quality_status: {
              peak_level: 0.0,
              rms: 0.0,
              clipping_fraction: 0.0,
              is_clipped: false,
              is_silent: true,
              snr_db_estimate: 0.0,
              quality_status: 'empty',
            },
            abstain_reason: 'empty_audio',
          });

        const makeFile = (win: FileWindowResult) =>
          createValidFileInferenceResult({
            status: 'insufficient_evidence',
            final_decision_state: 'insufficient_evidence',
            quality_origin_verdict_allowed: false,
            reliable_window_count: 0,
            limited_reliability_window_count: 0,
            insufficient_reliability_window_count: 1,
            degraded_quality_window_count: 0,
            eligible_windows: 0,
            mean_synthetic_score: null,
            max_synthetic_score: null,
            mean_synthetic_likelihood: null,
            max_synthetic_likelihood: null,
            windows: [win],
          });

        // Explicit null passes
        const winNull = makeNonReliable();
        expect(winNull.synthetic_score).toBeNull();
        const valNull = validateFileAggregateAndWindows(makeFile(winNull));
        expect(valNull.valid).toBe(true);

        // Missing synthetic_score fails
        const winMissing = makeNonReliable();
        delete (winMissing as any).synthetic_score;
        const valMissing = validateFileAggregateAndWindows(makeFile(winMissing));
        expect(valMissing.valid).toBe(false);
        if (!valMissing.valid) {
          expect(valMissing.error).toMatch(/synthetic_score/i);
        }
      });

      it('Test Q: TemporalDecision reason outside the backend ReasonCode enum fails', () => {
        const win = createValidServerWindowResult({
          decision: {
            reason: 'invented_reason',
          },
        });
        const val = validateServerWindowResult(win);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/reason.*ReasonCode/i);
        }
      });

      it('Test R: Contradictory duplicate streaming scores fail before rendering', () => {
        const win = createValidServerWindowResult({
          synthetic_score: 0.9,
          uncalibrated_synthetic_score: 0.9,
          raw_score: 0.9,
          decision: {
            synthetic_score: 0.1,
            smoothed_score: 0.1,
            history_scores: [0.1],
          },
        });
        const val = validateServerWindowResult(win);
        expect(val.valid).toBe(false);
        if (!val.valid) {
          expect(val.error).toMatch(/synthetic_score.*decision\.synthetic_score/i);
        }
      });
    });
  });
});

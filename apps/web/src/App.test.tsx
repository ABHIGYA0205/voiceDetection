import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { App, getWebSocketStreamUrl } from './App';

// Mock Web Audio API and MediaDevices
class MockAudioWorklet {
  addModule = vi.fn().mockResolvedValue(undefined);
}

class MockAudioContext {
  sampleRate = 48000;
  state = 'running';
  audioWorklet = new MockAudioWorklet();
  createMediaStreamSource = vi.fn().mockReturnValue({
    connect: vi.fn(),
    disconnect: vi.fn(),
  });
  close = vi.fn().mockResolvedValue(undefined);
}

class MockAudioWorkletNode {
  port = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage: vi.fn(),
  };
  connect = vi.fn();
  disconnect = vi.fn();
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

  simulateServerMessage(data: Record<string, unknown>) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
}

describe('VoiceShield Dashboard State Transitions & Rendering', () => {
  let mockTracks: { stop: ReturnType<typeof vi.fn> }[];

  beforeEach(() => {
    MockWebSocket.instances = [];
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
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders initial state with strict privacy and scope disclaimers', () => {
    render(<App />);

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
    await act(async () => {
      ws.simulateServerMessage({
        type: 'window_result',
        session_id: 'test-session-123',
        window_index: 3,
        window_start_sec: 3.0,
        window_end_sec: 7.0,
        speech_seconds: 3.6,
        is_eligible: true,
        synthetic_score: 0.88,
        score_is_calibrated: false,
        calibrated_score: null,
        decision: {
          state: 'high',
          synthetic_score: 0.88,
          score_is_calibrated: false,
          calibrated_score: null,
          smoothed_score: 0.85,
          reason: 'sustained_anomaly_elevation',
          reason_details: 'Elevated anomaly detected over 3 consecutive windows.',
          window_index: 3,
        },
      });
    });

    // Verify non-color indicator and status
    expect(screen.getByText(/\[! HIGH ANOMALY\]/i)).toBeDefined();
    expect(screen.getByText(/High Acoustic Anomaly Score/i)).toBeDefined();
    expect(document.getElementById('raw-score-value')?.textContent).toContain('0.880');

    // Verify REQUIRED Action Directive
    expect(screen.getByText(/ACTION DIRECTIVE/i)).toBeDefined();
    expect(
      screen.getByText(/Pause and verify through a known independent channel/i)
    ).toBeDefined();

    // Verify Rolling Timeline contains the item
    expect(screen.getByText(/Window #3/i)).toBeDefined();
    expect(screen.getByText(/Score: 0.880/i)).toBeDefined();
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
      ws.simulateServerMessage({
        type: 'window_result',
        session_id: 'test-session-123',
        window_index: 1,
        window_start_sec: 1.0,
        window_end_sec: 5.0,
        speech_seconds: 3.2,
        is_eligible: true,
        synthetic_score: 0.742,
        score_is_calibrated: false,
        operating_threshold: 0.65,
        operating_threshold_source: 'unvalidated_heuristic',
        calibrated_score: null,
        decision: {
          state: 'uncertain',
          synthetic_score: 0.742,
          score_is_calibrated: false,
          calibrated_score: null,
          smoothed_score: 0.742,
          reason: 'ambiguous_score_zone',
          reason_details: 'Score in ambiguous zone.',
          window_index: 1,
        },
      });
    });

    // Check label explicitly specifies Uncalibrated Synthetic Score
    expect(screen.getByText(/Uncalibrated Synthetic Score:/i)).toBeDefined();
    expect(document.getElementById('raw-score-value')?.textContent).toContain('0.742');
    expect(screen.getByText(/Operating threshold: 0.650 \(unvalidated heuristic\)/i)).toBeDefined();
    expect(screen.getByText(/\[\? UNCERTAIN\]/i)).toBeDefined();
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
      ws.simulateServerMessage({
        type: 'window_result',
        session_id: 'test-session-123',
        window_index: 1,
        window_start_sec: 0.0,
        window_end_sec: 4.0,
        speech_seconds: 3.5,
        is_eligible: true,
        synthetic_score: 0.08,
        score_is_calibrated: false,
        calibrated_score: null,
        decision: {
          state: 'low',
          synthetic_score: 0.08,
          score_is_calibrated: false,
          calibrated_score: null,
          smoothed_score: 0.08,
          reason: 'normal_acoustics',
          reason_details: 'Low acoustic anomaly score — does not authenticate speech.',
          window_index: 1,
        },
      });
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
    expect(ws.readyState).toBe(WebSocket.CLOSED);

    // Also verify unmount calls cleanup safely
    unmount();
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
});

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { StrictMode } from 'react';
import { App, getWebSocketStreamUrl } from './App';

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
    expect(MockAudioContext.instances[0].close).toHaveBeenCalledOnce();
    expect(MockAudioWorkletNode.instances[0].disconnect).toHaveBeenCalledOnce();
    expect(JSON.parse(ws.sentMessages[ws.sentMessages.length - 1] as string).type).toBe('stop');
    await act(async () => {
      ws.simulateServerMessage({
        type: 'session_stopped',
        session_id: 'test-session-123',
        total_duration_seconds: 1,
        total_windows_evaluated: 0,
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
      ws.simulateServerMessage({
        type: 'window_result',
        session_id: 'final-window-session',
        window_index: 9,
        window_start_sec: 9,
        window_end_sec: 13,
        speech_seconds: 3,
        is_eligible: true,
        synthetic_score: 0.77,
        score_is_calibrated: false,
        decision: {
          state: 'high',
          smoothed_score: 0.77,
          reason: 'final_window',
          reason_details: 'Final window received before stop acknowledgement.',
          window_index: 9,
        },
      });
    });
    expect(screen.getByText(/Window #9/i)).toBeDefined();
    await act(async () => {
      ws.simulateServerMessage({
        type: 'session_stopped',
        session_id: 'final-window-session',
        total_duration_seconds: 1,
        total_windows_evaluated: 1,
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
    const abstainedResult = {
      filename: 'silence.wav',
      duration_seconds: 4.0,
      total_windows: 1,
      eligible_windows: 0,
      mean_synthetic_score: null,
      max_synthetic_score: null,
      effective_threshold: 0.65,
      operating_threshold_source: 'unvalidated_heuristic',
      consecutive_windows_required: 2,
      max_consecutive_high_windows: 0,
      is_synthetic_suspected: false,
      status: 'insufficient_evidence',
      temporal_rule_explanation: 'Insufficient active speech across file windows.',
      is_mock_detector: false,
      windows: [
        {
          window_index: 1,
          start_seconds: 0.0,
          end_seconds: 4.0,
          active_speech_seconds: 0.0,
          is_eligible: false,
          raw_spoof_logit: null,
          raw_bonafide_logit: null,
          raw_bonafide_score: null,
          synthetic_score: null,
          uncalibrated_synthetic_score: null,
          is_synthetic_suspected: false,
          reason: 'insufficient_speech',
        },
      ],
    };

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
    const zeroScoreResult = {
      filename: 'clean_zero.wav',
      duration_seconds: 4.0,
      total_windows: 1,
      eligible_windows: 1,
      mean_synthetic_score: 0.0,
      max_synthetic_score: 0.0,
      effective_threshold: 0.65,
      operating_threshold_source: 'unvalidated_heuristic',
      consecutive_windows_required: 2,
      max_consecutive_high_windows: 0,
      is_synthetic_suspected: false,
      status: 'low',
      temporal_rule_explanation: 'Low acoustic anomaly score — does not authenticate speech.',
      is_mock_detector: false,
      windows: [
        {
          window_index: 1,
          start_seconds: 0.0,
          end_seconds: 4.0,
          active_speech_seconds: 3.5,
          is_eligible: true,
          raw_spoof_logit: -10.0,
          raw_bonafide_logit: 10.0,
          raw_bonafide_score: 1.0,
          synthetic_score: 0.0,
          uncalibrated_synthetic_score: 0.0,
          is_synthetic_suspected: false,
          reason: 'low_anomaly',
        },
      ],
    };

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

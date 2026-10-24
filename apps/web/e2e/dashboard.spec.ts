import { test, expect } from '@playwright/test';

test.describe('VoiceShield Real-Time Microphone Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Inject synthetic audio source and mocked WebSocket protocol
    await page.addInitScript(() => {
      // 1. Synthetic Web Audio MediaStream Source
      const OriginalAudioContext =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

      if (navigator.mediaDevices) {
        navigator.mediaDevices.getUserMedia = async (constraints) => {
          // Create synthetic oscillator stream
          const ctx = new OriginalAudioContext();
          const osc = ctx.createOscillator();
          const dst = ctx.createMediaStreamDestination();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(440, ctx.currentTime);
          osc.connect(dst);
          osc.start();
          return dst.stream;
        };
      }

      // 2. Mock WebSocket for VoiceShield streaming
      class MockStreamingWebSocket extends EventTarget {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        url: string;
        readyState = MockStreamingWebSocket.CONNECTING;
        binaryType = 'arraybuffer';
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;

        private frameCount = 0;

        constructor(url: string) {
          super();
          this.url = url;
          setTimeout(() => {
            this.readyState = MockStreamingWebSocket.OPEN;
            const ev = new Event('open');
            if (this.onopen) this.onopen(ev);
            this.dispatchEvent(ev);
          }, 50);
        }

        send(data: string | ArrayBuffer) {
          if (typeof data === 'string') {
            const parsed = JSON.parse(data);
            if (parsed.type === 'start') {
              // Respond with session_started ACK
              setTimeout(() => {
                this._dispatchMessage({
                  type: 'session_started',
                  session_id: 'mock-session-e2e',
                  protocol_version: '1.0',
                  canonical_sample_rate: 16000,
                  canonical_channels: 1,
                  window_seconds: 4.0,
                  hop_seconds: 1.0,
                  min_speech_seconds: 0.8,
                  max_frame_bytes: 131072,
                  max_stream_duration_seconds: 3600.0,
                  idle_timeout_seconds: 30.0,
                  is_mock_detector: true,
                });
              }, 50);
            } else if (parsed.type === 'stop') {
              setTimeout(() => {
                this._dispatchMessage({
                  type: 'session_stopped',
                  session_id: 'mock-session-e2e',
                  total_duration_seconds: 5.2,
                  total_windows_evaluated: 1,
                });
                this.close();
              }, 50);
            }
          } else if (data instanceof ArrayBuffer) {
            this.frameCount++;
            // After receiving multiple synthetic audio frames, emit quality and window results
            if (this.frameCount === 5) {
              setTimeout(() => {
                this._dispatchMessage({
                  type: 'quality_update',
                  session_id: 'mock-session-e2e',
                  timestamp: 2.5,
                  is_clipped: false,
                  is_silent: false,
                  has_timing_gap: false,
                  clipping_count: 0,
                  gap_count: 0,
                });
              }, 50);
            }

            if (this.frameCount === 10) {
              setTimeout(() => {
                this._dispatchMessage({
                  type: 'window_result',
                  session_id: 'mock-session-e2e',
                  window_index: 0,
                  window_start_sec: 0.0,
                  window_end_sec: 4.0,
                  speech_seconds: 3.8,
                  is_eligible: true,
                  synthetic_score: 0.87,
                  uncalibrated_synthetic_score: 0.87,
                  score_is_calibrated: false,
                  calibrated_score: null,
                  operating_threshold: 0.65,
                  operating_threshold_source: 'unvalidated_heuristic',
                  decision: {
                    state: 'high',
                    synthetic_score: 0.87,
                    score_is_calibrated: false,
                    calibrated_score: null,
                    smoothed_score: 0.85,
                    reason: 'sustained_anomaly_elevation',
                    reason_details:
                      'Elevated acoustic anomaly detected: 3 of last 3 windows exceeded 0.65 threshold.',
                    window_index: 0,
                    history_scores: [0.82, 0.85, 0.87],
                  },
                });
              }, 100);
            }
          }
        }

        close(code?: number, reason?: string) {
          this.readyState = MockStreamingWebSocket.CLOSED;
          const ev = new CloseEvent('close', { code: code || 1000, reason: reason || 'Closed' });
          if (this.onclose) this.onclose(ev);
          this.dispatchEvent(ev);
        }

        private _dispatchMessage(data: Record<string, unknown>) {
          const ev = new MessageEvent('message', { data: JSON.stringify(data) });
          if (this.onmessage) this.onmessage(ev);
          this.dispatchEvent(ev);
        }
      }

      // @ts-expect-error Mocking WebSocket
      window.WebSocket = MockStreamingWebSocket;
    });
  });

  test('displays disclaimers, activates microphone stream, and renders high anomaly action directive', async ({
    page,
  }) => {
    await page.goto('/');

    // 1. Verify Scope & Privacy Notices
    await expect(page.locator('h1.brand-title')).toHaveText('VoiceShield');
    await expect(page.locator('p.brand-subtitle')).toContainText(
      'Local input only — does not capture cellular telephony'
    );
    await expect(page.locator('#compliance-banner')).toContainText(
      'Audio is processed in volatile memory and immediately discarded. Audio is never stored in your browser or on the server.'
    );

    // Initial status should be insufficient speech / awaiting activation
    await expect(page.locator('#accessible-status-indicator')).toHaveText(
      '[... INSUFFICIENT SPEECH]'
    );
    await expect(page.locator('#connection-state-pill')).toContainText('disconnected');
    await expect(page.locator('#recording-state-pill')).toContainText('idle');

    // 2. Click Start Microphone Stream (Requires User Gesture)
    const startButton = page.locator('#start-mic-button');
    await expect(startButton).toBeVisible();
    await startButton.click();

    // 3. Verify Connection and Recording States Update
    await expect(page.locator('#connection-state-pill')).toContainText('connected');
    await expect(page.locator('#recording-state-pill')).toContainText('recording');

    // Stop button replaces start button
    const stopButton = page.locator('#stop-mic-button');
    await expect(stopButton).toBeVisible();

    // 4. Await streaming window inference result from mock WebSocket
    // Expect status to transition to HIGH synthetic anomaly score
    await expect(page.locator('#accessible-status-indicator')).toHaveText('[! HIGH ANOMALY]', {
      timeout: 10000,
    });
    await expect(page.locator('#status-heading')).toHaveText('High Acoustic Anomaly Score');
    await expect(page.locator('#raw-score-value')).toContainText('0.870');

    // 5. Verify REQUIRED High Anomaly Action Directive Banner is visible
    const actionBanner = page.locator('#high-status-action-banner');
    await expect(actionBanner).toBeVisible();
    await expect(actionBanner.locator('.action-strong')).toHaveText(
      'Pause and verify through a known independent channel.'
    );

    // 6. Verify Rolling Timeline reflects the window item
    const timelineCard = page.locator('.timeline-card').first();
    await expect(timelineCard).toBeVisible();
    await expect(timelineCard.locator('.timeline-idx')).toHaveText('Window #0');
    await expect(timelineCard.locator('.timeline-score')).toContainText('Score: 0.870');

    // 7. Verify Compliance: forbidden claims are NEVER present
    const content = await page.content();
    expect(content).not.toContain('identity verified');
    expect(content).not.toContain('fraud detected');
    expect(content).not.toContain('100% real');
    expect(content).not.toContain('Calibrated Synthetic Probability');
    expect(content).not.toContain('Consistent with Natural Speech');
    expect(content).not.toContain('Authentic');

    // 8. Click Stop Stream & Disconnect
    await stopButton.click();
    await expect(page.locator('#recording-state-pill')).toContainText('stopping');
    await expect(page.locator('#connection-state-pill')).toContainText('connected');
  });
});

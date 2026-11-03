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
                  total_windows_generated: 1,
                  total_windows_dropped: 0,
                  status: 'success',
                  final_reason: 'Client stop requested.',
                  reliable_window_count: 1,
                  limited_reliability_window_count: 0,
                  insufficient_reliability_window_count: 0,
                  degraded_quality_window_count: 0,
                  quality_origin_verdict_allowed: true,
                  final_decision_state: 'high',
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
                  raw_score: 0.87,
                  calibrated_score: null,
                  operating_threshold: 0.65,
                  threshold_low: 0.35,
                  threshold_high: 0.65,
                  operating_threshold_source: 'unvalidated_heuristic',
                  input_reliability: {
                    state: 'reliable',
                    reasons: [],
                    explanation: 'Window meets all quality and speech criteria.',
                    origin_verdict_allowed: true,
                  },
                  quality_status: {
                    peak_level: 0.5,
                    rms: 0.15,
                    clipping_fraction: 0.0,
                    is_clipped: false,
                    is_silent: false,
                    snr_db_estimate: 24.5,
                    quality_status: 'clean',
                  },
                  abstain_reason: null,
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
                    window_start_sec: 0.0,
                    window_end_sec: 4.0,
                    speech_seconds: 3.8,
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
    await expect(timelineCard.locator('.timeline-score').nth(0)).toContainText('Raw: 0.870');
    await expect(timelineCard.locator('.timeline-score').nth(1)).toContainText('Smoothed: 0.850');
    await expect(timelineCard.locator('.badge-quality')).toContainText('CLEAN');

    // Verify Compact Reliability Summary
    await expect(page.locator('#stream-total-windows')).toHaveText('1');
    await expect(page.locator('#stream-reliable-windows')).toHaveText('1');
    await expect(page.locator('#stream-degraded-windows')).toHaveText('0');
    await expect(page.locator('#stream-verdict-permitted')).toContainText('Yes (Permitted)');

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

  const responsiveViewports = [
    { label: 'mobile', width: 360, height: 800 },
    { label: 'tablet', width: 768, height: 1024 },
    { label: 'desktop', width: 1440, height: 900 },
  ] as const;

  for (const viewport of responsiveViewports) {
    test(`prevents document overflow at ${viewport.width}x${viewport.height} in stream and file modes`, async ({
      page,
    }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');

      const streamTab = page.getByRole('tab', {
        name: 'Live Microphone Stream',
        exact: true,
      });
      const fileTab = page.getByRole('tab', {
        name: 'File Audio Analysis',
        exact: true,
      });

      await expect(streamTab).toBeVisible();
      await expect(fileTab).toBeVisible();
      await expect(
        page.getByRole('button', { name: 'File Audio Analysis', exact: true })
      ).toHaveCount(0);

      const modes = [
        { name: 'stream', tab: streamTab, panel: page.locator('#panel-stream') },
        { name: 'file', tab: fileTab, panel: page.locator('#panel-file') },
      ] as const;

      for (const mode of modes) {
        await mode.tab.click();
        await expect(mode.panel).toBeVisible();

        const layout = await page.evaluate(() => {
          const html = document.documentElement;
          const body = document.body;
          const container = document.querySelector<HTMLElement>('.app-container');
          const containerRect = container?.getBoundingClientRect();

          return {
            documentScrollWidth: Math.max(html.scrollWidth, body.scrollWidth),
            viewportWidth: html.clientWidth,
            htmlOverflowX: window.getComputedStyle(html).overflowX,
            bodyOverflowX: window.getComputedStyle(body).overflowX,
            containerOverflowX: container ? window.getComputedStyle(container).overflowX : '',
            containerLeft: containerRect?.left ?? 0,
            containerRight: containerRect?.right ?? 0,
            innerWidth: window.innerWidth,
          };
        });

        expect(layout.documentScrollWidth, `${viewport.label} ${mode.name} document overflow`).toBeLessThanOrEqual(
          layout.viewportWidth
        );
        expect(['hidden', 'clip'], `${viewport.label} ${mode.name} html overflow masking`).not.toContain(
          layout.htmlOverflowX
        );
        expect(['hidden', 'clip'], `${viewport.label} ${mode.name} body overflow masking`).not.toContain(
          layout.bodyOverflowX
        );
        expect(['hidden', 'clip'], `${viewport.label} ${mode.name} app overflow masking`).not.toContain(
          layout.containerOverflowX
        );
        expect(layout.containerLeft).toBeGreaterThanOrEqual(0);
        expect(layout.containerRight).toBeLessThanOrEqual(layout.innerWidth);
      }

      await streamTab.click();
      const startButton = page.locator('#start-mic-button');
      await expect(startButton).toBeVisible();
      const buttonBox = await startButton.boundingBox();
      expect(buttonBox).not.toBeNull();
      expect(buttonBox!.height).toBeGreaterThanOrEqual(44);

      await fileTab.click();
      const dropzone = page.locator('.file-dropzone');
      await expect(dropzone).toBeVisible();
      const dropzoneBox = await dropzone.boundingBox();
      expect(dropzoneBox).not.toBeNull();
      expect(dropzoneBox!.height).toBeGreaterThanOrEqual(44);
    });
  }

  test('contains wide window table inside horizontally scrollable .window-table-container without document overflow', async ({
    page,
  }) => {
    // Intercept analyze/file request with a valid multi-column file result
    await page.route('**/api/v1/analyze/file*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          file_path: 'e2e_table_test.wav',
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
          temporal_rule_explanation: 'Low anomaly score across windows.',
          is_mock_detector: false,
          config_hash: null,
          metadata: { client: 'voiceshield' },
          disclaimers: {
            identity_proof: 'Not proof of identity',
            fraud_probability: 'Not fraud probability',
            advisory_action: 'Advisory only',
            capture_boundary: 'Boundary',
          },
          windows: [
            {
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
              quality_status: {
                peak_level: 0.5,
                rms: 0.15,
                clipping_fraction: 0.0,
                is_clipped: false,
                is_silent: false,
                snr_db_estimate: 24.5,
                quality_status: 'clean',
              },
              input_reliability: {
                state: 'reliable',
                reasons: [],
                explanation: 'Window meets all criteria.',
                origin_verdict_allowed: true,
              },
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
            },
          ],
        }),
      });
    });

    await page.setViewportSize({ width: 360, height: 800 });
    await page.goto('/');

    // Switch to file tab
    await page.locator('#tab-file-btn').click();
    await expect(page.locator('#panel-file')).toBeVisible();

    // Select audio file
    await page.locator('#audio-file-input').setInputFiles({
      name: 'sample.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.from('RIFF....WAVEfmt ....data....'),
    });

    // Click Analyze button
    const submitBtn = page.locator('#submit-file-analysis-button');
    await expect(submitBtn).toBeEnabled();
    await submitBtn.click();

    // Wait for the window breakdown table container
    const tableContainer = page.locator('.window-table-container');
    await expect(tableContainer).toBeVisible();

    // Verify .window-table-container is horizontally scrollable when content overflows
    const tableMetrics = await page.evaluate(() => {
      const el = document.querySelector('.window-table-container');
      if (!el) return null;
      const style = window.getComputedStyle(el);
      return {
        tableScrollWidth: el.scrollWidth,
        tableClientWidth: el.clientWidth,
        containerOverflowX: style.overflowX,
        docScrollWidth: document.documentElement.scrollWidth,
        docClientWidth: document.documentElement.clientWidth,
      };
    });

    expect(tableMetrics).not.toBeNull();
    // Table is wider than container on 360px viewport:
    expect(tableMetrics!.tableScrollWidth).toBeGreaterThan(tableMetrics!.tableClientWidth);
    // Table container has scroll capability:
    expect(['auto', 'scroll']).toContain(tableMetrics!.containerOverflowX);
    // Document does NOT overflow:
    expect(tableMetrics!.docScrollWidth).toBeLessThanOrEqual(tableMetrics!.docClientWidth);
  });

  test('supports keyboard navigation across mode tabs using Arrow keys, Home, and End', async ({
    page,
  }) => {
    await page.goto('/');

    const streamTab = page.locator('#tab-stream-btn');
    const fileTab = page.locator('#tab-file-btn');

    // Focus stream tab
    await streamTab.focus();
    await expect(streamTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#panel-stream')).toBeVisible();

    // ArrowRight moves to file tab
    await page.keyboard.press('ArrowRight');
    await expect(fileTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#panel-file')).toBeVisible();

    // ArrowLeft moves back to stream tab
    await page.keyboard.press('ArrowLeft');
    await expect(streamTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#panel-stream')).toBeVisible();

    // End key moves to last tab (file)
    await page.keyboard.press('End');
    await expect(fileTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#panel-file')).toBeVisible();

    // Home key moves to first tab (stream)
    await page.keyboard.press('Home');
    await expect(streamTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#panel-stream')).toBeVisible();
  });
});

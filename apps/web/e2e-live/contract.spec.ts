import { test, expect, type Page } from '@playwright/test';

const missingPreviewUrl = 'http://127.0.0.1:5182';

async function installSyntheticMicrophone(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const destination = context.createMediaStreamDestination();
    oscillator.frequency.value = 440;
    oscillator.connect(destination);
    oscillator.start();
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (!constraints.audio) return originalGetUserMedia(constraints);
      const stream = destination.stream;
      for (const track of stream.getTracks()) {
        const originalStop = track.stop.bind(track);
        track.stop = () => {
          originalStop();
          oscillator.stop();
          void context.close();
        };
      }
      return stream;
    };
  });
}

function wavFixture(): Buffer {
  const sampleRate = 16000;
  const samples = sampleRate * 4;
  const data = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i += 1) {
    data.writeInt16LE(Math.round(Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 12000), i * 2);
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

function websocketFrameText(frame: unknown): string {
  const wrapped = frame as { payload?: unknown };
  const payload = wrapped.payload ?? frame;
  if (typeof payload === 'string') return payload;
  if (
    payload &&
    typeof payload === 'object' &&
    'type' in payload &&
    payload.type === 'Buffer' &&
    'data' in payload &&
    Array.isArray(payload.data)
  ) {
    return Buffer.from(payload.data).toString('utf8');
  }
  return '';
}

test.describe('VoiceShield live browser/API contract', () => {
  test('uses page readiness responses for mock and missing-checkpoint previews', async ({ page }) => {
    const mockReadyResponse = page.waitForResponse(
      (response) => response.url().endsWith('/api/v1/ready')
    );
    await page.goto('/');
    const mockResponse = await mockReadyResponse;
    expect(mockResponse.status()).toBe(200);
    const mockReady = await mockResponse.json();
    expect(mockReady.model_id).toBe('mock/aasist-l');
    expect(mockReady.provenance_status).toBe('mock');
    expect(mockReady.checkpoint_verified).toBe(false);
    expect(mockReady.config_hash).toMatch(/^[0-9a-f]{64}$/);
    await expect(page.locator('#mock-detector-badge')).toBeVisible();
    await expect(page.locator('#model-not-ready-banner')).toHaveCount(0);
    await expect(page.locator('#recording-state-pill')).toContainText('idle');

    const missingReadyResponse = page.waitForResponse(
      (response) => response.url().endsWith('/api/v1/ready')
    );
    await page.goto(missingPreviewUrl);
    const missingResponse = await missingReadyResponse;
    expect(missingResponse.status()).toBe(503);
    const missingBody = await missingResponse.json();
    expect(missingBody.detail.reason).toContain('checkpoint');
    await expect(page.locator('#model-not-ready-banner')).toBeVisible();
    await expect(page.locator('#model-not-ready-banner')).toContainText(
      missingBody.detail.reason
    );
    await expect(page.locator('#mock-detector-badge')).toHaveCount(0);
  });

  test('uploads a real WAV and renders the returned file contract', async ({ page }) => {
    const readyResponse = page.waitForResponse(
      (response) => response.url().endsWith('/api/v1/ready')
    );
    await page.goto('/');
    const ready = await (await readyResponse).json();
    await page.getByRole('button', { name: /File Audio Analysis/i }).click();
    await page.locator('#audio-file-input').setInputFiles({
      name: 'r9-fixture.wav',
      mimeType: 'audio/wav',
      buffer: wavFixture(),
    });
    await expect(page.locator('#selected-file-badge')).toBeVisible();
    const analyzeButton = page.getByRole('button', { name: /Analyze Audio File/i });
    await expect(analyzeButton).toBeEnabled();
    const [response] = await Promise.all([
      page.waitForResponse((candidate) => candidate.url().includes('/api/v1/analyze/file')),
      analyzeButton.click(),
    ]);
    expect(response.status()).toBe(200);
    const result = await response.json();
    expect(result.is_mock_detector).toBe(true);
    expect(result.score_is_calibrated).toBe(false);
    expect(result.config_hash).toBe(ready.config_hash);
    expect(result.metadata.is_mock_detector).toBe(true);
    expect(result.metadata.provenance_status).toBe('mock');
    expect(result.metadata.checkpoint_sha256).toBeNull();
    expect(result.disclaimers.advisory_action).toMatch(/advisory/i);
    expect(result.disclaimers.fraud_probability).toMatch(/not/i);
    expect(result.disclaimers.identity_proof).toMatch(/not/i);
    expect(result.disclaimers.capture_boundary).toMatch(/browser/i);
    await expect(page.locator('.file-results-card')).toBeVisible();
    await expect(page.locator('.threshold-provenance-banner')).toContainText(
      'Uncalibrated model output'
    );
    await expect(page.locator('.file-results-card')).toContainText('Advisory Verdict');
  });

  test('drives the real React microphone path through native proxied WebSocket traffic', async ({
    page,
  }) => {
    await installSyntheticMicrophone(page);
    const sockets: {
      url: string;
      sent: string[];
      received: string[];
      binary: number;
      closed: boolean;
      sequence: string[];
    }[] = [];
    page.on('websocket', (socket) => {
      if (!socket.url().endsWith('/api/v1/stream')) return;
      const observed = {
        url: socket.url(),
        sent: [],
        received: [],
        binary: 0,
        closed: false,
        sequence: [],
      };
      sockets.push(observed);
      socket.on('framesent', (payload) => {
        const text = websocketFrameText(payload);
        try {
          JSON.parse(text);
          observed.sent.push(text);
          observed.sequence.push(`sent:${JSON.parse(text).type}`);
        } catch {
          observed.binary += 1;
          observed.sequence.push('sent:binary');
        }
      });
      socket.on('framereceived', (payload) => {
        const text = websocketFrameText(payload);
        try {
          JSON.parse(text);
          observed.received.push(text);
          observed.sequence.push(`received:${JSON.parse(text).type}`);
        } catch {
          // The live contract only needs structured server events.
        }
      });
      socket.on('close', () => {
        observed.closed = true;
        observed.sequence.push('close');
      });
    });

    const readyResponse = page.waitForResponse(
      (response) => response.url().endsWith('/api/v1/ready')
    );
    await page.goto('/');
    const ready = await (await readyResponse).json();
    await page.getByRole('button', { name: /Start Microphone Stream/i }).click();

    await expect.poll(() => sockets.length, { timeout: 15000 }).toBe(1);
    const socket = sockets[0];
    expect(socket.url).toMatch(/^ws:\/\/127\.0\.0\.1:5181\/api\/v1\/stream$/);
    await expect.poll(
      () => socket.sent.filter((frame) => JSON.parse(frame).type === 'start').length,
      { timeout: 15000 }
    ).toBe(1);
    await expect(page.locator('#connection-state-pill')).toContainText('connected', {
      timeout: 15000,
    });
    await expect.poll(
      () => socket.received.some((frame) => JSON.parse(frame).type === 'session_started'),
      { timeout: 15000 }
    ).toBe(true);
    const sessionStarted = JSON.parse(
      socket.received.find((frame) => JSON.parse(frame).type === 'session_started') as string
    );
    expect(sessionStarted.is_mock_detector).toBe(true);
    expect(sessionStarted.config_hash).toBe(ready.config_hash);
    await expect.poll(() => socket.binary, { timeout: 30000 }).toBeGreaterThan(0);
    await expect.poll(
      () => socket.received.some((frame) => JSON.parse(frame).type === 'window_result'),
      { timeout: 30000 }
    ).toBe(true);
    await expect(page.locator('.timeline-card')).toHaveCount(1);
    await expect(page.locator('#status-heading')).not.toContainText('Calibrated');

    await page.getByRole('button', { name: /Stop Stream/i }).click();
    await expect.poll(
      () => socket.sent.filter((frame) => JSON.parse(frame).type === 'stop').length,
      { timeout: 5000 }
    ).toBe(1);
    await expect.poll(
      () => socket.received.some((frame) => JSON.parse(frame).type === 'session_stopped'),
      { timeout: 15000 }
    ).toBe(true);
    await expect.poll(() => socket.closed, { timeout: 15000 }).toBe(true);
    expect(socket.sequence.indexOf('received:session_stopped')).toBeLessThan(
      socket.sequence.indexOf('close')
    );
    await expect(page.locator('#recording-state-pill')).toContainText('idle');
    await expect(page.locator('#connection-state-pill')).toContainText('disconnected');
    await expect(page.getByRole('button', { name: /Start Microphone Stream/i })).toBeVisible();
    await expect(page.locator('#stream-error-alert')).toHaveCount(0);
  });
});

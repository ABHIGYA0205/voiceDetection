import { test, expect, type Page } from '@playwright/test';

const missingPreviewUrl = 'http://127.0.0.1:5182';

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

test.describe('VoiceShield live browser/API contract', () => {
  test('uses real proxied readiness and reports missing checkpoint without claiming ready', async ({
    page,
    request,
  }) => {
    const response = await request.get('/api/v1/ready');
    expect(response.status()).toBe(200);
    const ready = await response.json();
    expect(ready.model_id).toBe('mock/aasist-l');
    expect(ready.provenance_status).toBe('mock');
    expect(ready.checkpoint_verified).toBe(false);

    await page.goto('/');
    await expect(page.locator('#mock-detector-badge')).toBeVisible();
    await expect(page.locator('#model-not-ready-banner')).toHaveCount(0);
    await expect(page.locator('#connection-state-pill')).toContainText('disconnected');

    const missing = await page.request.get(`${missingPreviewUrl}/api/v1/ready`);
    expect(missing.status()).toBe(503);
    const missingBody = await missing.json();
    expect(missingBody.detail.reason).toContain('checkpoint');

    await page.goto(missingPreviewUrl);
    await expect(page.locator('#model-not-ready-banner')).toBeVisible();
    await expect(page.locator('#model-not-ready-banner')).toContainText(
      missingBody.detail.reason
    );
    await expect(page.locator('#mock-detector-badge')).toHaveCount(0);
  });

  test('uploads a real in-memory WAV through the proxied file contract', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: /File Audio Analysis/i }).click();
    await page.locator('#audio-file-input').setInputFiles({
      name: 'r9-fixture.wav',
      mimeType: 'audio/wav',
      buffer: wavFixture(),
    });
    await page.getByRole('button', { name: /Analyze Audio File/i }).click();

    await expect(page.locator('.file-results-card')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="mock-detector-badge"]')).toBeVisible();
    await expect(page.locator('.threshold-provenance-banner')).toContainText(
      'Uncalibrated model output'
    );
    await expect(page.locator('.file-results-card')).toContainText('Advisory Verdict');
  });

  test('streams through the native proxied WebSocket and drains on one stop command', async ({
    page,
  }) => {
    await page.goto('/');
    await page.evaluate(() => {
      const socket = new WebSocket(
        `${window.location.origin.replace(/^http/, 'ws')}/api/v1/stream`
      );
      const received: string[] = [];
      (window as unknown as { r9Socket: WebSocket; r9Received: string[] }).r9Socket = socket;
      (window as unknown as { r9Socket: WebSocket; r9Received: string[] }).r9Received =
        received;
      socket.addEventListener('message', (event) => {
        if (typeof event.data === 'string') received.push(event.data);
      });
      socket.addEventListener('open', () => {
        socket.send(
          JSON.stringify({
            type: 'start',
            protocol_version: '1.0',
            sample_rate: 16000,
            channels: 1,
            pcm_encoding: 'int16',
          })
        );
      });
    });
    await expect.poll(() => page.evaluate(() => {
      const state = (window as unknown as { r9Received?: string[] }).r9Received ?? [];
      return state.some((frame) => JSON.parse(frame).type === 'session_started');
    }), { timeout: 15000 }).toBe(true);
    await page.evaluate(() => {
      const socket = (window as unknown as { r9Socket: WebSocket }).r9Socket;
      const frame = new ArrayBuffer(32000);
      const samples = new Int16Array(frame);
      for (let index = 0; index < samples.length; index += 1) {
        samples[index] = Math.round(Math.sin((2 * Math.PI * 440 * index) / 16000) * 12000);
      }
      for (let index = 0; index < 8; index += 1) {
        socket.send(frame);
      }
    });
    await expect.poll(() => page.evaluate(() => {
      const state = (window as unknown as { r9Received?: string[] }).r9Received ?? [];
      return state.some((frame) => JSON.parse(frame).type === 'window_result');
    }), { timeout: 30000 }).toBe(true);
    await page.evaluate(() => {
      (window as unknown as { r9Socket: WebSocket }).r9Socket.send(
        JSON.stringify({ type: 'stop' })
      );
    });
    await expect.poll(
      () => page.evaluate(() => {
        const state = (window as unknown as { r9Received?: string[] }).r9Received ?? [];
        return state.filter((frame) => JSON.parse(frame).type === 'session_stopped').length;
      })
    ).toBe(1);
    await expect.poll(() => page.evaluate(() => {
      return (window as unknown as { r9Socket: WebSocket }).r9Socket.readyState;
    })).toBe(WebSocket.CLOSED);
  });
});

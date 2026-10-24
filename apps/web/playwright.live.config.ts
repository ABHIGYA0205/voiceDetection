import { defineConfig, devices } from '@playwright/test';

const mockApiPort = 8101;
const missingApiPort = 8102;
const mockPreviewPort = 5181;
const missingPreviewPort = 5182;

export default defineConfig({
  testDir: './e2e-live',
  timeout: 45000,
  fullyParallel: false,
  webServer: [
    {
      command:
        `cd ../.. && VOICESHIELD_MOCK_MODEL=true VOICESHIELD_DEVICE=cpu ` +
        `VOICESHIELD_CORS_ORIGINS=http://127.0.0.1:${mockPreviewPort} ` +
        `uv run --offline uvicorn services.api.main:app --host 127.0.0.1 --port ${mockApiPort}`,
      url: `http://127.0.0.1:${mockApiPort}/health`,
      reuseExistingServer: false,
      timeout: 60000,
    },
    {
      command:
        `cd ../.. && rm -rf /tmp/voiceshield-r9-missing && mkdir -p /tmp/voiceshield-r9-missing && ` +
        `trap 'rm -rf /tmp/voiceshield-r9-missing' EXIT && ` +
        `VOICESHIELD_DEVICE=cpu ` +
        `VOICESHIELD_CORS_ORIGINS=http://127.0.0.1:${missingPreviewPort} ` +
        `VOICESHIELD_CHECKPOINT_PATH=/tmp/voiceshield-r9-missing/AASIST-L.pth ` +
        `uv run --offline uvicorn services.api.main:app --host 127.0.0.1 --port ${missingApiPort}`,
      url: `http://127.0.0.1:${missingApiPort}/health`,
      reuseExistingServer: false,
      timeout: 60000,
    },
    {
      command:
        `VOICESHIELD_API_TARGET=http://127.0.0.1:${mockApiPort} ` +
        `npm run preview -- --host 127.0.0.1 --port ${mockPreviewPort} --strictPort`,
      url: `http://127.0.0.1:${mockPreviewPort}`,
      reuseExistingServer: false,
      timeout: 60000,
    },
    {
      command:
        `VOICESHIELD_API_TARGET=http://127.0.0.1:${missingApiPort} ` +
        `npm run preview -- --host 127.0.0.1 --port ${missingPreviewPort} --strictPort`,
      url: `http://127.0.0.1:${missingPreviewPort}`,
      reuseExistingServer: false,
      timeout: 60000,
    },
  ],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: `http://127.0.0.1:${mockPreviewPort}`,
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});

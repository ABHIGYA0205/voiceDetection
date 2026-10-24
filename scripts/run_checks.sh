#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo " VoiceShield Quality & Integrity Checks "
echo "========================================"

echo "[1/8] Running frozen dependency setup..."
uv sync --frozen --all-groups
cd apps/web
npm ci
cd ..

echo "[2/8] Running Ruff linter..."
uv run --frozen ruff check src services tests

echo "[3/8] Running Ruff format check..."
uv run --frozen ruff format --check src services tests

echo "[4/8] Running strict Mypy..."
uv run --frozen mypy src services tests

echo "[5/8] Running Python coverage suite..."
uv run --frozen pytest tests --cov=voiceshield --cov=services/api --cov-report=term-missing --cov-fail-under=88 -ra

echo "[6/8] Running vulnerability and license audits..."
uv export --frozen --all-extras --all-groups --no-emit-project --output-file "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/voiceshield-requirements.txt"
uv run --frozen pip-audit --disable-pip --require-hashes --requirement "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/voiceshield-requirements.txt"
cd apps/web
npm audit --audit-level=low
cd ..
uv run --frozen python scripts/check_licenses.py

echo "[7/8] Running package and frontend gates..."
rm -rf dist
uv build
uv run --frozen python scripts/check_package_metadata.py
cd apps/web
npm run build
npm test -- --run
npx playwright install chromium --with-deps
npm run test:e2e
npm run test:e2e:live
cd ..

echo "[8/8] All checks passed."

#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo " VoiceShield Quality & Integrity Checks "
echo "========================================"

echo "[1/4] Running Ruff linter..."
uv run --no-sync ruff check src services tests

echo "[2/4] Running Ruff format check..."
uv run --no-sync ruff format --check src services tests

echo "[3/4] Running Mypy static type checker..."
uv run --no-sync mypy src services tests

echo "[4/4] Running Pytest test suite..."
uv run --no-sync pytest tests

echo "All checks passed successfully!"

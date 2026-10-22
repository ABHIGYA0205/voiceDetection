.PHONY: help setup install lint format typecheck test test-python test-web api api-mock web benchmark benchmark-quick check-all clean

UV ?= $(shell which uv 2>/dev/null || echo "$$HOME/.local/bin/uv")
PYTHON ?= $(shell command -v python3.11 2>/dev/null || echo "/opt/homebrew/bin/python3.11")

help:
	@echo "VoiceShield Monorepo Commands:"
	@echo "  make setup       - Create .venv and install all dependencies (python + web)"
	@echo "  make lint        - Run ruff linter on Python codebase"
	@echo "  make format      - Run ruff code formatter"
	@echo "  make typecheck   - Run mypy type checker on Python codebase"
	@echo "  make test        - Run Python and Web test suites"
	@echo "  make test-python - Run pytest suite"
	@echo "  make test-web    - Run React/TypeScript test suite"
	@echo "  make api         - Start FastAPI dev server (real mode; requires weights/AASIST-L.pth)"
	@echo "  make api-mock    - Start FastAPI dev server with mock weights for test/UI demo"
	@echo "  make web         - Start React Vite frontend dev server"
	@echo "  make benchmark   - Run full M3 benchmark suite including 10-minute soak test"
	@echo "  make benchmark-quick - Run quick M3 benchmark suite (10s soak test)"
	@echo "  make check-all   - Run lint, format check, typecheck, and all tests"
	@echo "  make clean       - Remove caches, build artifacts, and test reports"

setup:
	$(UV) venv --python $(PYTHON) .venv
	$(UV) sync --all-groups
	@if [ -d "apps/web" ] && [ -f "apps/web/package.json" ]; then \
		echo "Installing web app dependencies..."; \
		cd apps/web && npm install; \
	fi

lint:
	$(UV) run ruff check src services tests

format:
	$(UV) run ruff format src services tests

typecheck:
	$(UV) run mypy src services tests

test-python:
	$(UV) run pytest tests

test-web:
	@if [ -d "apps/web" ] && [ -f "apps/web/package.json" ]; then \
		cd apps/web && npm test -- --run; \
	fi

test: test-python test-web

api:
	$(UV) run uvicorn services.api.main:app --reload --host 127.0.0.1 --port 8000

api-mock:
	VOICESHIELD_MOCK_MODEL=true $(UV) run uvicorn services.api.main:app --reload --host 127.0.0.1 --port 8000

web:
	@cd apps/web && npm run dev

benchmark:
	$(UV) run python -m voiceshield.benchmark.cli --soak-duration 600

benchmark-quick:
	$(UV) run python -m voiceshield.benchmark.cli --soak-duration 10 --iterations 20 --warmup 3

check-all: lint typecheck test-python
	@if [ -d "apps/web" ] && [ -f "apps/web/package.json" ]; then \
		cd apps/web && npm run build; \
	fi
	@echo "All checks passed successfully."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

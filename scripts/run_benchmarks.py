#!/usr/bin/env python3
"""Executable script to run the VoiceShield MacBook Pro M3 benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from voiceshield.benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    main()

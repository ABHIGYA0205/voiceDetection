#!/usr/bin/env python3
"""Validate package metadata and reject forbidden generated or data artifacts."""

from __future__ import annotations

import glob
import subprocess
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

REQUIRED = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "src/voiceshield/vendor/aasist/LICENSE",
    "src/voiceshield/vendor/aasist/NOTICE",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".bin",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".pyc",
    ".pyo",
}
FORBIDDEN_COMPONENTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".vite",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "test-results",
    "playwright-report",
    "coverage",
}
ROOT = Path(__file__).resolve().parents[1]


def forbidden_paths(paths: Iterable[str]) -> list[str]:
    offending: list[str] = []
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/").strip("/")
        components = {component.casefold() for component in normalized.split("/") if component}
        suffix = Path(normalized).suffix.casefold()
        if suffix in FORBIDDEN_SUFFIXES or components & FORBIDDEN_COMPONENTS:
            offending.append(normalized)
    return sorted(set(offending))


def fail_on_forbidden_paths(source: str, paths: Iterable[str]) -> int:
    offending = forbidden_paths(paths)
    if offending:
        details = "\n".join(f"  {path}" for path in offending)
        raise SystemExit(f"forbidden paths in {source}:\n{details}")
    return 0


def tracked_paths() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"git ls-files failed: {exc}") from exc
    return result.stdout.splitlines()


def main() -> int:
    wheels = glob.glob("dist/*.whl")
    sdists = glob.glob("dist/*.tar.gz")
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one wheel and one sdist in dist/")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        wheel_forbidden = fail_on_forbidden_paths("wheel", names)
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if not metadata_names:
            raise SystemExit("wheel does not contain metadata")
        metadata_name = metadata_names[0]
        metadata = archive.read(metadata_name).decode()
        if "License-Expression: Apache-2.0" not in metadata:
            raise SystemExit("wheel does not declare Apache-2.0")
        missing = [path for path in REQUIRED if not any(name.endswith(path) for name in names)]
        if missing:
            raise SystemExit(f"wheel missing required files: {missing}")
    with tarfile.open(sdists[0], "r:gz") as archive:
        names = set(archive.getnames())
        sdist_forbidden = fail_on_forbidden_paths("sdist", names)
        metadata_names = [name for name in names if name.endswith("PKG-INFO")]
        metadata_file = archive.extractfile(metadata_names[0]) if metadata_names else None
        if (
            metadata_file is None
            or "License-Expression: Apache-2.0" not in metadata_file.read().decode()
        ):
            raise SystemExit("sdist does not declare Apache-2.0")
        missing = [path for path in REQUIRED if not any(name.endswith(path) for name in names)]
        if missing:
            raise SystemExit(f"sdist missing required files: {missing}")
    tracked_forbidden = fail_on_forbidden_paths("tracked tree", tracked_paths())
    print(
        "package metadata, license files, and forbidden-artifact scans passed "
        f"(wheel={wheel_forbidden}, sdist={sdist_forbidden}, tracked={tracked_forbidden})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

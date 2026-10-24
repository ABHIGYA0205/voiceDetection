#!/usr/bin/env python3
"""Fail-closed SPDX checks for resolved Python and npm dependency graphs."""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from license_expression import get_spdx_licensing

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "Apache-2.0",
    "MIT",
    "MIT-CMU",
    "BSL-1.0",
    "LLVM-exception",
    "Apache-2.0 WITH LLVM-exception",
    "MIT-0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "0BSD",
    "PSF-2.0",
    "Python-2.0",
    "MPL-2.0",
    "CC0-1.0",
    "BlueOak-1.0.0",
    "Zlib",
    "Unlicense",
    "CC-BY-4.0",
}
EXCEPTIONS: dict[str, tuple[str, str]] = {}
licensing = get_spdx_licensing()


def check_expression(value: str, package: str) -> None:
    if not value or value.strip().upper() in {"UNKNOWN", "SEE LICENSE IN"}:
        raise ValueError(f"{package}: missing or unknown license metadata")
    try:
        expression = licensing.parse(value, validate=True)
    except Exception as exc:
        raise ValueError(f"{package}: malformed SPDX expression {value!r}") from exc
    identifiers = {str(symbol) for symbol in expression.symbols}
    if not identifiers or not identifiers.issubset(ALLOWED):
        raise ValueError(f"{package}: unreviewed license expression {value!r}")


def python_license(dist: metadata.Distribution) -> str:
    expression_values = dist.metadata.get_all("License-Expression") or []
    expression = expression_values[0].strip() if expression_values else ""
    if expression:
        return expression
    raw_values = dist.metadata.get_all("License") or []
    raw = raw_values[0].strip() if raw_values else ""
    if raw and raw.lower() not in {"unknown", "see license in"}:
        normalized = raw.lower()
        if raw in ALLOWED:
            return raw
        if (
            "bsd 3-clause" in normalized
            or "redistribution and use in source and binary forms" in normalized
        ):
            return "BSD-3-Clause"
        if "bsd 2-clause" in normalized:
            return "BSD-2-Clause"
        if "mit license" in normalized or "permission is hereby granted" in normalized:
            return "MIT"
        if "apache license" in normalized:
            return "Apache-2.0"
        if "mozilla public license" in normalized:
            return "MPL-2.0"
        if "isc license" in normalized:
            return "ISC"
    for classifier in dist.metadata.get_all("Classifier") or []:
        prefix = "License :: OSI Approved :: "
        if classifier.startswith(prefix):
            mapping = {
                "Apache Software License": "Apache-2.0",
                "MIT License": "MIT",
                "BSD License": "BSD-3-Clause",
                "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
                "Python Software Foundation License": "PSF-2.0",
            }
            if classifier.removeprefix(prefix) in mapping:
                return mapping[classifier.removeprefix(prefix)]
    raise ValueError(f"{dist.metadata['Name']}: missing Python license metadata")


def check_python() -> int:
    count = 0
    for dist in sorted(metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        if name.lower() == "voiceshield":
            continue
        check_expression(python_license(dist), name)
        count += 1
    return count


def npm_packages() -> dict[str, dict[str, Any]]:
    lock_path = ROOT / "apps/web/package-lock.json"
    lock = json.loads(lock_path.read_text())
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("npm lockfile does not contain a packages graph")
    return packages


def check_npm() -> int:
    packages = npm_packages()
    count = 0
    for path, locked in packages.items():
        if not path or not path.startswith("node_modules/") or not isinstance(locked, dict):
            continue
        package_json = ROOT / "apps/web" / path / "package.json"
        if not package_json.is_file():
            if locked.get("optional") is True:
                continue
            raise ValueError(f"{path}: installed package metadata is missing")
        package = json.loads(package_json.read_text())
        license_value = package.get("license")
        if not isinstance(license_value, str):
            raise ValueError(f"{path}: missing npm license metadata")
        check_expression(license_value, package.get("name", path))
        count += 1
    return count


def main() -> int:
    try:
        python_count = check_python()
        npm_count = check_npm()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"license check failed: {exc}", file=sys.stderr)
        return 1
    print(f"license check passed: {python_count} Python packages, {npm_count} npm packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

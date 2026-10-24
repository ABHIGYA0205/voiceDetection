#!/usr/bin/env python3
"""Fail-closed SPDX checks for resolved Python and npm dependency graphs."""

from __future__ import annotations

import hashlib
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
# Full-text metadata is accepted only for these exact package/version/fingerprint
# triples. The fingerprint prevents a later license-text change from being accepted.
LEGACY_TEXT_EXCEPTIONS: dict[tuple[str, str, str], str] = {
    (
        "contourpy",
        "1.3.3",
        "b4099bb6133b28b1daecac01c5151b4f5d8d7d894d0885c565c750a683b60f1b",
    ): "BSD-3-Clause",
    (
        "cycler",
        "0.12.1",
        "d242da8f1d13aeb786f303b6ce5001e961e95eadcfb58002254e208874c67ed0",
    ): "BSD-3-Clause",
    (
        "kiwisolver",
        "1.5.1",
        "b120c7f02d5be4da749577ee9d408927463c774ef701be4093d95703860f6abc",
    ): "BSD-3-Clause",
    (
        "matplotlib",
        "3.11.2",
        "fdc4dbfe6c39df1a8a2325363cb14cdf4d3b137fb7193fc61576d349bc50eda3",
    ): "BSD-3-Clause",
    (
        "numpy",
        "2.2.6",
        "38a5f83e366fe46ff60fe676252e6b8d545b5ed4f90da1aa56fec77903a2402a",
    ): "BSD-3-Clause",
    (
        "pip_api",
        "0.0.35",
        "0e7a64e3ff18df112dedf585be24ab5386bb6cb5c1b357882f239b1b55ef573e",
    ): "Apache-2.0",
    (
        "python-discovery",
        "1.6.4",
        "fe2a9817987f862eaced948f0468c7f51d2fedfc48c5c505b246a49a3870e9a5",
    ): "MIT",
    (
        "scipy",
        "1.17.1",
        "719257d46efd862de6569b7aba3ae59c39d20fefcc89a933646c65d686985334",
    ): "BSD-3-Clause",
}
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
        if raw in ALLOWED:
            return raw
        normalized = " ".join(raw.split())
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        exception = LEGACY_TEXT_EXCEPTIONS.get((dist.metadata["Name"], dist.version, fingerprint))
        if exception is not None:
            return exception
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
    raise ValueError(f"{dist.metadata['Name']}: missing or unreviewed Python license metadata")


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

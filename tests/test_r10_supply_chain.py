"""Focused R10 licensing, packaging, and workflow guard tests."""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import tomllib
import zipfile
from importlib import metadata
from pathlib import Path

import pytest

from scripts.check_licenses import check_expression, python_license
from scripts.check_package_metadata import forbidden_paths

ROOT = Path(__file__).parents[1]


def fake_distribution(tmp_path: Path, name: str, license_value: str) -> metadata.Distribution:
    dist_info = tmp_path / f"{name}-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\nLicense: {license_value}\n"
    )
    return metadata.PathDistribution(dist_info)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_aasist_license_and_notice_are_exact() -> None:
    assert sha256(ROOT / "src/voiceshield/vendor/aasist/LICENSE") == (
        "da2e79b8592d166ef505224300968b80ebe1e4c217c43b94a5ec627d81cd4142"
    )
    assert sha256(ROOT / "src/voiceshield/vendor/aasist/NOTICE") == (
        "70edb07f04ddc88e7155d6e826495e5ef99c3b1f0bd4e1b7fa35897b3854dbd0"
    )


def test_license_policy_accepts_spdx_boolean_expression() -> None:
    check_expression("MIT OR Apache-2.0", "reviewed-package")
    check_expression("MIT AND Apache-2.0", "reviewed-package")


def test_license_policy_accepts_exact_legacy_alias(tmp_path: Path) -> None:
    dist = fake_distribution(tmp_path, "legacy-package", "MIT")
    assert python_license(dist) == "MIT"


def test_license_policy_rejects_restricted_text_with_mit_phrase(tmp_path: Path) -> None:
    dist = fake_distribution(
        tmp_path,
        "restricted-package",
        "Permission is hereby granted, but commercial use is prohibited.",
    )
    with pytest.raises(ValueError):
        python_license(dist)


@pytest.mark.parametrize(
    "license_value",
    [
        "UNKNOWN",
        "SEE LICENSE IN foo",
        "GPL-3.0",
        "AGPL-3.0-only",
        "SSPL-1.0",
        "Commons-Clause",
        "LicenseRef-NonCommercial",
        "MIT +",
    ],
)
def test_license_policy_rejects_unknown_malformed_or_forbidden(license_value: str) -> None:
    with pytest.raises(ValueError):
        check_expression(license_value, "unreviewed-package")


def test_root_metadata_declares_apache_and_required_files() -> None:
    pyproject_text = (ROOT / "pyproject.toml").read_text()
    pyproject = tomllib.loads(pyproject_text)
    assert pyproject["project"]["license"] == "Apache-2.0"
    for path in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "src/voiceshield/vendor/aasist/LICENSE",
        "src/voiceshield/vendor/aasist/NOTICE",
    ):
        assert path in pyproject_text
    for dependencies in (
        pyproject["project"]["optional-dependencies"]["dev"],
        pyproject["dependency-groups"]["dev"],
    ):
        assert dependencies.count("pip-audit>=2.10.1,<3") == 1
        assert dependencies.count("license-expression>=30.4.4,<31") == 1


def test_dependabot_and_workflows_are_pinned_and_read_only() -> None:
    dependabot = (ROOT / ".github/dependabot.yml").read_text()
    for ecosystem in ("uv", "npm", "github-actions"):
        assert f"package-ecosystem: {ecosystem}" in dependabot
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        content = workflow.read_text()
        assert "pull_request_target" not in content
        assert "permissions:" in content
        assert re.search(r"permissions:\s+contents:\s+read", content)
        for permission, access in re.findall(
            r"^\s{2}([a-z-]+):\s+(read|write|none)\s*$", content, re.MULTILINE
        ):
            assert access != "write", f"{permission} permission must not allow writes"
        for line in content.splitlines():
            match = re.match(r"\s+uses:\s+([^@\s]+)@([0-9a-f]+)", line)
            if match:
                assert re.fullmatch(r"[0-9a-f]{40}", match.group(2))
            elif re.match(r"\s+uses:", line):
                pytest.fail(f"unparseable or unpinned action: {line}")
    dependency_review = (ROOT / ".github/workflows/dependency-review.yml").read_text()
    assert "fail-on-severity: moderate" in dependency_review
    assert "fail-on-scopes: runtime,development,unknown" in dependency_review
    assert "allow-licenses:" in dependency_review
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    for required in (
        "--cov-fail-under=88",
        "npm audit",
        "pip-audit",
        "npm run build",
        "npm test",
        "test:e2e",
        "test:e2e:live",
    ):
        assert required in ci
    assert 'cron: "47 3 * * 1"' in ci
    security = (ROOT / ".github/workflows/security.yml").read_text()
    assert 'cron: "17 3 * * 1"' in security
    assert "pull-requests: read" in security


def test_checkpoint_boundary_contains_pinned_source_facts() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "AASIST-L.pth" in notices
    assert "a04c9863f63d44471dde8a6abcb3b082b07cd1d1" in notices
    assert "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a" in notices
    assert "426,428 bytes" in notices


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (
            ["weights/AASIST-L.pth", "recordings/input.wav"],
            ["recordings/input.wav", "weights/AASIST-L.pth"],
        ),
        (
            [
                "apps/web/node_modules/pkg/index.js",
                "apps/web/dist/assets/app.js",
                "coverage/index.html",
            ],
            [
                "apps/web/dist/assets/app.js",
                "apps/web/node_modules/pkg/index.js",
                "coverage/index.html",
            ],
        ),
    ],
)
def test_forbidden_artifact_paths_are_deterministic(paths: list[str], expected: list[str]) -> None:
    assert forbidden_paths(paths) == expected


def test_forbidden_artifact_scan_allows_normal_package_paths() -> None:
    assert (
        forbidden_paths(
            [
                "voiceshield/__init__.py",
                "voiceshield-0.1.0.dist-info/METADATA",
                "src/voiceshield/vendor/aasist/LICENSE",
                "tests/fixtures/example.txt",
            ]
        )
        == []
    )


def test_archive_member_paths_are_scanned(tmp_path: Path) -> None:
    wheel = tmp_path / "sample.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sample-1.0.dist-info/METADATA", "License-Expression: Apache-2.0")
        archive.writestr("sample/model.pth", b"weights")
    with zipfile.ZipFile(wheel) as archive:
        assert forbidden_paths(archive.namelist()) == ["sample/model.pth"]

    sdist = tmp_path / "sample.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("sample-1.0/audio.wav")
        info.size = 5
        archive.addfile(info, io.BytesIO(b"audio"))
    with tarfile.open(sdist, "r:gz") as archive:
        assert forbidden_paths(archive.getnames()) == ["sample-1.0/audio.wav"]


def test_ci_requires_tracked_tree_mutation_guard() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "git diff --exit-code" in ci
    assert "git status --porcelain --untracked-files=no" in ci

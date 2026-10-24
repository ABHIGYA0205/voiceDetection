"""Focused R10 licensing, packaging, and workflow guard tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.check_licenses import check_expression

ROOT = Path(__file__).parents[1]


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


@pytest.mark.parametrize("license_value", ["UNKNOWN", "SEE LICENSE IN foo", "GPL-3.0", "MIT +"])
def test_license_policy_rejects_unknown_malformed_or_forbidden(license_value: str) -> None:
    with pytest.raises(ValueError):
        check_expression(license_value, "unreviewed-package")


def test_root_metadata_declares_apache_and_required_files() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'license = "Apache-2.0"' in pyproject
    for path in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "src/voiceshield/vendor/aasist/LICENSE",
        "src/voiceshield/vendor/aasist/NOTICE",
    ):
        assert path in pyproject


def test_dependabot_and_workflows_are_pinned_and_read_only() -> None:
    dependabot = (ROOT / ".github/dependabot.yml").read_text()
    for ecosystem in ("uv", "npm", "github-actions"):
        assert f"package-ecosystem: {ecosystem}" in dependabot
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        content = workflow.read_text()
        assert "pull_request_target" not in content
        assert "permissions:" in content
        for token in content.split():
            if "@${{" not in token and "uses:" not in token:
                continue
            if "uses:" in token or "@" in token:
                assert not token.rsplit("@", 1)[-1].startswith(("v", "main", "master"))

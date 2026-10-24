#!/usr/bin/env python3
"""Validate built wheel and sdist licensing metadata and required files."""

from __future__ import annotations

import glob
import tarfile
import zipfile

REQUIRED = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "src/voiceshield/vendor/aasist/LICENSE",
    "src/voiceshield/vendor/aasist/NOTICE",
}


def main() -> int:
    wheels = glob.glob("dist/*.whl")
    sdists = glob.glob("dist/*.tar.gz")
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one wheel and one sdist in dist/")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        if "License-Expression: Apache-2.0" not in metadata:
            raise SystemExit("wheel does not declare Apache-2.0")
        missing = [path for path in REQUIRED if not any(name.endswith(path) for name in names)]
        if missing:
            raise SystemExit(f"wheel missing required files: {missing}")
    with tarfile.open(sdists[0], "r:gz") as archive:
        names = set(archive.getnames())
        metadata_names = [name for name in names if name.endswith("PKG-INFO")]
        if (
            not metadata_names
            or "License-Expression: Apache-2.0"
            not in archive.extractfile(metadata_names[0]).read().decode()
        ):
            raise SystemExit("sdist does not declare Apache-2.0")
        missing = [path for path in REQUIRED if not any(name.endswith(path) for name in names)]
        if missing:
            raise SystemExit(f"sdist missing required files: {missing}")
    print("package metadata and license files passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

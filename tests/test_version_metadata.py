from __future__ import annotations

import re
import tomllib
from pathlib import Path

import facetroute


def test_release_version_metadata_is_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        package_version = tomllib.load(stream)["project"]["version"]
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\s]+)\s*$", citation, re.MULTILINE)

    assert match is not None
    assert match.group(1) == package_version
    assert facetroute.__version__ == package_version

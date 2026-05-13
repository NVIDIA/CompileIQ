"""Tests for dev/build_search_space_release_notes.py."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

from compileiq.search_spaces.manifest import SearchSpaceEntry, SearchSpaceManifestModel


_DEV = pathlib.Path(__file__).resolve().parents[3] / "dev"
_SPEC = importlib.util.spec_from_file_location(
    "build_search_space_release_notes",
    _DEV / "build_search_space_release_notes.py",
)
assert _SPEC is not None and _SPEC.loader is not None
release_notes = importlib.util.module_from_spec(_SPEC)
sys.modules["build_search_space_release_notes"] = release_notes
_SPEC.loader.exec_module(release_notes)


def _manifest() -> SearchSpaceManifestModel:
    return SearchSpaceManifestModel(
        tag="search-spaces-2026.05.08-rc1",
        generated_at="2026-05-08T00:00:00+00:00",
        entries=[
            SearchSpaceEntry(
                compiler="ptxas",
                compiler_version="13.3",
                variant="att",
                filename="ptxas13.3_att_search_space.bin",
                sha256="a" * 64,
                size_bytes=123,
                description="Attribute controls",
            )
        ],
    )


def test_build_notes_lists_catalog_metadata():
    notes = release_notes.build_notes(_manifest())

    assert "# Search spaces 2026.05.08-rc1" in notes
    assert "- Entries: 1" in notes
    assert "| ptxas | 13.3 | att | `ptxas13.3_att_search_space.bin` | 123 |" in notes
    assert "`" + ("a" * 64) + "`" in notes
    assert "CIQ_SEARCH_SPACES_DIR" in notes


def test_main_writes_release_notes(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    out = tmp_path / "release-notes.md"
    manifest_path.write_text(_manifest().model_dump_json())

    assert release_notes.main(["--manifest", str(manifest_path), "--out", str(out)]) == 0
    assert "Attribute controls" in out.read_text()

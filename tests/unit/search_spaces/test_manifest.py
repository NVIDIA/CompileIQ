"""Tests for compileiq.search_spaces.manifest."""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from compileiq.search_spaces.manifest import (
    SearchSpaceEntry,
    SearchSpaceManifestModel,
    search_space_manifest_json_schema,
)


SHA = "a" * 64


def _entry(**overrides) -> SearchSpaceEntry:
    base = {
        "compiler": "ptxas",
        "compiler_version": "13.3",
        "variant": "default",
        "filename": "ptxas13.3_search_space.bin",
        "sha256": SHA,
        "size_bytes": 1234,
    }
    base.update(overrides)
    return SearchSpaceEntry(**base)


def _manifest(entries: list[SearchSpaceEntry]) -> SearchSpaceManifestModel:
    return SearchSpaceManifestModel(
        tag="search-spaces-2026.04.27",
        generated_at="2026-04-27T00:00:00Z",
        entries=entries,
    )


def test_round_trip_serialization():
    m = _manifest([_entry(), _entry(variant="att", sha256="b" * 64)])
    payload = m.model_dump_json()
    restored = SearchSpaceManifestModel.model_validate_json(payload)
    assert restored == m
    assert json.loads(payload)["manifest_format"] == "1.0.0"


def test_find_exact_match():
    m = _manifest([_entry()])
    hit = m.find("ptxas", "13.3")
    assert hit.filename == "ptxas13.3_search_space.bin"


def test_find_distinguishes_variants():
    default = _entry()
    att = _entry(variant="att", sha256="b" * 64, filename="ptxas13.3_att_search_space.bin")
    m = _manifest([default, att])
    assert m.find("ptxas", "13.3", variant="att").variant == "att"
    assert m.find("ptxas", "13.3").variant == "default"


def test_rejects_duplicate_resolver_selector():
    with pytest.raises(ValidationError) as excinfo:
        _manifest([_entry(), _entry(filename="ptxas13.3_copy_search_space.bin")])

    assert "Duplicate manifest entry" in str(excinfo.value)


def test_find_raises_with_available_listing():
    m = _manifest(
        [_entry(), _entry(compiler="nvcc", filename="nvcc13.3_search_space.bin", sha256="b" * 64)]
    )
    with pytest.raises(LookupError) as ei:
        m.find("ptxas", "99.9")
    assert "ptxas/13.3/default" in str(ei.value)
    assert "nvcc/13.3/default" in str(ei.value)


def test_rejects_invalid_sha256():
    with pytest.raises(ValidationError):
        _entry(sha256="not-hex")


@pytest.mark.parametrize(
    "filename",
    [
        "../ptxas.bin",
        "/tmp/ptxas.bin",
        "nested/ptxas.bin",
        r"nested\ptxas.bin",
        r"C:\tmp\ptxas.bin",
        "bad name.bin",
    ],
)
def test_rejects_path_like_filenames(filename):
    with pytest.raises(ValidationError):
        _entry(filename=filename)


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        SearchSpaceEntry.model_validate(
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "filename": "x.bin",
                "sha256": SHA,
                "size_bytes": 1,
                "unknown_field": "boom",
            }
        )


def test_rejects_target_arch_selector_field():
    with pytest.raises(ValidationError):
        _entry(target_arch="sm_90a")


def test_rejects_zero_size():
    with pytest.raises(ValidationError):
        _entry(size_bytes=0)


def test_checked_in_json_schema_matches_pydantic_model():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    checked_in = json.loads(
        (repo_root / "schemas" / "search-space-manifest-v1.schema.json").read_text()
    )
    assert checked_in == search_space_manifest_json_schema()

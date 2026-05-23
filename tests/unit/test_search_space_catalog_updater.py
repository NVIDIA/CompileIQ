"""Tests for dev/update_search_space_catalog.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import yaml


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_SPEC = importlib.util.spec_from_file_location(
    "update_search_space_catalog",
    _DEV / "update_search_space_catalog.py",
)
assert _SPEC is not None and _SPEC.loader is not None
updater = importlib.util.module_from_spec(_SPEC)
sys.modules["update_search_space_catalog"] = updater
_SPEC.loader.exec_module(updater)

NEW_TAG = "search-spaces-2026.05.22"
NEW_GENERATED_AT = "2026-05-22T00:00:00Z"


def _write_source(input_dir: pathlib.Path, entries: list[dict[str, object]]) -> None:
    (input_dir / "manifest-source.yaml").write_text(
        yaml.safe_dump({"entries": entries}, sort_keys=False)
    )


def _write_prior(input_dir: pathlib.Path, entries: list[dict[str, object]]) -> None:
    (input_dir / ".search-space-manifest.prior-release.json").write_text(
        json.dumps(
            {
                "manifest_format": "1.0.0",
                "tag": "search-spaces-2026.05.21",
                "generated_at": "2026-05-21T00:00:00Z",
                "entries": entries,
            }
        )
    )


def _write_bin(input_dir: pathlib.Path, filename: str, payload: bytes = b"search-space") -> str:
    path = input_dir / filename
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _summary(
    *,
    kept: list[str],
    added: list[str],
    removed: list[str],
    replaced: list[str],
    unknown: dict[str, str],
    fixme_count: int,
):
    return updater.CatalogUpdateSummary(
        kept=kept,
        added=added,
        removed=removed,
        replaced=replaced,
        unknown=unknown,
        fixme_count=fixme_count,
        release_tag=NEW_TAG,
        generated_at=NEW_GENERATED_AT,
    )


def test_update_catalog_bootstraps_first_release_from_staged_bins(tmp_path):
    _write_prior(tmp_path, [])
    _write_source(
        tmp_path,
        [
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "ptxas13.3_search_space.bin",
            },
            {
                "compiler": "nvcc",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "nvcc13.3_search_space.bin",
            },
        ],
    )
    _write_bin(tmp_path, "ptxas13.3_search_space.bin")

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary == _summary(
        kept=[],
        added=["ptxas13.3_search_space.bin"],
        removed=[],
        replaced=[],
        unknown={},
        fixme_count=0,
    )
    updated = yaml.safe_load((tmp_path / "manifest-source.yaml").read_text())
    assert updated["entries"] == [
        {
            "compiler": "ptxas",
            "compiler_version": "13.3",
            "variant": "default",
            "filename": "ptxas13.3_search_space.bin",
        }
    ]


def test_update_catalog_reports_kept_replaced_and_removed_names(tmp_path):
    kept_sha = _write_bin(tmp_path, "ptxas13.3_search_space.bin", b"kept")
    replaced_sha = _write_bin(tmp_path, "nvcc13.3_search_space.bin", b"old")
    _write_bin(tmp_path, "nvcc13.3_search_space.bin", b"new")
    _write_prior(
        tmp_path,
        [
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "ptxas13.3_search_space.bin",
                "sha256": kept_sha,
                "size_bytes": 4,
            },
            {
                "compiler": "nvcc",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "nvcc13.3_search_space.bin",
                "sha256": replaced_sha,
                "size_bytes": 3,
            },
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "att",
                "filename": "ptxas13.3_att_search_space.bin",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
        ],
    )
    _write_source(
        tmp_path,
        [
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "ptxas13.3_search_space.bin",
            },
            {
                "compiler": "nvcc",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "nvcc13.3_search_space.bin",
            },
        ],
    )

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary == _summary(
        kept=["ptxas13.3_search_space.bin"],
        added=[],
        removed=["ptxas13.3_att_search_space.bin"],
        replaced=["nvcc13.3_search_space.bin"],
        unknown={},
        fixme_count=0,
    )


def test_update_catalog_reports_unknown_prior_hash(tmp_path):
    _write_bin(tmp_path, "ptxas13.3_search_space.bin", b"payload")
    _write_prior(
        tmp_path,
        [
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "ptxas13.3_search_space.bin",
            }
        ],
    )
    _write_source(
        tmp_path,
        [
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": "ptxas13.3_search_space.bin",
            }
        ],
    )

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary.unknown == {"ptxas13.3_search_space.bin": "no prior sha256"}


def test_update_catalog_infers_standard_filename_metadata(tmp_path):
    _write_prior(tmp_path, [])
    _write_source(tmp_path, [])
    _write_bin(tmp_path, "nvcc13.3_search_space.bin")

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary.fixme_count == 0
    updated = yaml.safe_load((tmp_path / "manifest-source.yaml").read_text())
    assert updated["entries"] == [
        {
            "compiler": "nvcc",
            "compiler_version": "13.3",
            "variant": "default",
            "filename": "nvcc13.3_search_space.bin",
        }
    ]


def test_update_catalog_marks_unparseable_new_filename_with_fixmes(tmp_path):
    _write_prior(tmp_path, [])
    _write_source(tmp_path, [])
    _write_bin(tmp_path, "mystery.bin")

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary.added == ["mystery.bin"]
    assert summary.fixme_count == 2
    updated = yaml.safe_load((tmp_path / "manifest-source.yaml").read_text())
    assert updated["entries"][0]["compiler"].startswith("FIXME:")
    assert updated["entries"][0]["compiler_version"].startswith("FIXME:")


def test_update_catalog_accepts_revision_suffix_tag(tmp_path):
    tag = "search-spaces-2026.05.22-rev1"
    _write_prior(tmp_path, [])
    _write_source(tmp_path, [])
    _write_bin(tmp_path, "ptxas13.3_att_search_space.bin")

    summary = updater.update_catalog(tmp_path, tag)

    assert summary.release_tag == tag
    assert summary.generated_at == NEW_GENERATED_AT

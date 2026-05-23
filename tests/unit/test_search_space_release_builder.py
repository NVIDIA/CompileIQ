"""Tests for dev/build_search_space_release.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import yaml


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_search_space_release",
    _DEV / "build_search_space_release.py",
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(_BUILDER_SPEC)
sys.modules["build_search_space_release"] = builder
_BUILDER_SPEC.loader.exec_module(builder)

_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_search_space_release",
    _DEV / "verify_search_space_release.py",
)
assert _VERIFIER_SPEC is not None and _VERIFIER_SPEC.loader is not None
verifier = importlib.util.module_from_spec(_VERIFIER_SPEC)
sys.modules["verify_search_space_release"] = verifier
_VERIFIER_SPEC.loader.exec_module(verifier)

TAG = "search-spaces-2026.05.22"


def _write_source_release(source_dir: pathlib.Path) -> None:
    source_dir.mkdir()
    (source_dir / "ptxas13.3_search_space.bin").write_bytes(b"ptxas-default")
    (source_dir / "nvcc13.3_search_space.bin").write_bytes(b"nvcc-default")
    (source_dir / ".search-space-manifest.prior-release.json").write_text(
        json.dumps(
            {
                "manifest_format": "1.0.0",
                "tag": "search-spaces-2026.05.21",
                "generated_at": "2026-05-21T00:00:00Z",
                "entries": [
                    {
                        "compiler": "ptxas",
                        "compiler_version": "13.3",
                        "variant": "default",
                        "filename": "ptxas13.3_search_space.bin",
                        "sha256": "0" * 64,
                        "size_bytes": 1,
                    }
                ],
            }
        )
    )
    (source_dir / "manifest-source.yaml").write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {
                        "compiler": "ptxas",
                        "compiler_version": "13.3",
                        "variant": "default",
                        "filename": "ptxas13.3_search_space.bin",
                        "description": "PTXAS default controls",
                    },
                    {
                        "compiler": "nvcc",
                        "compiler_version": "13.3",
                        "variant": "default",
                        "filename": "nvcc13.3_search_space.bin",
                    },
                ]
            },
            sort_keys=False,
        )
    )


def test_build_release_writes_manifest_assets_checksums_and_body(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_source_release(source_dir)

    assets = builder.build_release(
        source_dir,
        output_dir,
        TAG,
        None,
        builder.DEFAULT_DOCS_URL,
        clean_output=False,
    )

    assert [asset.name for asset in assets] == [
        "manifest.json",
        "ptxas13.3_search_space.bin",
        "nvcc13.3_search_space.bin",
        "SHA256SUMS.txt",
        "release-body.md",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["tag"] == TAG
    assert manifest["generated_at"] == "2026-05-22T00:00:00Z"
    assert {entry["filename"] for entry in manifest["entries"]} == {
        "ptxas13.3_search_space.bin",
        "nvcc13.3_search_space.bin",
    }
    body = (output_dir / "release-body.md").read_text()
    assert "## Search Space Catalog Release" in body
    assert (
        "This release contains the complete compiler search-space catalog as of 2026.05.22."
        in body
    )
    assert "## Changes" in body
    assert "### Added" in body
    assert "### Updated" in body
    assert verifier.validate_release_assets(
        output_dir,
        TAG,
        extra_ok={"release-body.md"},
        require_release_body=True,
    ) == []


def test_build_release_accepts_revision_suffix_tag(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_source_release(source_dir)

    builder.build_release(
        source_dir,
        output_dir,
        "search-spaces-2026.05.22-rev1",
        None,
        builder.DEFAULT_DOCS_URL,
        clean_output=False,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["tag"] == "search-spaces-2026.05.22-rev1"
    assert manifest["generated_at"] == "2026-05-22T00:00:00Z"


def test_build_release_rejects_extra_bin_not_in_manifest_source(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_source_release(source_dir)
    (source_dir / "extra.bin").write_bytes(b"extra")

    try:
        builder.build_release(
            source_dir,
            output_dir,
            TAG,
            None,
            builder.DEFAULT_DOCS_URL,
            clean_output=False,
        )
    except ValueError as exc:
        assert "unreferenced .bin assets" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_release_rejects_unresolved_fixme(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_source_release(source_dir)
    source = yaml.safe_load((source_dir / "manifest-source.yaml").read_text())
    source["entries"][0]["description"] = "FIXME: review validation"
    (source_dir / "manifest-source.yaml").write_text(yaml.safe_dump(source, sort_keys=False))

    try:
        builder.build_release(
            source_dir,
            output_dir,
            TAG,
            None,
            builder.DEFAULT_DOCS_URL,
            clean_output=False,
        )
    except ValueError as exc:
        assert "FIXME" in str(exc)
    else:
        raise AssertionError("expected ValueError")

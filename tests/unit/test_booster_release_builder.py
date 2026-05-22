"""Tests for dev/build_booster_pack_release.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import zipfile


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _REPO_ROOT / "dev"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_booster_pack_release",
    _DEV / "build_booster_pack_release.py",
)
assert _BUILDER_SPEC is not None
builder = importlib.util.module_from_spec(_BUILDER_SPEC)
sys.modules["build_booster_pack_release"] = builder
assert _BUILDER_SPEC.loader is not None
_BUILDER_SPEC.loader.exec_module(builder)

_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_booster_pack_release",
    _DEV / "verify_booster_pack_release.py",
)
assert _VERIFIER_SPEC is not None
verifier = importlib.util.module_from_spec(_VERIFIER_SPEC)
sys.modules["verify_booster_pack_release"] = verifier
assert _VERIFIER_SPEC.loader is not None
_VERIFIER_SPEC.loader.exec_module(verifier)


def _write_pack_zip(
    source_dir: pathlib.Path,
    artifact_name: str,
    display_name: str,
    pack_id: str,
    payload: bytes = b"candidate",
) -> str:
    manifest = {
        "schema_version": 1,
        "catalog_version": "2026.05.01",
        "release_tag": "booster-packs-2026.05.01",
        "display_name": display_name,
        "description": f"{display_name} candidates",
        "pack_id": pack_id,
        "pack_type": "diagnostic",
        "cuda_version": "13.3.0",
        "controls_stage": "ptxas",
        "supported_gpus": ["ALL"],
        "validation_summary": {"evidence": "smoke tested"},
        "acfs": [
            {
                "filename": "candidate.acf",
                "sha256": builder.sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        ],
    }
    pack_dir = artifact_name.removesuffix(".zip")
    with zipfile.ZipFile(source_dir / artifact_name, "w") as archive:
        archive.writestr(f"{pack_dir}/booster-pack-manifest.json", json.dumps(manifest))
        archive.writestr(f"{pack_dir}/candidate.acf", payload)
    return builder.sha256_file(source_dir / artifact_name)


def _write_source_release(source_dir: pathlib.Path) -> None:
    source_dir.mkdir()
    _write_pack_zip(source_dir, "booster-pack-debug.zip", "Debug Pack", "debug-pack")

    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.01",
        "generated_at": "2026-05-01T00:00:00Z",
        "release_tag": "booster-packs-2026.05.01",
        "packs": [
            {
                "artifact_name": "booster-pack-debug.zip",
                "manifest_path": "booster-pack-debug/booster-pack-manifest.json",
                "pack_id": "debug-pack",
            }
        ],
    }
    (source_dir / "booster-pack-catalog.json").write_text(json.dumps(catalog))


def test_build_release_generates_valid_stable_docs_bundle(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_source_release(source_dir)

    assets = builder.build_release(
        source_dir,
        output_dir,
        "booster-packs-2026.05.21",
        catalog_version=None,
        generated_at=None,
        docs_url=builder.DEFAULT_DOCS_URL,
        clean_output=False,
    )

    assert [asset.name for asset in assets] == [
        "booster-pack-catalog.json",
        "booster-pack-debug.zip",
        "SHA256SUMS.txt",
        "release-body.md",
    ]
    assert (
        verifier.validate_release_assets(
            output_dir,
            "booster-packs-2026.05.21",
            extra_ok=["release-body.md"],
        )
        == []
    )

    catalog = json.loads((output_dir / "booster-pack-catalog.json").read_text())
    assert catalog["catalog_version"] == "2026.05.21"
    assert catalog["generated_at"] == "2026-05-21T00:00:00Z"
    assert catalog["packs"][0]["artifact_name"] == "booster-pack-debug.zip"
    assert catalog["packs"][0]["acf_count"] == 1

    release_body = (output_dir / "release-body.md").read_text()
    assert "## Booster Pack Catalog Release" in release_body
    assert "complete Booster Pack catalog as of 2026.05.21" in release_body
    assert builder.DEFAULT_DOCS_URL in release_body
    assert "`booster-pack-debug.zip`" in release_body
    assert "## Changes" in release_body
    assert "### Added" in release_body
    assert "- Debug Pack (`booster-pack-debug.zip`)" in release_body

    with zipfile.ZipFile(output_dir / "booster-pack-debug.zip") as archive:
        manifest = json.loads(archive.read("booster-pack-debug/booster-pack-manifest.json"))
    assert manifest["release_tag"] == "booster-packs-2026.05.21"


def test_build_release_accepts_revision_suffix_tag(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    tag = "booster-packs-2026.05.21-rev1"
    _write_source_release(source_dir)

    builder.build_release(
        source_dir,
        output_dir,
        tag,
        catalog_version=None,
        generated_at=None,
        docs_url=builder.DEFAULT_DOCS_URL,
        clean_output=False,
    )

    assert (
        verifier.validate_release_assets(
            output_dir,
            tag,
            extra_ok=["release-body.md"],
        )
        == []
    )
    catalog = json.loads((output_dir / "booster-pack-catalog.json").read_text())
    assert catalog["release_tag"] == tag
    assert catalog["catalog_version"] == "2026.05.21"
    assert catalog["generated_at"] == "2026-05-21T00:00:00Z"
    assert "as of 2026.05.21" in (output_dir / "release-body.md").read_text()

    with zipfile.ZipFile(output_dir / "booster-pack-debug.zip") as archive:
        manifest = json.loads(archive.read("booster-pack-debug/booster-pack-manifest.json"))
    assert manifest["release_tag"] == tag
    assert manifest["catalog_version"] == "2026.05.21"


def test_build_release_body_changelog_uses_prior_catalog(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    _write_pack_zip(
        source_dir,
        "booster-pack-debug.zip",
        "Debug Pack",
        "debug-pack",
        payload=b"updated",
    )
    _write_pack_zip(source_dir, "booster-pack-new.zip", "New Pack", "new-pack")

    current_catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.01",
        "generated_at": "2026-05-01T00:00:00Z",
        "release_tag": "booster-packs-2026.05.01",
        "packs": [
            {"artifact_name": "booster-pack-debug.zip", "pack_id": "debug-pack"},
            {"artifact_name": "booster-pack-new.zip", "pack_id": "new-pack"},
        ],
    }
    prior_catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.01",
        "generated_at": "2026-05-01T00:00:00Z",
        "release_tag": "booster-packs-2026.05.01",
        "packs": [
            {
                "artifact_name": "booster-pack-debug.zip",
                "display_name": "Debug Pack",
                "pack_id": "debug-pack",
                "artifact_sha256": "0" * 64,
            },
            {
                "artifact_name": "booster-pack-old.zip",
                "display_name": "Old Pack",
                "pack_id": "old-pack",
                "artifact_sha256": "1" * 64,
            },
        ],
    }
    (source_dir / "booster-pack-catalog.json").write_text(json.dumps(current_catalog))
    (source_dir / ".booster-pack-catalog.prior-release.json").write_text(
        json.dumps(prior_catalog)
    )

    builder.build_release(
        source_dir,
        output_dir,
        "booster-packs-2026.05.21",
        catalog_version=None,
        generated_at=None,
        docs_url=builder.DEFAULT_DOCS_URL,
        clean_output=False,
    )

    release_body = (output_dir / "release-body.md").read_text()
    assert "## Changes" in release_body
    assert "### Added\n\n- New Pack (`booster-pack-new.zip`)" in release_body
    assert "### Updated\n\n- Debug Pack (`booster-pack-debug.zip`)" in release_body
    assert "### Removed\n\n- Old Pack (`booster-pack-old.zip`)" in release_body

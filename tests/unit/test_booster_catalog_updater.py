"""Tests for dev/update_booster_pack_catalog.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import zipfile
from collections.abc import Mapping


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_SPEC = importlib.util.spec_from_file_location(
    "update_booster_pack_catalog",
    _DEV / "update_booster_pack_catalog.py",
)
assert _SPEC is not None and _SPEC.loader is not None
updater = importlib.util.module_from_spec(_SPEC)
sys.modules["update_booster_pack_catalog"] = updater
_SPEC.loader.exec_module(updater)

NEW_TAG = "booster-packs-2026.05.22"
NEW_CATALOG_VERSION = "2026.05.22"
NEW_GENERATED_AT = "2026-05-22T00:00:00Z"


def _write_pack_zip(
    input_dir: pathlib.Path,
    artifact_name: str,
    manifest: Mapping[str, object],
) -> None:
    pack_dir = artifact_name.removesuffix(".zip")
    with zipfile.ZipFile(input_dir / artifact_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{pack_dir}/booster-pack-manifest.json", json.dumps(manifest))


def _write_catalog(input_dir: pathlib.Path, catalog: dict[str, object]) -> None:
    (input_dir / "booster-pack-catalog.json").write_text(json.dumps(catalog))
    (input_dir / ".booster-pack-catalog.prior-release.json").write_text(json.dumps(catalog))


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
        catalog_version=NEW_CATALOG_VERSION,
        generated_at=NEW_GENERATED_AT,
    )


def test_update_catalog_adds_new_zip_and_removes_missing_zip(tmp_path):
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [
            {
                "artifact_name": "booster-pack-old.zip",
                "pack_id": "old-pack",
                "artifact_sha256": "old-generated-value",
            }
        ],
    }
    _write_catalog(tmp_path, catalog)
    _write_pack_zip(
        tmp_path,
        "booster-pack-new.zip",
        {
            "display_name": "New Pack",
            "pack_id": "new-pack",
            "pack_type": "performance",
            "description": "New compiler controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
        },
    )

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary == _summary(
        kept=[],
        added=["booster-pack-new.zip"],
        removed=["booster-pack-old.zip"],
        replaced=[],
        unknown={},
        fixme_count=1,
    )
    updated = json.loads((tmp_path / "booster-pack-catalog.json").read_text())
    assert updated["release_tag"] == NEW_TAG
    assert updated["catalog_version"] == NEW_CATALOG_VERSION
    assert updated["generated_at"] == NEW_GENERATED_AT
    assert updated["packs"] == [
        {
            "artifact_name": "booster-pack-new.zip",
            "manifest_path": "booster-pack-new/booster-pack-manifest.json",
            "display_name": "New Pack",
            "pack_id": "new-pack",
            "pack_type": "performance",
            "description": "New compiler controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
            "validation_summary": {
                "evidence": "FIXME: review validation evidence for booster-pack-new.zip"
            },
        }
    ]


def test_update_catalog_accepts_revision_suffix_tag(tmp_path):
    tag = "booster-packs-2026.05.22-rev1"
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [],
    }
    _write_catalog(tmp_path, catalog)
    _write_pack_zip(
        tmp_path,
        "booster-pack-revision.zip",
        {
            "display_name": "Revision Pack",
            "pack_id": "revision-pack",
            "pack_type": "performance",
            "description": "Revision compiler controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
            "validation_summary": {"evidence": "smoke tested"},
        },
    )

    summary = updater.update_catalog(tmp_path, tag)

    assert summary.release_tag == tag
    assert summary.catalog_version == NEW_CATALOG_VERSION
    assert summary.generated_at == NEW_GENERATED_AT
    updated = json.loads((tmp_path / "booster-pack-catalog.json").read_text())
    assert updated["release_tag"] == tag
    assert updated["catalog_version"] == NEW_CATALOG_VERSION
    assert updated["generated_at"] == NEW_GENERATED_AT


def test_update_catalog_reports_kept_and_replaced_zip_names(tmp_path):
    existing_manifest = {
        "display_name": "Debug Pack",
        "pack_id": "debug-pack",
        "pack_type": "diagnostic",
        "description": "Old controls",
        "cuda_version": "13.3.0",
        "controls_stage": "ptxas",
        "supported_gpus": ["B200"],
    }
    _write_pack_zip(tmp_path, "booster-pack-debug.zip", existing_manifest)
    original_sha = updater._sha256_file(tmp_path / "booster-pack-debug.zip")

    _write_pack_zip(
        tmp_path,
        "booster-pack-helion.zip",
        {
            "display_name": "Helion Pack",
            "pack_id": "helion-pack",
            "pack_type": "performance",
            "description": "Helion controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
        },
    )

    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [
            {
                "artifact_name": "booster-pack-debug.zip",
                "pack_id": "debug-pack",
                "artifact_sha256": original_sha,
            },
            {
                "artifact_name": "booster-pack-helion.zip",
                "pack_id": "helion-pack",
                "artifact_sha256": updater._sha256_file(tmp_path / "booster-pack-helion.zip"),
            },
        ],
    }
    _write_catalog(tmp_path, catalog)

    _write_pack_zip(
        tmp_path,
        "booster-pack-debug.zip",
        existing_manifest | {"description": "Replacement controls"},
    )

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary == _summary(
        kept=["booster-pack-helion.zip"],
        added=[],
        removed=[],
        replaced=["booster-pack-debug.zip"],
        unknown={},
        fixme_count=2,
    )

    rerun_summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert rerun_summary == summary


def test_main_prints_artifact_names_by_action(tmp_path, capsys):
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [{"artifact_name": "booster-pack-old.zip", "pack_id": "old-pack"}],
    }
    _write_catalog(tmp_path, catalog)
    _write_pack_zip(
        tmp_path,
        "booster-pack-new.zip",
        {
            "display_name": "New Pack",
            "pack_id": "new-pack",
            "pack_type": "performance",
            "description": "New compiler controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
        },
    )

    assert updater.main([str(tmp_path), "--tag", NEW_TAG]) == 0

    output = capsys.readouterr().out
    assert f"release_tag: {NEW_TAG}" in output
    assert f"catalog_version: {NEW_CATALOG_VERSION}" in output
    assert f"generated_at: {NEW_GENERATED_AT}" in output
    assert "kept (0):\n  - none" in output
    assert "added (1):\n  - booster-pack-new.zip" in output
    assert "removed (1):\n  - booster-pack-old.zip" in output
    assert "replaced (0):\n  - none" in output
    assert "unknown (0):\n  - none" in output
    assert "FIXME markers: 1" in output
    assert "Resolve every FIXME marker before building the release assets." in output


def test_update_catalog_reports_unknown_when_prior_sha_is_missing(tmp_path):
    _write_pack_zip(
        tmp_path,
        "booster-pack-debug.zip",
        {
            "display_name": "Debug Pack",
            "pack_id": "debug-pack",
            "pack_type": "diagnostic",
            "description": "Debug controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
        },
    )
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [{"artifact_name": "booster-pack-debug.zip", "pack_id": "debug-pack"}],
    }
    _write_catalog(tmp_path, catalog)

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary == _summary(
        kept=[],
        added=[],
        removed=[],
        replaced=[],
        unknown={"booster-pack-debug.zip": "no prior artifact_sha256"},
        fixme_count=1,
    )


def test_main_warns_when_pack_comparison_is_unknown(tmp_path, capsys):
    _write_pack_zip(
        tmp_path,
        "booster-pack-debug.zip",
        {
            "display_name": "Debug Pack",
            "pack_id": "debug-pack",
            "pack_type": "diagnostic",
            "description": "Debug controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
        },
    )
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [{"artifact_name": "booster-pack-debug.zip", "pack_id": "debug-pack"}],
    }
    _write_catalog(tmp_path, catalog)

    assert updater.main([str(tmp_path), "--tag", NEW_TAG]) == 0

    output = capsys.readouterr().out
    assert "unknown (1):\n  - booster-pack-debug.zip: no prior artifact_sha256" in output
    assert "WARNING: resolve unknown pack comparisons before building release assets." in output


def test_update_catalog_uses_recovery_catalog_as_prior_release(tmp_path):
    manifest = {
        "display_name": "Debug Pack",
        "pack_id": "debug-pack",
        "pack_type": "diagnostic",
        "description": "Debug controls",
        "cuda_version": "13.3.0",
        "controls_stage": "ptxas",
        "supported_gpus": ["B200"],
    }
    _write_pack_zip(tmp_path, "booster-pack-debug.zip", manifest)

    backup_catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [
            {
                "artifact_name": "booster-pack-debug.zip",
                "pack_id": "debug-pack",
                "artifact_sha256": updater._sha256_file(tmp_path / "booster-pack-debug.zip"),
            }
        ],
    }
    current_catalog = {
        **backup_catalog,
        "packs": [{"artifact_name": "booster-pack-debug.zip", "pack_id": "debug-pack"}],
    }
    (tmp_path / "booster-pack-catalog.backup.json").write_text(json.dumps(backup_catalog))
    (tmp_path / "booster-pack-catalog.json").write_text(json.dumps(current_catalog))

    summary = updater.update_catalog(tmp_path, NEW_TAG)

    assert summary == _summary(
        kept=["booster-pack-debug.zip"],
        added=[],
        removed=[],
        replaced=[],
        unknown={},
        fixme_count=1,
    )
    assert (tmp_path / ".booster-pack-catalog.prior-release.json").is_file()


def test_update_catalog_requires_prior_release_catalog(tmp_path, capsys):
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [],
    }
    (tmp_path / "booster-pack-catalog.json").write_text(json.dumps(catalog))
    _write_pack_zip(
        tmp_path,
        "booster-pack-debug.zip",
        {
            "display_name": "Debug Pack",
            "pack_id": "debug-pack",
            "pack_type": "diagnostic",
            "description": "Debug controls",
            "cuda_version": "13.3.0",
            "controls_stage": "ptxas",
            "supported_gpus": ["B200"],
        },
    )

    assert updater.main([str(tmp_path), "--tag", NEW_TAG]) == 1

    output = capsys.readouterr()
    assert "missing .booster-pack-catalog.prior-release.json" in output.err
    assert "Run make setup-booster-pack-release before updating the catalog." in output.err


def test_main_reports_zero_fixmes(tmp_path, capsys):
    manifest = {
        "display_name": "Debug Pack",
        "pack_id": "debug-pack",
        "pack_type": "diagnostic",
        "description": "Debug controls",
        "cuda_version": "13.3.0",
        "controls_stage": "ptxas",
        "supported_gpus": ["B200"],
        "validation_summary": {"status": "passed", "evidence": "smoke tested"},
    }
    _write_pack_zip(tmp_path, "booster-pack-debug.zip", manifest)
    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "generated_at": "2026-05-21T00:00:00Z",
        "release_tag": "booster-packs-2026.05.21",
        "packs": [
            {
                "artifact_name": "booster-pack-debug.zip",
                "pack_id": "debug-pack",
                "artifact_sha256": updater._sha256_file(tmp_path / "booster-pack-debug.zip"),
            }
        ],
    }
    _write_catalog(tmp_path, catalog)

    assert updater.main([str(tmp_path), "--tag", NEW_TAG]) == 0

    output = capsys.readouterr().out
    assert "FIXME markers: 0" in output
    assert "No FIXME markers found in the updated catalog." in output

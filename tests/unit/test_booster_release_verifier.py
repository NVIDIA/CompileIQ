"""Tests for dev/verify_booster_pack_release.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import zipfile


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_SPEC = importlib.util.spec_from_file_location(
    "verify_booster_pack_release",
    _DEV / "verify_booster_pack_release.py",
)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules["verify_booster_pack_release"] = verifier
_SPEC.loader.exec_module(verifier)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _write_valid_release(asset_dir: pathlib.Path, tag: str) -> None:
    acf_payload = b"controls"
    acf = {
        "filename": "candidate.acf",
        "sha256": verifier.sha256_bytes(acf_payload),
        "size_bytes": len(acf_payload),
    }
    manifest = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "release_tag": tag,
        "pack_id": "debug-pack",
        "pack_type": "diagnostic",
        "display_name": "Debug Pack",
        "acfs": [acf],
    }
    manifest_payload = _json_bytes(manifest)

    zip_path = asset_dir / "booster-pack-debug.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("debug-pack/booster-pack-manifest.json", manifest_payload)
        archive.writestr("debug-pack/candidate.acf", acf_payload)

    catalog = {
        "schema_version": 1,
        "catalog_version": "2026.05.21",
        "release_tag": tag,
        "packs": [
            {
                "pack_id": "debug-pack",
                "pack_type": "diagnostic",
                "display_name": "Debug Pack",
                "acf_count": 1,
                "acfs": [acf],
                "artifact_name": zip_path.name,
                "artifact_sha256": verifier.sha256_file(zip_path),
                "artifact_size_bytes": zip_path.stat().st_size,
                "manifest_path": "debug-pack/booster-pack-manifest.json",
                "manifest_sha256": verifier.sha256_bytes(manifest_payload),
            }
        ],
    }
    catalog_path = asset_dir / "booster-pack-catalog.json"
    catalog_path.write_bytes(_json_bytes(catalog))

    checksums = [
        f"{verifier.sha256_file(catalog_path)}  {catalog_path.name}",
        f"{verifier.sha256_file(zip_path)}  {zip_path.name}",
    ]
    (asset_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")


def test_validate_release_assets_accepts_valid_bundle(tmp_path):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)

    assert verifier.validate_release_assets(tmp_path, tag) == []


def test_main_prints_pass_on_valid_bundle(tmp_path, capsys):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)

    assert verifier.main([str(tmp_path), "--tag", tag]) == 0

    output = capsys.readouterr()
    assert f"PASS: Validated Booster Pack release assets for {tag}" in output.out
    assert output.err == ""


def test_main_prints_fail_on_invalid_bundle(tmp_path, capsys):
    tag = "booster-packs-2026.05.21"
    tmp_path.mkdir(exist_ok=True)

    assert verifier.main([str(tmp_path), "--tag", tag]) == 1

    output = capsys.readouterr()
    assert f"FAIL: Booster Pack release validation failed for {tag}." in output.err
    assert "ERROR: missing required release asset SHA256SUMS.txt" in output.err


def test_validate_release_assets_allows_explicit_local_extra(tmp_path):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)
    (tmp_path / "release-body.md").write_text(
        "\n".join(
            [
                verifier.DEFAULT_DOCS_URL,
                "booster-pack-catalog.json",
                "booster-pack-debug.zip",
                "SHA256SUMS.txt",
            ]
        )
    )

    errors = verifier.validate_release_assets(tmp_path, tag)
    assert "release-body.md" in "\n".join(errors)

    assert verifier.validate_release_assets(tmp_path, tag, extra_ok=["release-body.md"]) == []


def test_validate_release_assets_rejects_incomplete_release_body(tmp_path):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)
    (tmp_path / "release-body.md").write_text("release notes\n")

    errors = verifier.validate_release_assets(tmp_path, tag, extra_ok=["release-body.md"])

    assert (
        "release-body.md: missing public docs URL "
        "https://nvidia.github.io/CompileIQ/stable/booster_packs.html"
    ) in errors
    assert "release-body.md: missing asset name booster-pack-debug.zip" in errors


def test_validate_release_assets_can_require_release_body(tmp_path):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)

    errors = verifier.validate_release_assets(tmp_path, tag, require_release_body=True)

    assert "missing required local staging file release-body.md" in errors


def test_validate_release_assets_rejects_missing_checksum_asset(tmp_path):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)
    with (tmp_path / "SHA256SUMS.txt").open("a") as f:
        f.write(f"{'a' * 64}  RELEASE_NOTES.md\n")

    errors = verifier.validate_release_assets(tmp_path, tag)

    assert "SHA256SUMS.txt references missing assets: RELEASE_NOTES.md" in errors


def test_validate_release_assets_rejects_unresolved_fixme(tmp_path):
    tag = "booster-packs-2026.05.21"
    _write_valid_release(tmp_path, tag)
    catalog_path = tmp_path / "booster-pack-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["packs"][0]["validation_summary"] = {"evidence": "FIXME: add validation evidence"}
    catalog_path.write_bytes(_json_bytes(catalog))

    checksums = [
        f"{verifier.sha256_file(catalog_path)}  {catalog_path.name}",
        f"{verifier.sha256_file(tmp_path / 'booster-pack-debug.zip')}  booster-pack-debug.zip",
    ]
    (tmp_path / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")

    errors = verifier.validate_release_assets(tmp_path, tag)

    assert (
        "booster-pack-catalog.json.packs[0].validation_summary.evidence: "
        "contains unresolved FIXME"
    ) in errors

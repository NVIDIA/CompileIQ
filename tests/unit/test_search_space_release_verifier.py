"""Tests for dev/verify_search_space_release.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_SPEC = importlib.util.spec_from_file_location(
    "verify_search_space_release",
    _DEV / "verify_search_space_release.py",
)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules["verify_search_space_release"] = verifier
_SPEC.loader.exec_module(verifier)

TAG = "search-spaces-2026.05.22"
BIN_NAME = "ptxas13.3_search_space.bin"
BIN_BYTES = b"ptxas-search-space"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _write_valid_release(asset_dir: pathlib.Path) -> None:
    bin_path = asset_dir / BIN_NAME
    bin_path.write_bytes(BIN_BYTES)
    manifest = {
        "manifest_format": "1.0.0",
        "tag": TAG,
        "generated_at": "2026-05-22T00:00:00Z",
        "entries": [
            {
                "compiler": "ptxas",
                "compiler_version": "13.3",
                "variant": "default",
                "filename": BIN_NAME,
                "sha256": _sha256(BIN_BYTES),
                "size_bytes": len(BIN_BYTES),
                "search_space_format": "1.0.0",
                "description": "PTXAS default controls",
            }
        ],
    }
    manifest_path = asset_dir / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    checksums = [
        f"{verifier.sha256_file(manifest_path)}  manifest.json",
        f"{verifier.sha256_file(bin_path)}  {BIN_NAME}",
    ]
    (asset_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    (asset_dir / "release-body.md").write_text(
        "\n".join(
            [
                "## Search Space Catalog Release",
                "",
                f"Documentation: {verifier.DEFAULT_DOCS_URL}",
                "",
                "- `manifest.json`",
                f"- `{BIN_NAME}`",
                "- `SHA256SUMS.txt`",
                "",
            ]
        )
    )


def _manifest(asset_dir: pathlib.Path) -> dict[str, Any]:
    return json.loads((asset_dir / "manifest.json").read_text())


def _write_manifest(asset_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    (asset_dir / "manifest.json").write_bytes(_json_bytes(manifest))


def test_validate_release_assets_accepts_valid_staged_release(tmp_path):
    _write_valid_release(tmp_path)

    assert verifier.validate_release_assets(
        tmp_path,
        TAG,
        extra_ok={"release-body.md"},
        require_release_body=True,
    ) == []


def test_validate_release_assets_accepts_downloaded_assets_without_release_body(tmp_path):
    _write_valid_release(tmp_path)
    (tmp_path / "release-body.md").unlink()

    assert verifier.validate_release_assets(tmp_path, TAG) == []


def test_validate_release_assets_rejects_missing_bin(tmp_path):
    _write_valid_release(tmp_path)
    (tmp_path / BIN_NAME).unlink()

    errors = verifier.validate_release_assets(tmp_path, TAG, extra_ok={"release-body.md"})

    assert any("missing .bin assets" in error for error in errors)


def test_validate_release_assets_rejects_extra_bin(tmp_path):
    _write_valid_release(tmp_path)
    (tmp_path / "extra.bin").write_bytes(b"extra")

    errors = verifier.validate_release_assets(tmp_path, TAG, extra_ok={"release-body.md"})

    assert any("not listed in SHA256SUMS.txt" in error for error in errors)
    assert any("not in manifest.json" in error for error in errors)


def test_validate_release_assets_rejects_tag_mismatch(tmp_path):
    _write_valid_release(tmp_path)

    errors = verifier.validate_release_assets(tmp_path, "search-spaces-2026.05.23")

    assert any("tag is" in error for error in errors)


def test_validate_release_assets_rejects_size_mismatch(tmp_path):
    _write_valid_release(tmp_path)
    manifest = _manifest(tmp_path)
    manifest["entries"][0]["size_bytes"] += 1
    _write_manifest(tmp_path, manifest)

    errors = verifier.validate_release_assets(tmp_path, TAG, extra_ok={"release-body.md"})

    assert any("size_bytes mismatch" in error for error in errors)


def test_validate_release_assets_rejects_sha_mismatch(tmp_path):
    _write_valid_release(tmp_path)
    (tmp_path / BIN_NAME).write_bytes(b"changed")

    errors = verifier.validate_release_assets(tmp_path, TAG, extra_ok={"release-body.md"})

    assert any("SHA256 mismatch" in error for error in errors)
    assert any("sha256 mismatch" in error for error in errors)


def test_validate_release_assets_rejects_missing_release_body_docs_url(tmp_path):
    _write_valid_release(tmp_path)
    (tmp_path / "release-body.md").write_text("## Search Space Catalog Release\n")

    errors = verifier.validate_release_assets(
        tmp_path,
        TAG,
        extra_ok={"release-body.md"},
        require_release_body=True,
    )

    assert any("missing public docs URL" in error for error in errors)


def test_validate_release_assets_rejects_unresolved_fixme(tmp_path):
    _write_valid_release(tmp_path)
    manifest = _manifest(tmp_path)
    manifest["entries"][0]["description"] = "FIXME: review this"
    _write_manifest(tmp_path, manifest)

    errors = verifier.validate_release_assets(tmp_path, TAG, extra_ok={"release-body.md"})

    assert any("contains unresolved FIXME" in error for error in errors)

from __future__ import annotations

import json
from pathlib import Path

from xla_ciq.cli import main
from xla_ciq.bundle import EntryMetadata, append_acf, new_bundle, save_bundle
from xla_ciq.fingerprint import content_id


def test_bundle_inspect_text(tmp_path: Path, capsys):
    path = tmp_path / "t.bundle"
    bundle = new_bundle()
    acf = b"acf-bytes"
    cid = content_id(acf)
    append_acf(
        bundle,
        fingerprint="fp1",
        acf_bytes=acf,
        content_id=cid,
        metadata=EntryMetadata(
            kernel_name="wrapped_xor",
            shape="f32[4]",
            baseline_ms=0.02,
            best_ms=0.01,
            speedup=2.0,
        ),
    )
    save_bundle(path, bundle)

    assert main(["bundle", "inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "format_version: 1" in out
    assert "mappings: 1" in out
    assert "fingerprint=fp1" in out
    assert "kernel=wrapped_xor" in out
    assert "speedup=2.0000x" in out
    assert cid in out


def test_bundle_inspect_json(tmp_path: Path, capsys):
    path = tmp_path / "t.bundle"
    bundle = new_bundle()
    acf = b"acf-bytes"
    cid = content_id(acf)
    append_acf(
        bundle,
        fingerprint="fp1",
        acf_bytes=acf,
        content_id=cid,
        metadata=EntryMetadata(
            kernel_name="wrapped_xor",
            shape="f32[4,4096,16]",
            arch="sm_90a",
            cuda_version="13.3",
            baseline_ms=0.02,
            best_ms=0.01,
            speedup=2.0,
        ),
    )
    save_bundle(path, bundle)

    assert main(["bundle", "inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    entry = payload["entries"][0]
    assert entry["content_id"] == cid
    assert entry["acf_bytes"] == len(acf)
    assert entry["kernel_name"] == "wrapped_xor"
    assert entry["shape"] == "f32[4,4096,16]"
    assert entry["speedup"] == 2.0
    assert entry["baseline_ms"] == 0.02
    assert entry["best_ms"] == 0.01


def test_bundle_inspect_missing(tmp_path: Path):
    assert main(["bundle", "inspect", str(tmp_path / "missing.bundle")]) == 2


def test_bundle_inspect_json_without_metadata(tmp_path: Path, capsys):
    path = tmp_path / "t.bundle"
    bundle = new_bundle()
    acf = b"acf-bytes"
    cid = content_id(acf)
    append_acf(bundle, fingerprint="fp1", acf_bytes=acf, content_id=cid)
    save_bundle(path, bundle)
    assert main(["bundle", "inspect", str(path), "--json"]) == 0
    entry = json.loads(capsys.readouterr().out)["entries"][0]
    assert entry["kernel_name"] is None
    assert entry["speedup"] is None

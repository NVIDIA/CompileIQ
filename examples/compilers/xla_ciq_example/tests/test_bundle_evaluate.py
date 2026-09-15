from __future__ import annotations

import json
from pathlib import Path

from xla_ciq.cli import main
from xla_ciq.bundle import EntryMetadata, append_acf, new_bundle, save_bundle
from xla_ciq.fingerprint import content_id, fingerprint

FIXTURES = Path(__file__).parent / "fixtures"


def _seed_bundle(path: Path, *, kernel: str, shape: str) -> str:
    ptx = FIXTURES / "sample.ptx"
    source = ptx.read_bytes()
    fp = fingerprint(
        cuda_version="13.3",
        arch="sm_90a",
        source_bytes=source,
        kernel_name=kernel,
        shape=shape,
    )
    acf = b"MOCK-ACF-for-evaluate"
    cid = content_id(acf)
    bundle = new_bundle()
    append_acf(
        bundle,
        fingerprint=fp,
        acf_bytes=acf,
        content_id=cid,
        metadata=EntryMetadata(
            kernel_name=kernel,
            shape=shape,
            arch="sm_90a",
            cuda_version="13.3",
            baseline_ms=0.02,
            best_ms=0.01,
            speedup=2.0,
        ),
    )
    save_bundle(path, bundle)
    return fp


def test_bundle_evaluate_mock_json(tmp_path: Path, capsys):
    bundle = tmp_path / "t.bundle"
    ptx = FIXTURES / "sample.ptx"
    fp = _seed_bundle(bundle, kernel="fusion_demo", shape="f32[1024]")
    code = main(
        [
            "bundle",
            "evaluate",
            str(bundle),
            "--input",
            str(ptx),
            "--kernel-name",
            "fusion_demo",
            "--shapes",
            "f32[1024]",
            "--mock",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["entries"][0]["fingerprint"] == fp
    assert payload["entries"][0]["recorded_speedup"] == 2.0
    assert payload["entries"][0]["ok"] is True
    assert "speedup_reproduced" in payload["entries"][0]


def test_bundle_evaluate_uses_metadata_when_shapes_omitted(tmp_path: Path, capsys):
    bundle = tmp_path / "t.bundle"
    ptx = FIXTURES / "sample.ptx"
    _seed_bundle(bundle, kernel="fusion_demo", shape="f32[2048]")
    code = main(
        [
            "bundle",
            "evaluate",
            str(bundle),
            "--input",
            str(ptx),
            "--kernel-name",
            "fusion_demo",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"][0]["shape"] == "f32[2048]"
    assert payload["mock"] is True


def test_bundle_evaluate_rejects_fingerprint_mismatch(tmp_path: Path, capsys):
    bundle = tmp_path / "t.bundle"
    ptx = FIXTURES / "sample.ptx"
    _seed_bundle(bundle, kernel="fusion_demo", shape="f32[1024]")
    changed_ptx = tmp_path / "changed.ptx"
    changed_ptx.write_bytes(ptx.read_bytes() + b"\n// changed producer input\n")

    code = main(
        [
            "bundle",
            "evaluate",
            str(bundle),
            "--input",
            str(changed_ptx),
            "--kernel-name",
            "fusion_demo",
        ]
    )

    assert code == 2
    assert "fingerprint mismatch" in capsys.readouterr().err


def test_bundle_evaluate_missing_shape(tmp_path: Path):
    bundle = tmp_path / "t.bundle"
    ptx = FIXTURES / "sample.ptx"
    _seed_bundle(bundle, kernel="fusion_demo", shape="f32[1024]")
    code = main(
        [
            "bundle",
            "evaluate",
            str(bundle),
            "--input",
            str(ptx),
            "--kernel-name",
            "fusion_demo",
            "--shapes",
            "f32[999]",
            "--mock",
        ]
    )
    assert code == 2

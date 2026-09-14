from __future__ import annotations

from pathlib import Path

import pytest

from xla_ciq import xla_ciq_bundle_pb2
from xla_ciq.bundle import (
    BundleError,
    append_acf,
    decode_bundle,
    encode_bundle,
    load_bundle,
    new_bundle,
    remove_acf,
    save_bundle,
)
from xla_ciq.fingerprint import content_id


def test_bundle_descriptor_identity():
    descriptor = xla_ciq_bundle_pb2.XlaCiqTuningBundle.DESCRIPTOR

    assert descriptor.full_name == "xla_ciq.bundle.v1alpha1.XlaCiqTuningBundle"
    assert descriptor.file.name == "xla_ciq_bundle.proto"


def test_roundtrip_encode_decode():
    bundle = new_bundle()
    acf = b"MOCK-ACF-v1\ndata"
    cid = content_id(acf)
    append_acf(
        bundle,
        fingerprint="fp1",
        acf_bytes=acf,
        content_id=cid,
    )
    raw = encode_bundle(bundle)
    loaded = decode_bundle(raw)
    assert loaded.format_version == 1
    assert loaded.fingerprint_to_control_file_id["fp1"] == cid
    assert loaded.control_files[cid].content == acf


@pytest.mark.parametrize("format_version", [0, 2, 999])
def test_decode_rejects_unsupported_format_version(format_version: int):
    bundle = new_bundle(format_version=format_version)

    with pytest.raises(BundleError, match="unsupported bundle format version"):
        decode_bundle(bundle.SerializeToString())


def test_append_rejects_incorrect_content_id():
    bundle = new_bundle()

    with pytest.raises(BundleError, match="does not match content SHA-256"):
        append_acf(
            bundle,
            fingerprint="fp",
            acf_bytes=b"acf",
            content_id="not-the-content-hash",
        )


def test_decode_rejects_incorrect_content_id():
    bundle = new_bundle()
    bundle.control_files["not-the-content-hash"].content = b"acf"
    bundle.fingerprint_to_control_file_id["fp"] = "not-the-content-hash"

    with pytest.raises(BundleError, match="does not match content SHA-256"):
        decode_bundle(bundle.SerializeToString())


def test_decode_rejects_metadata_without_route():
    bundle = new_bundle()
    bundle.fingerprint_to_metadata["orphan"].kernel_name = "example"

    with pytest.raises(BundleError, match="has no routing mapping"):
        decode_bundle(bundle.SerializeToString())


def test_preserve_without_replace(tmp_path: Path):
    path = tmp_path / "t.bundle"
    bundle = new_bundle()
    first = b"first"
    second = b"second"
    append_acf(
        bundle,
        fingerprint="fp",
        acf_bytes=first,
        content_id=content_id(first),
    )
    save_bundle(path, bundle)

    bundle = load_bundle(path)
    action = append_acf(
        bundle,
        fingerprint="fp",
        acf_bytes=second,
        content_id=content_id(second),
        replace=False,
    )
    assert action == "preserved"
    assert bundle.control_files[content_id(first)].content == first


def test_replace_mapping(tmp_path: Path):
    path = tmp_path / "t.bundle"
    bundle = new_bundle()
    first = b"first"
    second = b"second"
    append_acf(
        bundle,
        fingerprint="fp",
        acf_bytes=first,
        content_id=content_id(first),
    )
    action = append_acf(
        bundle,
        fingerprint="fp",
        acf_bytes=second,
        content_id=content_id(second),
        replace=True,
    )
    assert action == "replaced"
    save_bundle(path, bundle)
    loaded = load_bundle(path)
    assert loaded.fingerprint_to_control_file_id["fp"] == content_id(second)
    assert content_id(first) not in loaded.control_files


def test_dedup_shared_acf():
    bundle = new_bundle()
    acf = b"shared"
    cid = content_id(acf)
    append_acf(bundle, fingerprint="a", acf_bytes=acf, content_id=cid)
    append_acf(bundle, fingerprint="b", acf_bytes=acf, content_id=cid)
    assert len(bundle.control_files) == 1
    assert bundle.fingerprint_to_control_file_id["a"] == cid
    assert bundle.fingerprint_to_control_file_id["b"] == cid


def test_remove_acf_gcs_orphans():
    bundle = new_bundle()
    acf = b"only"
    cid = content_id(acf)
    append_acf(bundle, fingerprint="fp", acf_bytes=acf, content_id=cid)
    assert remove_acf(bundle, "fp") is True
    assert "fp" not in bundle.fingerprint_to_control_file_id
    assert cid not in bundle.control_files
    assert remove_acf(bundle, "fp") is False

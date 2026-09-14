"""Load, save, and update the experimental XLA-CIQ tuning bundle.

The schema lives in schemas/xla_ciq_bundle.proto. Regenerate Python bindings with:

    python -m grpc_tools.protoc -I schemas --python_out=src/xla_ciq schemas/xla_ciq_bundle.proto
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xla_ciq import xla_ciq_bundle_pb2
from xla_ciq.fingerprint import content_id as compute_content_id

FORMAT_VERSION = 1

AdvancedControlFile = xla_ciq_bundle_pb2.AdvancedControlFile
XlaCiqTuningBundle = xla_ciq_bundle_pb2.XlaCiqTuningBundle
FingerprintMetadata = xla_ciq_bundle_pb2.FingerprintMetadata


@dataclass(frozen=True)
class EntryMetadata:
    """CLI telemetry stored alongside a fingerprint mapping."""

    kernel_name: str
    shape: str
    arch: str = ""
    cuda_version: str = ""
    baseline_ms: float | None = None
    best_ms: float | None = None
    speedup: float | None = None

    def to_proto(self) -> FingerprintMetadata:
        meta = FingerprintMetadata()
        meta.kernel_name = self.kernel_name
        meta.shape = self.shape
        meta.arch = self.arch
        meta.cuda_version = self.cuda_version
        if (
            self.baseline_ms is not None
            or self.best_ms is not None
            or self.speedup is not None
        ):
            meta.has_timing = True
            if self.baseline_ms is not None:
                meta.baseline_ms = float(self.baseline_ms)
            if self.best_ms is not None:
                meta.best_ms = float(self.best_ms)
            if self.speedup is not None:
                meta.speedup = float(self.speedup)
        return meta

    @classmethod
    def from_proto(cls, meta: FingerprintMetadata) -> EntryMetadata:
        baseline = best = speedup = None
        if meta.has_timing:
            baseline = float(meta.baseline_ms)
            best = float(meta.best_ms)
            speedup = float(meta.speedup)
        return cls(
            kernel_name=meta.kernel_name,
            shape=meta.shape,
            arch=meta.arch,
            cuda_version=meta.cuda_version,
            baseline_ms=baseline,
            best_ms=best,
            speedup=speedup,
        )

    def as_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "shape": self.shape,
            "arch": self.arch or None,
            "cuda_version": self.cuda_version or None,
            "baseline_ms": self.baseline_ms,
            "best_ms": self.best_ms,
            "speedup": self.speedup,
        }


class BundleError(ValueError):
    """Raised for bundle I/O or integrity problems."""


def new_bundle(format_version: int = FORMAT_VERSION) -> XlaCiqTuningBundle:
    bundle = XlaCiqTuningBundle()
    bundle.format_version = format_version
    return bundle


def load_bundle(path: Path) -> XlaCiqTuningBundle:
    data = path.read_bytes()
    if not data:
        raise BundleError(f"bundle is empty: {path}")
    return decode_bundle(data)


def save_bundle(path: Path, bundle: XlaCiqTuningBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encode_bundle(bundle))
    tmp.replace(path)


def encode_bundle(bundle: XlaCiqTuningBundle) -> bytes:
    _validate_bundle(bundle)
    return bundle.SerializeToString()


def decode_bundle(data: bytes) -> XlaCiqTuningBundle:
    bundle = XlaCiqTuningBundle()
    try:
        bundle.ParseFromString(data)
    except Exception as exc:  # protobuf raises DecodeError subclasses
        raise BundleError(f"failed to parse ACF Bundle: {exc}") from exc
    _validate_bundle(bundle)
    return bundle


def append_acf(
    bundle: XlaCiqTuningBundle,
    *,
    fingerprint: str,
    acf_bytes: bytes,
    content_id: str,
    replace: bool = False,
    metadata: EntryMetadata | None = None,
) -> str:
    """Append or update a fingerprint mapping. Returns action label."""
    existing = bundle.fingerprint_to_control_file_id.get(fingerprint)
    if existing is not None and not replace:
        return "preserved"

    expected_content_id = compute_content_id(acf_bytes)
    if content_id != expected_content_id:
        raise BundleError(
            f"control file id {content_id!r} does not match content SHA-256 "
            f"{expected_content_id!r}"
        )

    acf = bundle.control_files[content_id]
    acf.content = acf_bytes
    bundle.fingerprint_to_control_file_id[fingerprint] = content_id
    if metadata is not None:
        bundle.fingerprint_to_metadata[fingerprint].CopyFrom(metadata.to_proto())
    elif fingerprint in bundle.fingerprint_to_metadata:
        del bundle.fingerprint_to_metadata[fingerprint]
    _gc_unreferenced(bundle)
    return "replaced" if existing is not None else "added"


def remove_acf(bundle: XlaCiqTuningBundle, fingerprint: str) -> bool:
    """Remove a fingerprint mapping (and GC orphans). Returns True if removed."""
    if fingerprint not in bundle.fingerprint_to_control_file_id:
        return False
    del bundle.fingerprint_to_control_file_id[fingerprint]
    if fingerprint in bundle.fingerprint_to_metadata:
        del bundle.fingerprint_to_metadata[fingerprint]
    _gc_unreferenced(bundle)
    return True


def get_entry_metadata(
    bundle: XlaCiqTuningBundle, fingerprint: str
) -> EntryMetadata | None:
    if fingerprint not in bundle.fingerprint_to_metadata:
        return None
    return EntryMetadata.from_proto(bundle.fingerprint_to_metadata[fingerprint])


def _validate_bundle(bundle: XlaCiqTuningBundle) -> None:
    if bundle.format_version != FORMAT_VERSION:
        raise BundleError(
            f"unsupported bundle format version {bundle.format_version}; "
            f"expected {FORMAT_VERSION}"
        )
    for cid, acf in bundle.control_files.items():
        expected_cid = compute_content_id(bytes(acf.content))
        if cid != expected_cid:
            raise BundleError(
                f"control file id {cid!r} does not match content SHA-256 "
                f"{expected_cid!r}"
            )
    for fp, cid in bundle.fingerprint_to_control_file_id.items():
        if cid not in bundle.control_files:
            raise BundleError(
                f"fingerprint {fp!r} references missing control file id {cid!r}"
            )
    for fp in bundle.fingerprint_to_metadata:
        if fp not in bundle.fingerprint_to_control_file_id:
            raise BundleError(f"metadata for fingerprint {fp!r} has no routing mapping")


def _gc_unreferenced(bundle: XlaCiqTuningBundle) -> None:
    referenced = set(bundle.fingerprint_to_control_file_id.values())
    for cid in list(bundle.control_files.keys()):
        if cid not in referenced:
            del bundle.control_files[cid]
    for fp in list(bundle.fingerprint_to_metadata.keys()):
        if fp not in bundle.fingerprint_to_control_file_id:
            del bundle.fingerprint_to_metadata[fp]

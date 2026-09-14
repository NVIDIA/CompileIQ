"""Implement `xla-ciq bundle inspect` for an experimental bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from xla_ciq.bundle import BundleError, get_entry_metadata, load_bundle
from xla_ciq.fingerprint import content_id


def run_bundle_inspect(*, bundle: Path, as_json: bool = False) -> int:
    if not bundle.is_file():
        print(f"error: bundle not found: {bundle}", file=sys.stderr)
        return 2
    try:
        message = load_bundle(bundle)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries = []
    for fingerprint, cid in sorted(message.fingerprint_to_control_file_id.items()):
        acf = message.control_files[cid]
        payload = bytes(acf.content)
        entry = {
            "fingerprint": fingerprint,
            "content_id": cid,
            "acf_bytes": len(payload),
            "content_sha256": content_id(payload),
            "kernel_name": None,
            "shape": None,
            "arch": None,
            "cuda_version": None,
            "baseline_ms": None,
            "best_ms": None,
            "speedup": None,
        }
        meta = get_entry_metadata(message, fingerprint)
        if meta is not None:
            entry.update(meta.as_dict())
        entries.append(entry)

    summary = {
        "path": str(bundle),
        "format_version": int(message.format_version),
        "mappings": len(message.fingerprint_to_control_file_id),
        "control_files": len(message.control_files),
        "entries": entries,
    }

    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"bundle: {bundle}")
    print(f"format_version: {summary['format_version']}")
    print(f"mappings: {summary['mappings']}")
    print(f"control_files: {summary['control_files']}")
    if not entries:
        print("(empty)")
        return 0
    for entry in entries:
        parts = [
            f"fingerprint={entry['fingerprint']}",
            f"content_id={entry['content_id']}",
            f"acf_bytes={entry['acf_bytes']}",
        ]
        if entry.get("kernel_name"):
            parts.append(f"kernel={entry['kernel_name']}")
        if entry.get("shape"):
            parts.append(f"shape={entry['shape']}")
        if entry.get("speedup") is not None:
            parts.append(f"speedup={entry['speedup']:.4f}x")
        elif entry.get("baseline_ms") is not None or entry.get("best_ms") is not None:
            parts.append(
                f"baseline_ms={entry.get('baseline_ms')} best_ms={entry.get('best_ms')}"
            )
        print("  " + " ".join(parts))
    return 0

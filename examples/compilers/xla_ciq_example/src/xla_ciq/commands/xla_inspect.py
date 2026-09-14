"""Implement `xla-ciq xla inspect` for an XLA dump directory."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from xla_ciq.xla.dump import DumpInspectError, inspect_dump


def run_xla_inspect(
    *,
    dump_dir: Path,
    kernel_name: str | None = None,
    as_json: bool = False,
) -> int:
    try:
        inspection = inspect_dump(dump_dir, kernel_name=kernel_name)
    except DumpInspectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(asdict(inspection), indent=2, sort_keys=True))
        return 0

    if not inspection.kernels:
        print(f"no PTX kernels found under {dump_dir}")
        return 0

    for kernel in inspection.kernels:
        print(f"kernel: {kernel.kernel_name}")
        for ptx in kernel.ptx_files:
            print(f"  ptx: {ptx}")
        for seq in kernel.thunk_sequence_files:
            print(f"  thunk_sequence: {seq}")
        for meta in kernel.thunk_metadata_files:
            print(f"  thunk_metadata: {meta}")
        for ann in kernel.profile_annotations:
            print(f"  profile_annotation: {ann}")
        if kernel.shapes:
            print(f"  shapes: {','.join(kernel.shapes)}")
        for note in kernel.notes:
            print(f"  note: {note}")
        print()
    return 0

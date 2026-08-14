"""Smoke-test an encrypted search-space artifact through the public CompileIQ API."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from compileiq.ciq import Search
from compileiq.types import SearchConfiguration


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _sha256_value(value: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64:
        raise argparse.ArgumentTypeError("must be a 64-character SHA-256 digest")
    try:
        bytes.fromhex(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a hexadecimal SHA-256 digest") from error
    return normalized


def _decode_candidate(candidate: object, index: int) -> bytes:
    if not isinstance(candidate, str) or not candidate:
        raise RuntimeError(f"candidate {index} is not a non-empty encrypted hex string")

    try:
        decoded = bytes.fromhex(candidate)
    except ValueError as error:
        raise RuntimeError(f"candidate {index} is not valid hexadecimal") from error

    if not decoded:
        raise RuntimeError(f"candidate {index} decodes to an empty payload")
    return decoded


def _smoke_objective(candidate: object) -> float:
    """Return a deterministic score while validating the protected candidate boundary."""
    return float(len(_decode_candidate(candidate, 0)))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_candidates(candidates: Sequence[object], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True)
    manifest: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        payload = _decode_candidate(candidate, index)
        filename = f"candidate-{index:03d}.bin"
        (output_dir / filename).write_bytes(payload)
        manifest.append(
            {
                "filename": filename,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    return manifest


def _new_search(search_space: Path, cache_dir: Path, pool_size: int, cull_size: int) -> Search:
    return Search(
        objective_function=_smoke_objective,
        search_space=search_space,
        search_config=SearchConfiguration(
            generations=1,
            pool_size=pool_size,
            cull_size=cull_size,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    )


def _run_sample(
    search_space: Path,
    output_dir: Path,
    sample_count: int,
    pool_size: int,
    cull_size: int,
) -> dict[str, Any]:
    search = _new_search(search_space, output_dir / "cache", pool_size, cull_size)
    candidates = search.sample(num_samples=sample_count)
    if len(candidates) != sample_count:
        raise RuntimeError(
            f"sample() returned {len(candidates)} candidates; expected {sample_count}"
        )
    return {
        "candidate_count": len(candidates),
        "candidates": _write_candidates(candidates, output_dir / "candidates"),
    }


def _run_search(
    search_space: Path,
    output_dir: Path,
    pool_size: int,
    cull_size: int,
    workers: int,
) -> dict[str, Any]:
    search = _new_search(search_space, output_dir / "cache", pool_size, cull_size)
    results = search.start(num_workers=workers).get_results()
    required_columns = {"score_1", "params", "generation"}
    missing_columns = sorted(required_columns.difference(results.columns))
    if missing_columns:
        raise RuntimeError(f"search results are missing columns: {', '.join(missing_columns)}")
    if results.empty:
        raise RuntimeError("search returned no results")
    if len(results) > pool_size:
        raise RuntimeError(
            f"search returned {len(results)} rows; expected no more than {pool_size}"
        )

    candidates = results["params"].tolist()
    return {
        "candidate_count": len(candidates),
        "candidates": _write_candidates(candidates, output_dir / "candidates"),
        "result_columns": sorted(results.columns.tolist()),
        "result_rows": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("sample", "search", "both"), default="both")
    parser.add_argument("--samples", type=_positive_int, default=8)
    parser.add_argument("--pool-size", type=_positive_int, default=6)
    parser.add_argument("--cull-size", type=_positive_int, default=2)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--expected-sha256", type=_sha256_value)
    parser.add_argument("--expected-size", type=_positive_int)
    args = parser.parse_args()

    if not args.search_space.is_file():
        parser.error(f"search-space file does not exist: {args.search_space}")
    if args.cull_size >= args.pool_size:
        parser.error("--cull-size must be smaller than --pool-size")
    artifact = args.search_space.read_bytes()
    artifact_sha256 = _sha256(artifact)
    if args.expected_sha256 is not None and artifact_sha256 != args.expected_sha256:
        parser.error(
            f"search-space SHA-256 mismatch: expected {args.expected_sha256}, "
            f"got {artifact_sha256}"
        )
    if args.expected_size is not None and len(artifact) != args.expected_size:
        parser.error(
            f"search-space size mismatch: expected {args.expected_size}, got {len(artifact)}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "artifact": {
            "filename": args.search_space.name,
            "sha256": artifact_sha256,
            "size_bytes": len(artifact),
        },
        "mode": args.mode,
        "schema_version": 1,
    }
    if args.mode in {"sample", "both"}:
        manifest["sample"] = _run_sample(
            args.search_space,
            args.output_dir / "sample",
            args.samples,
            max(args.pool_size, args.samples),
            args.cull_size,
        )
    if args.mode in {"search", "both"}:
        manifest["search"] = _run_search(
            args.search_space,
            args.output_dir / "search",
            args.pool_size,
            args.cull_size,
            args.workers,
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

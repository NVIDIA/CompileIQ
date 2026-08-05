"""
NVBench result parsing utilities backed by ``cuda.bench.results``.

This module is intended as a drop-in replacement for ``nvbench_utils.py`` once
the CompileIQ example is ready to depend on the cuda-bench result reader.
"""

from pathlib import Path

import numpy as np
from cuda.bench.results import BenchmarkResult


def parse_nvbench_result(json_path: Path) -> float | None:
    """Parse NVBench --jsonbin output and return P75 latency in seconds.

    The first non-skipped state with timing samples is used. Callers should
    invoke NVBench with filters that leave one benchmark state of interest,
    such as ``-b reduction -a "Elements[pow2]=26"``.
    """

    try:
        result = BenchmarkResult.from_json(json_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None

    for bench in result.values():
        for state in bench:
            if state.is_skipped or state.samples is None:
                continue
            return float(np.percentile(state.samples, 75))

    return None

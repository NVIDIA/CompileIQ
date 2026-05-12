"""
NVBench result parsing utilities.

This module provides helpers for parsing NVBench JSON output. It is kept
separate from the optimization script so that it can be updated independently
— for example, when the ``cuda-bench`` Python package
(https://pypi.org/project/cuda-bench/) provides equivalent functionality.
"""

import json
from pathlib import Path

import numpy as np


def parse_nvbench_result(json_path: Path) -> float | None:
    """Parse NVBench --jsonbin output, return P75 latency (seconds).

    Reads the JSON file, finds the cold sample times binary file,
    and computes the 75th percentile of the timing distribution.

    Note: This function returns the P75 of the *first* cold sample data
    found across all benchmarks and states. It assumes the JSON corresponds
    to a single benchmark run with a single set of parameters (e.g., one
    value in the "Elements" axis). For multi-benchmark or multi-axis runs,
    only the first matching measurement is returned and all others are
    silently ignored. Callers should ensure the NVBench invocation targets
    a single benchmark/parameter combination (e.g., via
    ``-b reduction -a "Elements[pow2]=26"``).
    """
    try:
        with open(json_path) as f:
            root = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    for bench in root.get("benchmarks", []):
        for state in bench.get("states", []):
            if state.get("is_skipped", False):
                continue
            for summary in state.get("summaries", []):
                if summary.get("tag") != "nv/json/bin:nv/cold/sample_times":
                    continue
                data = {d["name"]: d["value"] for d in summary.get("data", [])}
                bin_path = data.get("filename")
                n_samples = int(data.get("size", 0))
                if not bin_path or n_samples == 0:
                    continue
                try:
                    samples = np.fromfile(bin_path, dtype="<f4")
                except FileNotFoundError:
                    continue
                if len(samples) != n_samples:
                    continue
                return float(np.percentile(samples, 75))
    return None

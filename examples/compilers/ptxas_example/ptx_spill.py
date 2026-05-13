"""
CompileIQ PTXAS Example: Minimize register spills in PTX assembly.

Usage:
    python ptx_spill.py [--arch sm_90a]
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config

SCRIPT_DIR = Path(__file__).parent.resolve()
PTX_SOURCE = SCRIPT_DIR / "w8_spill.ptx"


def objective(config: str, arch: str = "sm_90a") -> float:
    """Compile PTX with given config and return spill bytes (lower is better)."""
    tmp_file = f"tmp_{uuid4().hex}.acf"
    save_compiler_config(tmp_file, config)

    try:
        result = subprocess.run(
            ["ptxas", "-v", f"-arch={arch}", "--apply-controls", tmp_file, str(PTX_SOURCE)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return INVALID_SCORE

        match = re.search(r"(\d+) bytes spill stores", result.stdout + result.stderr)
        return float(match.group(1)) if match else INVALID_SCORE

    except Exception:
        return INVALID_SCORE
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def main():
    parser = argparse.ArgumentParser(description="Minimize register spills in PTX")
    parser.add_argument("--arch", default="sm_90a", help="GPU architecture (default: sm_90a)")
    args = parser.parse_args()

    # Check CUDA version
    version_output = subprocess.run(
        ["ptxas", "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
    assert float(cuda_version) >= 13.3, "CompileIQ requires CUDA 13.3+"

    # Configure search
    config = SearchConfiguration(
        problem_type="min",
        generations=5,
        pool_size=32,
    )

    # Run optimization
    tuner = Search(
        objective_function=lambda c: objective(c, args.arch),
        search_space=PtxasSearchSpace(version=cuda_version),
        search_config=config,
    )
    results = tuner.start(num_workers=4)

    # Save best result
    best = results.get_best_result()
    print(f"Best spill bytes: {best['score_1']}")
    save_compiler_config("best_compiler_controls.acf", best["params"])


if __name__ == "__main__":
    main()

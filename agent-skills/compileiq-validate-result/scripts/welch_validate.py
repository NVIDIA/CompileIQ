#!/usr/bin/env python3
"""Validate a CompileIQ candidate via Welch's t-test + Cohen's d.

CLI:
    python welch_validate.py \
        --acf best.acf \
        --baseline-cmd "python bench.py" \
        --opt-cmd     "PTXAS_OPTIONS='--apply-controls=best.acf' python bench.py" \
        --trials 100 --warmup 50 \
        --score-regex 'mean: ([0-9.]+)' \
        --output validation-log.csv

Importable:
    from welch_validate import validate_speedup
    result = validate_speedup(baseline_ms, optimized_ms)

Self-test:
    python welch_validate.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def validate_speedup(baseline_ms: np.ndarray, optimized_ms: np.ndarray) -> dict:
    """Run Welch's t-test + Cohen's d on two arrays of measurements.

    Lower-is-better convention (latency in ms or seconds).

    Returns a dict with: speedup_mean, speedup_median, p_value, cohens_d,
    significant, baseline (mean/std/p5/p95), optimized (mean/std/p5/p95).
    """
    from scipy import stats

    baseline_ms = np.asarray(baseline_ms, dtype=float)
    optimized_ms = np.asarray(optimized_ms, dtype=float)

    if baseline_ms.size < 5 or optimized_ms.size < 5:
        raise ValueError("Need >=5 trials per side; got "
                         f"baseline={baseline_ms.size}, optimized={optimized_ms.size}")

    _, p = stats.ttest_ind(baseline_ms, optimized_ms, equal_var=False)
    b_mean = float(baseline_ms.mean())
    b_std = float(baseline_ms.std(ddof=1))
    o_mean = float(optimized_ms.mean())
    o_std = float(optimized_ms.std(ddof=1))
    pooled = float(np.sqrt((b_std**2 + o_std**2) / 2))
    d = (b_mean - o_mean) / pooled if pooled > 0 else 0.0

    return {
        "speedup_mean":   b_mean / o_mean if o_mean > 0 else float("inf"),
        "speedup_median": float(np.median(baseline_ms) / np.median(optimized_ms)),
        "p_value":        float(p),
        "cohens_d":       float(d),
        "significant":    bool(p < 0.05 and o_mean < b_mean and d > 0.2),
        "baseline": {
            "mean": b_mean, "std": b_std,
            "p5":  float(np.percentile(baseline_ms, 5)),
            "p95": float(np.percentile(baseline_ms, 95)),
        },
        "optimized": {
            "mean": o_mean, "std": o_std,
            "p5":  float(np.percentile(optimized_ms, 5)),
            "p95": float(np.percentile(optimized_ms, 95)),
        },
    }


def reject_higher_variance(result: dict, threshold: float = 0.25) -> str | None:
    """Return a reject reason if optimized p5-p95 range is materially wider, else None."""
    b = result["baseline"]
    o = result["optimized"]
    b_range = b["p95"] - b["p5"]
    o_range = o["p95"] - o["p5"]
    if b_range <= 0:
        return None
    ratio = (o_range - b_range) / b_range
    if ratio > threshold:
        return f"higher_variance(p5_p95_widened_by_{ratio*100:.1f}%)"
    return None


def reject_lucky_min(result: dict) -> str | None:
    """Return a reject reason if optimized.mean >= baseline.mean (lucky-min pattern), else None."""
    if result["optimized"]["mean"] >= result["baseline"]["mean"]:
        return "lucky_min(optimized_mean_not_lower)"
    return None


def run_n(cmd: str, n: int, score_regex: re.Pattern) -> list[float]:
    """Run a shell command N times, extract the score regex from stdout each time."""
    times = []
    for _ in range(n):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        m = score_regex.search(proc.stdout)
        if not m:
            continue
        try:
            times.append(float(m.group(1)))
        except (ValueError, IndexError):
            continue
    return times


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_log(output: str, fields: dict) -> None:
    path = Path(output)
    header = list(fields.keys())
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(fields)


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a CompileIQ candidate via Welch's t-test.")
    ap.add_argument("--acf", help="Path to the ACF being validated.")
    ap.add_argument("--baseline-cmd", help="Shell command for the baseline run.")
    ap.add_argument("--opt-cmd", help="Shell command for the optimized run.")
    ap.add_argument("--trials", type=int, default=100, help="Trials per side (default 100).")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup runs discarded (default 50).")
    ap.add_argument("--score-regex", default=r"mean: ([0-9.]+)",
                    help="Regex with one capture group for the per-run score "
                         "(default: 'mean: ([0-9.]+)').")
    ap.add_argument("--output", default="validation-log.csv",
                    help="CSV file to append the result row to.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run unit-test-style synthetic checks and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not (args.acf and args.baseline_cmd and args.opt_cmd):
        ap.error("--acf, --baseline-cmd, --opt-cmd are required (unless --self-test).")

    regex = re.compile(args.score_regex)
    print(f"Warmup ({args.warmup} runs each side)...")
    run_n(args.baseline_cmd, args.warmup, regex)
    run_n(args.opt_cmd, args.warmup, regex)
    print(f"Baseline ({args.trials} trials)...")
    baseline_ms = run_n(args.baseline_cmd, args.trials, regex)
    print(f"Candidate ({args.trials} trials)...")
    optimized_ms = run_n(args.opt_cmd, args.trials, regex)

    if len(baseline_ms) < args.trials // 2 or len(optimized_ms) < args.trials // 2:
        print(f"ERROR: fewer than half of trials produced a score "
              f"(baseline {len(baseline_ms)}/{args.trials}, "
              f"optimized {len(optimized_ms)}/{args.trials}). "
              f"Check --score-regex or the benchmark stdout.", file=sys.stderr)
        return 2

    result = validate_speedup(np.array(baseline_ms), np.array(optimized_ms))
    reasons = [r for r in (reject_lucky_min(result), reject_higher_variance(result)) if r]

    if result["significant"] and not reasons:
        decision = (
            f"KEPT:speedup={result['speedup_mean']:.4f}x,"
            f"p={result['p_value']:.4g},"
            f"d={result['cohens_d']:.3f}"
        )
        rc = 0
    else:
        if reasons:
            reason = ",".join(reasons)
        else:
            reason = f"not_significant(p={result['p_value']:.4g},d={result['cohens_d']:.3f})"
        decision = f"REJECTED:{reason}"
        rc = 1

    print(decision)
    b = result["baseline"]
    o = result["optimized"]
    print(f"  baseline: mean={b['mean']:.6f} ± {b['std']:.6f}  "
          f"p5={b['p5']:.6f}  p95={b['p95']:.6f}")
    print(f"  optimized: mean={o['mean']:.6f} ± {o['std']:.6f}  "
          f"p5={o['p5']:.6f}  p95={o['p95']:.6f}")

    append_log(args.output, {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "acf": args.acf,
        "sha256": sha256_of(args.acf),
        "trials": args.trials,
        "warmup": args.warmup,
        "baseline_mean": result["baseline"]["mean"],
        "baseline_std":  result["baseline"]["std"],
        "optimized_mean": result["optimized"]["mean"],
        "optimized_std":  result["optimized"]["std"],
        "speedup_mean":   result["speedup_mean"],
        "p_value":        result["p_value"],
        "cohens_d":       result["cohens_d"],
        "decision":       decision,
        "baseline_cmd":   args.baseline_cmd,
        "opt_cmd":        args.opt_cmd,
    })
    return rc


def _self_test() -> int:
    rng = np.random.default_rng(42)

    # Identical distributions: expect NOT significant
    a = rng.normal(loc=1.0, scale=0.01, size=200)
    b = rng.normal(loc=1.0, scale=0.01, size=200)
    r = validate_speedup(a, b)
    assert not r["significant"], f"identical dists should not be significant: {r}"
    print(
        f"identical: p={r['p_value']:.4g}  d={r['cohens_d']:.3f}  "
        f"significant={r['significant']}  OK"
    )

    # Optimized clearly faster: expect significant
    a = rng.normal(loc=1.0, scale=0.01, size=200)
    b = rng.normal(loc=0.85, scale=0.01, size=200)
    r = validate_speedup(a, b)
    assert r["significant"], f"clear speedup should be significant: {r}"
    assert r["speedup_mean"] > 1.1
    print(
        f"speedup: p={r['p_value']:.4g}  d={r['cohens_d']:.3f}  "
        f"speedup={r['speedup_mean']:.3f}  OK"
    )

    # Lucky-min pattern: optimized has lower min but same mean
    a = rng.normal(loc=1.0, scale=0.01, size=200)
    b_lucky = rng.normal(loc=1.0, scale=0.03, size=200)   # same mean, wider
    r = validate_speedup(a, b_lucky)
    assert reject_higher_variance(r) is not None, "wider distribution should be flagged"
    print("higher_variance correctly flagged  OK")

    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(cli())

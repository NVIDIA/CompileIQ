# Manifest

One-line index of every skill in this directory. Keep this concise; the
authoritative content lives in each `SKILL.md`.

| Skill | Version | One-liner |
|---|---|---|
| [`compileiq-bootstrap`](compileiq-bootstrap/SKILL.md) | 1.1.0 | Validate CUDA 13.3+, imports, and search-space resolution; flag the precise next fix. |
| [`compileiq-booster-pack`](compileiq-booster-pack/SKILL.md) | 1.0.0 | Try curated ACF candidates before paying for a full search; includes the O0/O3 ACF-injection canary. |
| [`compileiq-search-space`](compileiq-search-space/SKILL.md) | 1.1.0 | Pick `PtxasSearchSpace` / `NvccSearchSpace` / `LocalSearchSpaceBin`; the `att` variant is for **attention** workloads. |
| [`compileiq-author-objective`](compileiq-author-objective/SKILL.md) | 1.0.0 | Write a self-contained objective using the current `save_compiler_config` API across PTXAS / NVCC / Triton / Helion / cuTeDSL / FlashInfer. |
| [`compileiq-run-search`](compileiq-run-search/SKILL.md) | 1.1.0 | Compose `Search(...)`, pick the right Worker, size `SearchConfiguration`, set `dump_results=`. |
| [`compileiq-validate-result`](compileiq-validate-result/SKILL.md) | 1.0.0 | Welch's t-test + Cohen's d gate, three false-positive rejections, reproducibility log. |
| [`compileiq-debug`](compileiq-debug/SKILL.md) | 1.1.0 | Symptom-indexed cheat sheet; CSV diagnostics; NCU and register-spill pointers. |

## Scripts

| Path | Purpose |
|---|---|
| [`compileiq-bootstrap/scripts/check_env.sh`](compileiq-bootstrap/scripts/check_env.sh) | Env validation; exit code = # of failures. |
| [`compileiq-booster-pack/scripts/apply_one_acf.sh`](compileiq-booster-pack/scripts/apply_one_acf.sh) | Apply one ACF, measure, append a row to `booster-pack-log.csv`. `--self-test`. |
| [`compileiq-run-search/scripts/smoke_search.py`](compileiq-run-search/scripts/smoke_search.py) | 2-generation `x**2 + y` smoke run. |
| [`compileiq-validate-result/scripts/welch_validate.py`](compileiq-validate-result/scripts/welch_validate.py) | Welch's t-test CLI + importable. `--self-test`. |
| [`compileiq-debug/scripts/diagnose_csv.py`](compileiq-debug/scripts/diagnose_csv.py) | Heuristic CSV diagnostics. `--self-test`. |

## References

| Path | Purpose |
|---|---|
| [`compileiq-author-objective/references/templates.md`](compileiq-author-objective/references/templates.md) | Five paste-ready objective-function templates (NVBench, Triton, Helion, raw PTX, cuTeDSL). |

## Changelog

- **1.1.0** (unreleased): the native `amber` engine replaces the original search
  core and is the default. `compileiq-bootstrap` and `compileiq-debug` drop
  `CIQ_SOCKET_TIMEOUT`, and `compileiq-debug` adds rows for retired search
  settings, a missing native runtime, and searches that end early when amber
  runs out of untried candidates.
  `compileiq-run-search` replaces `cull_size` and `mutate_rate` guidance with
  amber options. `compileiq-search-space` says "search engine" instead of
  "optimization core".
- **1.0.0** (2026-05-14): Initial release. Replaces the older
  `compileiq-skills-v1.0.0` slash-command tarball. Major API updates:
  `save_compiler_config` instead of inline `bytes.fromhex`; provider classes
  instead of `CompilerSearchSpaces` enum; first-class booster-pack workflow;
  release-backed search-space resolution; cross-agent installer.

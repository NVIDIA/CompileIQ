# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

CompileIQ is NVIDIA's evolutionary-based hyperparameter optimizer for tuning compiler controls. It wraps an evolutionary core (precompiled binaries in `compileiq/core/executable/`) with a Python API, communicating via socket-based IPC.

## Build & Development Commands

A `Makefile` provides common targets. Use `make help` to list them all.

```bash
# Install all dev dependencies
make install

# Quick validation (lint + unit tests)
make validate

# Lint
make lint

# Unit tests
make test-unit

# Integration tests
make test-integration

# All tests (unit + integration, NOT fuzz)
make test-all

# Fuzz tests (slow, Hypothesis with wide parameter ranges — excluded by default)
make test-fuzz

# Tests with coverage report
make test-cov

# Run a single test
poetry run pytest tests/unit/test_ciq.py -vvv
poetry run pytest tests/unit/test_ciq.py::TestClassName::test_method -vvv

# Build docs
make install-docs && poetry run sphinx-multiversion -E -a docs/ public/

# Build wheel
make build
```

Python version: >=3.11, <3.14. CI uses 3.11.4.

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures (mock core, mock sockets, sandbox cache dir)
├── utils.py             # Test helpers (objective functions, param generators, validators)
├── unit/                # Fast, deterministic, no external deps
├── integration/         # Mocked-core Search tests, worker tests, legacy config tests
└── fuzz/                # Hypothesis-based exploration with wide parameter ranges
```

- **Unit tests** — pure logic, no mocks of Search/core needed.
- **Integration tests** — exercise the Search API with mocked core/sockets. Deterministic `@pytest.mark.parametrize` with representative configs.
- **Fuzz tests** — Hypothesis with wide ranges (pool_size 6-360, gens 1-5, etc.). **Excluded from default pytest runs** via `addopts = "--ignore=tests/fuzz"` in pyproject.toml. Run explicitly with `make test-fuzz` or `pytest tests/fuzz/ -vvv`. Default `max_examples=20`; set `CIQ_FUZZ_EXAMPLES=100` for thorough runs (nightly CI does this).

### Pytest markers

- `requires_ray` — test needs a running Ray cluster
- `requires_ipc` — test needs real sockets or subprocesses
- `requires_core` — test runs the real core binary (not sandbox-compatible)

## Architecture

**Core flow:** User defines an objective function and search space → `Search` (ciq.py) serializes config and launches the core subprocess → core generates parameter candidates via evolutionary algorithms → Python workers evaluate the objective function in parallel → scores are sent back to core via socket IPC → repeat for N generations.

Key modules:

- **`compileiq/ciq.py`** — `Search` class, the main entry point. Manages core subprocess lifecycle and socket communication.
- **`compileiq/worker.py`** — Worker backends: `MultiProcessWorker` (default, local), `IsoMultiProcessWorker` (one process per task, kill-safe on timeout), `RayWorker` (distributed), `AsyncWorker` (asyncio).
- **`compileiq/types.py`** — All configuration types and enums (ProblemType, SearchConfiguration, WorkerType, etc.). Uses Pydantic models.
- **`compileiq/core/core_comms.py`** — `CoreIPC` class handling socket-based message exchange with the core.
- **`compileiq/core/core_types.py`** — Pydantic models for core IPC messages (ParameterSet, EvaluatedDnaResponse).
- **`compileiq/tracker.py`** — Pluggable experiment tracking (LoguruTracker, MLflowTracker, DisabledTracker).
- **`compileiq/results.py`** — `SearchResult` wrapping pandas DataFrame with optimization-specific methods (get_best_result, pareto_front).
- **`compileiq/search_spaces/`** — Search space definitions; `base.py` has primitives (range, choice, literal, log_sampling), `compilers.py` has NVIDIA compiler-specific spaces.
- **`compileiq/utils/`** — Score validation, encoding/decoding helpers.
- **`compileiq/config/const.py`** — Constants and environment variable configuration (CIQ_SOCKET_TIMEOUT, CIQ_KEEP_CACHE).
- **`assets/`** — Curated search space binaries and configs for compiler tuning.

## Code Style

- Ruff linter with rules E and F selected, line length 100, indent width 4
- Double quotes, space indentation
- Target Python 3.11

## Environment Variables

- `CIQ_SOCKET_TIMEOUT` (default 20): Socket timeout for core communication. Increase for large search spaces.
- `CIQ_KEEP_CACHE` (default False): Retain `.cache` files after runs.
- `CIQ_FUZZ_EXAMPLES` (default 20): Hypothesis `max_examples` for fuzz tests. Nightly CI uses 100.

## CI/CD

GitHub Actions workflows cover validation, testing, and deploy-related tasks. Linting and unit tests run in validation jobs. Integration tests, example runs, fuzz tests, and binary/internal validation run in test jobs. Fuzz tests run last after integration tests pass.

## Available Agent Skills

This repository ships an agent-agnostic skill set under `agent-skills/` that any AGENTS.md-aware coding agent (Claude Code, Codex, Cursor, GitHub Copilot, Aider, Windsurf) can use to drive CompileIQ optimization campaigns. The skills follow the [agentskills.io](https://agentskills.io) `SKILL.md` convention; an installer renders the right format for each agent.

| Skill | Use when |
| --- | --- |
| `compileiq-bootstrap` | First-time setup, socket timeout, or before any other compileiq-* skill. |
| `compileiq-booster-pack` | Try a curated ACF candidate (incl. the Debug pack's O0/O3 injection canary) before paying for a full search. |
| `compileiq-search-space` | Pick a provider class or pin a release; the `att` variant is curated for **attention** workloads (FA / GQA / MHA / MLA / FlashInfer Batch Decode). |
| `compileiq-author-objective` | Write or fix the function passed as `objective_function=`; covers PTXAS / NVCC / Triton / Helion / cuTeDSL / FlashInfer injection. |
| `compileiq-run-search` | Compose `Search(...)`, pick the right Worker, size `SearchConfiguration`. |
| `compileiq-validate-result` | Welch's t-test + Cohen's d gate before claiming any speedup or shipping an ACF. |
| `compileiq-debug` | Symptom-indexed cheat sheet for anything unexpected. |

**Recommended order for a fresh project:** `bootstrap → booster-pack → (only if no pack helps) author-objective → run-search → validate-result`. `debug` is available throughout.

To install for your agent: `bash agent-skills/install.sh --agents <claude-code|codex|cursor|copilot|aider|windsurf>` (or run with no `--agents` flag to auto-detect). See [`agent-skills/README.md`](agent-skills/README.md) for the full layout, authoring conventions, and `--check` / `--uninstall`.

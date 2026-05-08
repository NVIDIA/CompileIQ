# Pyright configuration rationale

`pyrightconfig.json` is JSON, so this file holds the reasoning behind the
non-default choices.

## Scope

`include = ["compileiq", "tests"]` checks library code and tests. Fuzz tests stay
included, with targeted relaxations for noisy Hypothesis call patterns.

`exclude` skips precompiled core binaries, bytecode caches, and local virtual
environments.

## Python version and platform

`pythonVersion = "3.11"` matches the minimum supported Python version. The
project supports newer versions too, but checking the floor keeps accidental use
of newer syntax or APIs visible. `pythonPlatform = "All"` keeps the check
portable across Linux and Windows.

## Checking mode

`typeCheckingMode = "standard"` is a practical starting gate. `basic` misses too
many useful diagnostics; `strict` would require much higher annotation coverage
before it becomes signal instead of migration noise.

## Third-party stubs

`reportMissingTypeStubs = "none"` avoids noise from dependencies that do not ship
complete stubs. First-party missing imports still report as errors.

## Hygiene

`reportUnnecessaryTypeIgnoreComment = "warning"` and
`reportUnnecessaryCast = "warning"` keep suppressions from becoming stale as
types improve.

## Test relaxations

Tests attach runtime state to `pytest` and make heavy use of mocks, dynamic
Pydantic construction, and Hypothesis-generated kwargs. The execution
environments keep those known patterns quiet without weakening diagnostics for
the library code.

The ordering matters: pyright uses the first matching environment, so the more
specific `tests/integration` and `tests/fuzz` roots must appear before the
parent `tests` root.

## Running

```bash
poetry run pyright
make typecheck
```

CI runs pyright as a gating job.

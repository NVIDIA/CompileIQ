# Contributing to CompileIQ

CompileIQ is NVIDIA's hyperparameter optimizer for tuning compiler controls. This guide is for both external community contributors and NVIDIA team members. The expected flow is: open an issue, branch from `main`, code, add tests, sign off your commits per the [DCO](#signing-your-work), and open a pull request for review.

For installation and end-user docs, see [`README.md`](README.md) and the [CompileIQ documentation](https://nvidia.github.io/CompileIQ/stable/). For internal architecture and module layout, see [`AGENTS.md`](AGENTS.md).

## Reporting issues

File issues from the templates in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/):

- [`1-bug-report.yml`](.github/ISSUE_TEMPLATE/1-bug-report.yml) — bugs and regressions.
- [`2-feature-request.md`](.github/ISSUE_TEMPLATE/2-feature-request.md) — new functionality.
- [`3-question.md`](.github/ISSUE_TEMPLATE/3-question.md) — usage questions.
- [`4-documentation.md`](.github/ISSUE_TEMPLATE/4-documentation.md) — documentation gaps or errors.

For bug reports, include the CompileIQ version (`pip show compileiq`), Python version, OS, and a minimal reproducer. The maintainers listed in [`.github/CODEOWNERS`](.github/CODEOWNERS) triage incoming issues.

**Security:** Do *not* report security vulnerabilities through public GitHub issues or pull requests. Follow the disclosure process in [`SECURITY.md`](SECURITY.md) (web form or `psirt@nvidia.com`).

## Types of contributions

- **Bug fix or small change** — open a pull request directly; reference the issue with `closes #N` if one exists.
- **New feature or non-trivial enhancement** — open an issue first to align on design before writing code.
- **Documentation or [`agent-skills/`](agent-skills/) tweaks** — open a pull request directly; label the related issue (if any) as `documentation`.

## Development setup

Prerequisites: Python 3.11–3.13, [Poetry](https://python-poetry.org/), `git`, and either Linux or Windows (the supported CI hosts).

External contributors fork [`NVIDIA/CompileIQ`](https://github.com/NVIDIA/CompileIQ) and add an upstream remote:

```bash
git clone https://github.com/<your-username>/CompileIQ.git
cd CompileIQ
git remote add upstream https://github.com/NVIDIA/CompileIQ.git
```

NVIDIA team members may clone `NVIDIA/CompileIQ` directly.

Install dependencies:

```bash
make install           # dev dependencies (linter, typecheck, unittest, tracking)
make install-examples  # add examples deps
make install-docs      # add docs deps
```

Verify your environment:

```bash
make verify-core   # validate bundled core binaries against core-manifest.json
make validate      # lint + typecheck + verify-core + unit tests
```

For deeper architecture and module-by-module orientation, see [`AGENTS.md`](AGENTS.md).

## Branching workflow

- **External contributors:** branch in your fork and open a pull request against `NVIDIA/CompileIQ:main`. Any branch name is fine in your fork.
- **NVIDIA team members:** push a feature branch directly to `NVIDIA/CompileIQ` and open a pull request against `main`. Direct pushes to `main` are not allowed — everything lands through pull request review.
- **Branch naming (internal):** use `<github-username>/<short-description>`, e.g. `asrikanth/add-contributing-md`. Keep it descriptive and lowercase-with-hyphens.
- **Long-lived branches:**
  - `main` — active development; all pull requests target this.
  - `release-x.y` — release branches; hotfixes only.
  - `gh-pages` — managed by the docs workflow; do not push directly.

## Coding standards

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting and [Pyright](https://microsoft.github.io/pyright/) for type checking. Both are configured in [`pyproject.toml`](pyproject.toml) and [`pyrightconfig.json`](pyrightconfig.json).

- Ruff: rules `E` and `F`, line length 100, indent 4 spaces, double quotes, target Python 3.11.
- Pyright: standard mode against `compileiq/` and `tests/`. Looser checks under `tests/integration/` and `tests/fuzz/` are intentional.

Run before pushing:

```bash
make lint           # ruff check
make lint-fix       # ruff check --fix
make format         # ruff format
make format-check   # verify formatting without changes
make typecheck      # pyright
```

Match the patterns of existing modules; the module map in [`AGENTS.md`](AGENTS.md) is the canonical reference.

## Tests

Every behavior-changing pull request must add or update tests.

The test suite has three tiers (see [`tests/`](tests/) and [`AGENTS.md`](AGENTS.md)):

- `tests/unit/` — fast, deterministic, no external deps.
- `tests/integration/` — mocked-core Search tests with deterministic `@pytest.mark.parametrize`.
- `tests/fuzz/` — [Hypothesis](https://hypothesis.readthedocs.io/)-based with wide parameter ranges. **Excluded from default `pytest` runs** via `addopts = "--ignore=tests/fuzz"` in [`pyproject.toml`](pyproject.toml). Run explicitly. Set `CIQ_FUZZ_EXAMPLES=100` for thorough sweeps (the nightly workflow does this).

Markers (defined in `pyproject.toml`):

- `requires_ray` — needs a running Ray cluster.
- `requires_ipc` — needs real sockets or subprocesses.
- `requires_core` — runs the real core binary (not sandbox-compatible).

Commands:

```bash
make test-unit          # all unit tests
make test-integration   # all integration tests
make test-fuzz          # fuzz tests (slow)
make test-cov           # unit + integration with terminal coverage
make test-cov-html      # ... with HTML coverage report

# Single test:
poetry run pytest tests/unit/test_ciq.py::TestClassName::test_method -vvv
```

If your change affects user-facing behavior or public API, preview the docs locally with `make docs-preview` (then open <http://localhost:8000/latest/>) and update files under [`docs/`](docs/) as needed.

## Commit messages

- Imperative subject, ≤ 72 characters: *"Add core manifest verification"*, *"Fix flaky single-objective integration test"*.
- An optional `type:` prefix is used loosely in this repo — `doc:`, `chore:`, `fix:`. Not validated, not required.
- Use the commit body to explain *why* when the change is not obvious from the diff.
- Reference issues with `closes #N` or `refs #N` in the body or the pull-request description.
- **Every commit must be DCO-signed** — see [Signing your work](#signing-your-work). Use `git commit -s`.

## Submitting a pull request

1. Push your branch and open a pull request against `main`.
2. Fill in [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) end-to-end: description, checklist, test plan, and the bug-fix or new-feature code example where relevant.
3. Keep pull requests focused — one concern per pull request. Split refactors from behavior changes when feasible.
4. Update [`docs/`](docs/) for any public-API or user-facing change.

CI must be green before review. Required and informative checks include `lint`, `typecheck`, `verify-core`, `unit-test` (Python 3.11/3.12/3.13 × Ubuntu/Windows), `integration-test`, `fuzz-test`, `run-examples`, `validate-binary-ss`, and the wheel smoke tests. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full matrix.

## Code review

- At least one [CODEOWNERS](.github/CODEOWNERS) approval is required before merge.
- Address review comments with additional commits — do not force-push during active review.
- Pull requests idle for more than 30 days may be closed; reopen when you have time to continue.

## Signing your work

Every commit must carry a `Signed-off-by:` trailer matching your git `user.name` and `user.email`. The simplest way is to use `-s` on every commit:

```bash
git commit -s -m "Add core manifest verification"
```

This requirement applies to commits made on or after this guide lands; existing history is not retroactively re-signed.

If you forgot to sign off, fix it before pushing:

```bash
git commit --amend -s --no-edit                # last commit
git rebase -i --signoff <upstream-base>        # older commits in your branch
```

By signing off, you certify the following Developer Certificate of Origin for every contribution you make:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## License

CompileIQ is distributed under the NVIDIA Software License Agreement; see [`LICENSE`](LICENSE), [`EULA`](EULA), and [`NOTICE`](NOTICE) for the full terms. Opening a pull request signifies acceptance of those terms in addition to the DCO sign-off above.

---


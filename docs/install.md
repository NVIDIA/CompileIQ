# Install Instructions

You can either install through PyPI:

```bash
pip install compileiq
```

The CompileIQ 1.1 release contract is a single, complete `compileiq`
distribution: each published platform wheel contains the native `amber` engine
(NSGA-III), with no separate runtime package for users to select or keep
compatible. `amber` is the default and the only bundled engine on every platform.

In CompileIQ 1.1, `amber` replaces the original search core, so re-validate results
tuned with an earlier release. The `SearchConfiguration` settings `cull_size`,
`mutate_rate`, `objective_weights`, and `enable_large_fail_pool` have no effect since
1.1; `Search` ignores a non-default value with a `FutureWarning` that names the
closest `amber` option, so 1.0 configurations keep running, and a future release will
remove these fields. `amber` never repeats a candidate, so a search in a small or
converged space can end early with a `SearchSpaceExhaustedWarning` and the results
so far.

For editable repository development, the default bootstrap installs the frontend
and materializes the pinned, manifest-verified native `amber` payload checked into
the repository through Git LFS, so fetch the LFS objects first:

```bash
git lfs pull
make install
```

`make install-complete` remains as an alias for `make install`. To install only
the frontend, without an engine, for linting, type checking, and unit tests, use:

```bash
make install-frontend
```

The repository contains one manifest-verified native artifact per supported
runtime cell. The installer selects exactly one artifact compatible with the
active Python and platform and rejects mismatched or ambiguous cells. Provider
source is not included. There is no alternate frontend API or second user
installation contract.

## Supported Platforms

CompileIQ 1.1 targets Python 3.11, 3.12, 3.13, and 3.14. A platform is published only
when its complete wheel contains every backend available for that platform; no
same-version incomplete wheel or source fallback is published for a cell whose
native payload is missing. The current beta CI matrix composes, verifies, and
clean-installs these cells end to end:

- Linux x86_64 / CPython 3.11, 3.12, 3.13, and 3.14
- Linux ARM64 / CPython 3.11, 3.12, 3.13, and 3.14
- Windows AMD64 / CPython 3.11, 3.12, 3.13, and 3.14
- Apple Silicon macOS 15+ / CPython 3.11, 3.12, 3.13, and 3.14

Every cell provides the same `amber` engine.

Other platform and Python combinations are outside this release's supported matrix.

Linux wheels target glibc 2.34 or newer and are tagged `manylinux_2_34`.

## Runtime Requirements

On Linux, the native runtime expects the platform C/C++ runtime libraries to be
available on the host. Package names vary by distribution; on Debian/Ubuntu, the
direct runtime packages for the current Linux wheels are:

```bash
sudo apt-get update
sudo apt-get install -y \
  libc6 \
  libgcc-s1 \
  libstdc++6
```

On Windows, install the Microsoft Visual C++ Redistributable 2015 or newer
(x64) if it is not already present.

## Managing Searches with Coding Agents

CompileIQ ships an agent-agnostic skill set under `agent-skills/` that any
AGENTS.md-aware coding agent (Claude Code, Codex, Cursor, GitHub Copilot,
Aider, Windsurf) can use to drive optimization campaigns. The skills follow the
[agentskills.io](https://agentskills.io) `SKILL.md` convention.

| Skill | Use when |
| --- | --- |
| `compileiq-bootstrap` | First-time setup, a hang on the first evaluation, or before any other skill |
| `compileiq-booster-pack` | Try a curated ACF candidate before paying for a full search |
| `compileiq-search-space` | Choosing a provider class or pinning a search-space release (incl. the `att` variant for attention workloads) |
| `compileiq-author-objective` | Writing or fixing the function passed as `objective_function=` |
| `compileiq-run-search` | Composing `Search(...)` and calling `.start()` |
| `compileiq-validate-result` | Validating a winning ACF with Welch's t-test before shipping |
| `compileiq-debug` | Any unexpected behavior; symptom-indexed table |

Install for your agent of choice (auto-detects available agents if `--agents`
is omitted):

```bash
bash agent-skills/install.sh                          # auto-detect
bash agent-skills/install.sh --agents claude-code,codex,cursor,copilot
bash agent-skills/install.sh --check                   # verify
bash agent-skills/install.sh --uninstall               # remove mounts
```

Recommended order for a fresh project: **bootstrap → booster-pack → (only if
no pack helps) author-objective → run-search → validate-result**, with
**debug** available throughout. See [`agent-skills/README.md`](../agent-skills/README.md)
for the full layout and authoring conventions.

## Environment Configuration Options

| Environment Variable | Default Value | Type | Description
| ------ | ------ | ------ | ------ |
| CIQ_KEEP_CACHE | False | bool | If set to True, `.cache` files will not be deleted.
| CIQ_PROCESS_MODE | "forkserver" | str | User can set this to "fork" to better separate processes and deal with threads. `IsoMultiProcessWorker` uses `fork` as the default.
| CIQ_SEARCH_SPACES_DIR | unset | path | Reads compiler search-space `manifest.json` and `.bin` files from a local mirror instead of GitHub.
| CIQ_SEARCH_SPACES_REPO | NVIDIA/CompileIQ | str | GitHub repository used for search-space release lookups, useful for staging or a future dedicated asset repo.
| CIQ_SS_LATEST_TTL_DAYS | 7 | float | Reuses the cached GitHub `latest` search-space release tag for this many days, reducing anonymous REST API calls that can be throttled on shared IPs. Set to `0` to check GitHub every time.

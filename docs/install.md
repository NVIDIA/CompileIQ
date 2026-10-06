# Install Instructions

You can either install through PyPI:

```bash
pip install compileiq
```

The CompileIQ 1.1 release contract is a single, complete `compileiq`
distribution: each published platform wheel contains every backend available for
that platform, with no separate runtime package for users to select or keep
compatible. Linux and Windows complete wheels contain the bundled `ruby` and native
`amber` backends. The macOS ARM64 beta wheel contains `amber` only because no macOS
legacy core is shipped.

For editable repository development, the default bootstrap installs the frontend
with the bundled `ruby` engine only, so it needs no native payload:

```bash
make install
```

To also materialize the pinned, manifest-verified native `amber` payload checked
into the repository through Git LFS, fetch the LFS objects and use the complete
install target:

```bash
git lfs pull
make install-complete
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

- Linux x86_64 / CPython 3.11, 3.12, 3.13, and 3.14: `ruby` and `amber`
- Linux ARM64 / CPython 3.11, 3.12, 3.13, and 3.14: `ruby` and `amber`
- Windows AMD64 / CPython 3.11, 3.12, 3.13, and 3.14: `ruby` and `amber`
- Apple Silicon macOS 15+ / CPython 3.11, 3.12, 3.13, and 3.14: `amber` only

Other platform and Python combinations are outside this release's supported matrix.

Linux wheels target glibc 2.34 or newer and are tagged `manylinux_2_34`. The
macOS ARM64 release wheels are intentionally coreless because the legacy core is
not packaged for macOS.

## Runtime Requirements

On Linux, the bundled core expects the platform C/C++ runtime libraries and
zlib to be available on the host. Package names vary by distribution; on
Debian/Ubuntu, the direct runtime packages for the current Linux wheels are:

```bash
sudo apt-get update
sudo apt-get install -y \
  libc6 \
  libgcc-s1 \
  libstdc++6 \
  zlib1g
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
| `compileiq-bootstrap` | First-time setup, socket timeout, or before any other skill |
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
| CIQ_SOCKET_TIMEOUT | 20 | int | Controls how long CompileIQ waits for a core response. If you experience timeouts because your search space is too big, consider increasing this value.
| CIQ_KEEP_CACHE | False | bool | If set to True, `.cache` files will not be deleted.
| CIQ_PROCESS_MODE | "forkserver" | str | User can set this to "fork" to better separate processes and deal with threads. `IsoMultiProcessWorker` uses `fork` as the default.
| CIQ_SEARCH_SPACES_DIR | unset | path | Reads compiler search-space `manifest.json` and `.bin` files from a local mirror instead of GitHub.
| CIQ_SEARCH_SPACES_REPO | NVIDIA/CompileIQ | str | GitHub repository used for search-space release lookups, useful for staging or a future dedicated asset repo.
| CIQ_SS_LATEST_TTL_DAYS | 7 | float | Reuses the cached GitHub `latest` search-space release tag for this many days, reducing anonymous REST API calls that can be throttled on shared IPs. Set to `0` to check GitHub every time.
| CIQ_CORE_BINARY | unset | path | Uses an explicit core executable for local core development instead of the bundled binary.
| CIQ_CORE_MANIFEST | unset | path | Optionally verifies `CIQ_CORE_BINARY` against a local core manifest rooted next to the platform directories.

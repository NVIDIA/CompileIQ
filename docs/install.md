# Install Instructions

You can either install through PyPI:

```bash
pip install compileiq
```

Or, build from the [repo](https://github.com/NVIDIA/CompileIQ) yourself:

```bash
pip install -e .
```

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
| CIQ_CORE_BINARY | unset | path | Uses an explicit core executable for local core development instead of the bundled binary.
| CIQ_CORE_MANIFEST | unset | path | Optionally verifies `CIQ_CORE_BINARY` against a local core manifest rooted next to the platform directories.

# CompileIQ Agent Skills

Agent-agnostic skill set for driving CompileIQ optimization campaigns from any
coding agent: Claude Code, Codex, Cursor, GitHub Copilot, Aider, Windsurf, and
anything else that respects [agents.md](https://agents.md) or the
[agentskills.io](https://agentskills.io) `SKILL.md` convention.

## What's here

Seven skills, each in its own directory with a `SKILL.md`:

| Skill | Use when |
|---|---|
| [`compileiq-bootstrap`](compileiq-bootstrap/) | First-time setup, socket timeout, or before any other skill. |
| [`compileiq-booster-pack`](compileiq-booster-pack/) | Try a curated ACF candidate **before** paying for a full search. |
| [`compileiq-search-space`](compileiq-search-space/) | Choose a provider class or pin a search-space release; covers the attention-focused `att` variant. |
| [`compileiq-author-objective`](compileiq-author-objective/) | Write or fix the function passed as `objective_function=`. |
| [`compileiq-run-search`](compileiq-run-search/) | Compose `Search(...)` and call `.start()`. |
| [`compileiq-validate-result`](compileiq-validate-result/) | Validate a winning ACF with Welch's t-test before shipping. |
| [`compileiq-debug`](compileiq-debug/) | Anything unexpected; symptom-indexed cheat sheet. |

Recommended order for a fresh project:

```
bootstrap → booster-pack → (only if no pack helps) author-objective →
run-search → validate-result
```

`debug` is available throughout.

## Install

```bash
# Detect installed agents and mount skills for all of them
bash agent-skills/install.sh

# Or pick specific agents
bash agent-skills/install.sh --agents claude-code,codex,cursor,copilot

# Check whether installation looks healthy
bash agent-skills/install.sh --check

# Uninstall (removes mount points, leaves agent-skills/ intact)
bash agent-skills/install.sh --uninstall
```

Mount targets:

| Agent | Where | Method |
|---|---|---|
| Claude Code | `.claude/skills/<skill>/` | symlink |
| Codex | `${CODEX_HOME:-$HOME/.codex}/skills/<skill>/` | symlink; restart Codex after install |
| Cursor | `.cursor/rules/<skill>.mdc` | render Cursor frontmatter from `SKILL.md` |
| GitHub Copilot | `.github/instructions/<skill>.instructions.md` | render Copilot frontmatter |
| Aider | `.aider.conf.yml` (`read:` list) | edit config (opt-in with `--agents aider`) |
| Windsurf | `.windsurf/rules/<skill>.md` | symlink |

The installer is idempotent. On systems without symlinks (Windows) it falls
back to file copies; `--uninstall` then deletes the copies.

## File layout

```
agent-skills/
├── README.md
├── MANIFEST.md                       # one-line index, versions
├── install.sh                        # cross-agent mounter
├── uninstall.sh                      # alias for install.sh --uninstall
├── compileiq-bootstrap/
│   ├── SKILL.md
│   └── scripts/check_env.sh
├── compileiq-booster-pack/
│   ├── SKILL.md
│   └── scripts/apply_one_acf.sh
├── compileiq-search-space/
│   └── SKILL.md
├── compileiq-author-objective/
│   ├── SKILL.md
│   └── references/templates.md       # NVBench, Triton, Helion, raw PTX, cuTeDSL
├── compileiq-run-search/
│   ├── SKILL.md
│   └── scripts/smoke_search.py
├── compileiq-validate-result/
│   ├── SKILL.md
│   └── scripts/welch_validate.py
└── compileiq-debug/
    ├── SKILL.md
    └── scripts/diagnose_csv.py
```

Every script supports `--self-test` (or `--check`) and exits non-zero on
failure. Run them once after install to confirm the env is sane.

## Authoring conventions

When adding or modifying a skill, every `SKILL.md` follows the same
frontmatter shape:

```yaml
---
name: compileiq-<slug>                    # must match parent directory
description: >                            # ≤1024 chars; trigger words first
  Use when …
when_to_use: |                            # Claude Code extension; safe on others
  - bullets
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA CompileIQ
  domain: compiler-optimization
allowed-tools: Bash Read                  # Claude Code extension
paths: ["**/*.py"]                        # Claude Code extension; ignored by others
---
```

Cross-agent rule of thumb: anything inside `description`, `when_to_use`, and
the body **must** make sense to an agent that doesn't recognize the extension
fields. Stay portable.

Body shape:

1. **# `compileiq-<slug>`** — one-line summary.
2. **## When** — short bullets describing when to invoke and when not to.
3. **## Steps / Body** — the meat. Numbered when sequential.
4. **## Self-test** — how to verify the skill works on a fresh host.
5. **## Gotchas** — non-obvious pitfalls.
6. **## Next** — pointers to the skills that follow.

Keep bodies under ~400 lines; push deep reference material to
`references/<topic>.md`.

## Versioning

Each skill carries a `version:` field in its frontmatter. Bump on any change
that an agent's behavior depends on (new trigger words, dropped sections,
changed API references). The repo-level [`MANIFEST.md`](MANIFEST.md) tracks
versions for quick scan.

## Contributing

- Run every script with `--self-test` before opening a PR.
- Lint frontmatter:
  `python -c "import yaml, pathlib; [yaml.safe_load(open(p).read().split('---')[1]) for p in pathlib.Path('agent-skills').rglob('SKILL.md')]"`
- Update `MANIFEST.md` if you add/rename/delete a skill.
- Don't claim a skill works against an agent you haven't tested.

#!/usr/bin/env bash
# Cross-agent installer for CompileIQ agent-skills.
#
# Usage:
#   install.sh                          # auto-detect installed agents and mount for each
#   install.sh --agents claude-code,codex,cursor,copilot,aider,windsurf
#   install.sh --check                  # report mount status, exit 0 if everything's mounted
#   install.sh --uninstall              # remove mounts; leave agent-skills/ intact
#   install.sh --help
#
# Idempotent. Writes .installed-by-agent-skills markers so --uninstall is surgical.
# Falls back to copy on platforms without symlink support.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS=(compileiq-bootstrap compileiq-booster-pack compileiq-search-space \
        compileiq-author-objective compileiq-run-search \
        compileiq-validate-result compileiq-debug)

AGENTS=""
MODE="install"

usage() {
    sed -nE '2,/^$/ s/^# ?//p' "$0"
    exit 2
}

log()  { printf '%s\n' "$*"; }
ok()   { printf '  OK   %s\n' "$*"; }
warn() { printf '  WARN %s\n' "$*"; }
err()  { printf '  ERR  %s\n' "$*" >&2; }

codex_home() {
    printf '%s\n' "${CODEX_HOME:-$HOME/.codex}"
}

python_bin() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    err "python3 or python is required for this agent target"
    return 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --agents) AGENTS="$2"; shift 2 ;;
        --check) MODE="check"; shift ;;
        --uninstall) MODE="uninstall"; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

# Auto-detect agents based on directories users typically have configured.
auto_detect() {
    local found=()
    [ -d "$REPO_ROOT/.claude" ] || command -v claude >/dev/null 2>&1   && found+=("claude-code")
    [ -d "$(codex_home)" ] || command -v codex >/dev/null 2>&1         && found+=("codex")
    [ -d "$REPO_ROOT/.cursor" ] || command -v cursor >/dev/null 2>&1   && found+=("cursor")
    [ -d "$REPO_ROOT/.github" ] && found+=("copilot")
    [ -f "$REPO_ROOT/.aider.conf.yml" ] && found+=("aider")
    [ -d "$REPO_ROOT/.windsurf" ] || command -v windsurf >/dev/null 2>&1 && found+=("windsurf")
    # Dedup
    printf '%s\n' "${found[@]}" | awk '!seen[$0]++' | paste -sd, -
}

if [ -z "$AGENTS" ]; then
    AGENTS="$(auto_detect)"
fi

if [ -z "$AGENTS" ] && [ "$MODE" != "uninstall" ]; then
    AGENTS="claude-code"
    warn "no agents auto-detected; defaulting to --agents claude-code"
fi

# Convert comma-separated list to space-separated for iteration.
AGENT_LIST="$(printf '%s' "$AGENTS" | tr ',' ' ')"

# ---- Per-agent mount/unmount handlers ----

mount_claude_code() {
    local dst="$REPO_ROOT/.claude/skills"
    mkdir -p "$dst"
    for s in "${SKILLS[@]}"; do
        local src="$SCRIPT_DIR/$s"
        local target="$dst/$s"
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
        fi
        if ln -s "$src" "$target" 2>/dev/null; then
            ok "claude-code: symlinked $s"
        else
            cp -r "$src" "$target"
            ok "claude-code: copied $s (symlink unsupported)"
        fi
    done
    touch "$dst/.installed-by-agent-skills"
}

unmount_claude_code() {
    local dst="$REPO_ROOT/.claude/skills"
    [ -d "$dst" ] || { ok "claude-code: nothing to remove"; return; }
    [ -f "$dst/.installed-by-agent-skills" ] || { warn "claude-code: marker absent; refusing to touch $dst"; return; }
    for s in "${SKILLS[@]}"; do
        rm -rf "$dst/$s"
    done
    rm -f "$dst/.installed-by-agent-skills"
    rmdir "$dst" 2>/dev/null || true
    ok "claude-code: unmounted"
}

check_claude_code() {
    local dst="$REPO_ROOT/.claude/skills"
    local missing=0
    for s in "${SKILLS[@]}"; do
        if [ ! -e "$dst/$s/SKILL.md" ]; then
            err "claude-code: missing $dst/$s/SKILL.md"
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] && ok "claude-code: all $(( ${#SKILLS[@]} )) skills mounted"
    return "$missing"
}

mount_codex() {
    local dst="$(codex_home)/skills"
    local marker="$dst/.installed-by-agent-skills"
    mkdir -p "$dst"
    for s in "${SKILLS[@]}"; do
        local target="$dst/$s"
        if { [ -e "$target" ] || [ -L "$target" ]; } && [ ! -f "$marker" ]; then
            err "codex: $target already exists; refusing to overwrite without $marker"
            return 1
        fi
    done
    for s in "${SKILLS[@]}"; do
        local src="$SCRIPT_DIR/$s"
        local target="$dst/$s"
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
        fi
        if ln -s "$src" "$target" 2>/dev/null; then
            ok "codex: symlinked $s"
        else
            cp -r "$src" "$target"
            ok "codex: copied $s (symlink unsupported)"
        fi
    done
    touch "$marker"
    warn "codex: restart Codex to pick up new skills"
}

unmount_codex() {
    local dst="$(codex_home)/skills"
    local marker="$dst/.installed-by-agent-skills"
    [ -d "$dst" ] || { ok "codex: nothing to remove"; return; }
    [ -f "$marker" ] || { warn "codex: marker absent; refusing to touch $dst"; return; }
    for s in "${SKILLS[@]}"; do
        rm -rf "$dst/$s"
    done
    rm -f "$marker"
    ok "codex: unmounted"
}

check_codex() {
    local dst="$(codex_home)/skills"
    local missing=0
    for s in "${SKILLS[@]}"; do
        if [ ! -e "$dst/$s/SKILL.md" ]; then
            err "codex: missing $dst/$s/SKILL.md"
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] && ok "codex: all $(( ${#SKILLS[@]} )) skills mounted"
    return "$missing"
}

# Render a Cursor .mdc from a SKILL.md. Frontmatter is rewritten into Cursor's
# expected shape; body is preserved verbatim after the front-matter delimiter.
render_cursor_mdc() {
    local skill_dir="$1"
    local output="$2"
    local py
    py="$(python_bin)"
    "$py" - "$skill_dir" "$output" <<'PY'
import sys, pathlib, yaml, re
skill_dir = pathlib.Path(sys.argv[1])
out_path  = pathlib.Path(sys.argv[2])
text = (skill_dir / "SKILL.md").read_text()
m = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)
if not m:
    sys.exit("missing frontmatter")
meta = yaml.safe_load(m.group(1))
body = m.group(2)
globs = meta.get("paths", ["**/*"])
cursor = {
    "description": meta.get("description", "").strip(),
    "globs": globs,
    "alwaysApply": False,
}
front = "---\n" + yaml.safe_dump(cursor, sort_keys=False).strip() + "\n---\n"
out_path.write_text(front + body)
PY
}

mount_cursor() {
    local dst="$REPO_ROOT/.cursor/rules"
    mkdir -p "$dst"
    for s in "${SKILLS[@]}"; do
        local src="$SCRIPT_DIR/$s"
        local target="$dst/$s.mdc"
        render_cursor_mdc "$src" "$target"
        ok "cursor: rendered $s.mdc"
    done
    touch "$dst/.installed-by-agent-skills"
}

unmount_cursor() {
    local dst="$REPO_ROOT/.cursor/rules"
    [ -d "$dst" ] || { ok "cursor: nothing to remove"; return; }
    [ -f "$dst/.installed-by-agent-skills" ] || { warn "cursor: marker absent; refusing to touch $dst"; return; }
    for s in "${SKILLS[@]}"; do
        rm -f "$dst/$s.mdc"
    done
    rm -f "$dst/.installed-by-agent-skills"
    rmdir "$dst" 2>/dev/null || true
    ok "cursor: unmounted"
}

check_cursor() {
    local dst="$REPO_ROOT/.cursor/rules"
    local missing=0
    for s in "${SKILLS[@]}"; do
        if [ ! -f "$dst/$s.mdc" ]; then
            err "cursor: missing $dst/$s.mdc"
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] && ok "cursor: all $(( ${#SKILLS[@]} )) rules rendered"
    return "$missing"
}

# Render a Copilot .instructions.md from SKILL.md (description + applyTo glob).
render_copilot_instructions() {
    local skill_dir="$1"
    local output="$2"
    local py
    py="$(python_bin)"
    "$py" - "$skill_dir" "$output" <<'PY'
import sys, pathlib, yaml, re
skill_dir = pathlib.Path(sys.argv[1])
out_path  = pathlib.Path(sys.argv[2])
text = (skill_dir / "SKILL.md").read_text()
m = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)
if not m:
    sys.exit("missing frontmatter")
meta = yaml.safe_load(m.group(1))
body = m.group(2)
paths = meta.get("paths", ["**/*"])
copilot = {
    "description": meta.get("description", "").strip(),
    "applyTo": ",".join(paths) if isinstance(paths, list) else str(paths),
}
front = "---\n" + yaml.safe_dump(copilot, sort_keys=False).strip() + "\n---\n"
out_path.write_text(front + body)
PY
}

mount_copilot() {
    local dst="$REPO_ROOT/.github/instructions"
    mkdir -p "$dst"
    for s in "${SKILLS[@]}"; do
        local src="$SCRIPT_DIR/$s"
        local target="$dst/$s.instructions.md"
        render_copilot_instructions "$src" "$target"
        ok "copilot: rendered $s.instructions.md"
    done
    touch "$dst/.installed-by-agent-skills"
}

unmount_copilot() {
    local dst="$REPO_ROOT/.github/instructions"
    [ -d "$dst" ] || { ok "copilot: nothing to remove"; return; }
    [ -f "$dst/.installed-by-agent-skills" ] || { warn "copilot: marker absent; refusing to touch $dst"; return; }
    for s in "${SKILLS[@]}"; do
        rm -f "$dst/$s.instructions.md"
    done
    rm -f "$dst/.installed-by-agent-skills"
    rmdir "$dst" 2>/dev/null || true
    ok "copilot: unmounted"
}

check_copilot() {
    local dst="$REPO_ROOT/.github/instructions"
    local missing=0
    for s in "${SKILLS[@]}"; do
        if [ ! -f "$dst/$s.instructions.md" ]; then
            err "copilot: missing $dst/$s.instructions.md"
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] && ok "copilot: all $(( ${#SKILLS[@]} )) instructions rendered"
    return "$missing"
}

mount_windsurf() {
    local dst="$REPO_ROOT/.windsurf/rules"
    mkdir -p "$dst"
    for s in "${SKILLS[@]}"; do
        local src="$SCRIPT_DIR/$s/SKILL.md"
        local target="$dst/$s.md"
        if ln -sf "$src" "$target" 2>/dev/null; then
            ok "windsurf: symlinked $s.md"
        else
            cp "$src" "$target"
            ok "windsurf: copied $s.md"
        fi
    done
    touch "$dst/.installed-by-agent-skills"
}

unmount_windsurf() {
    local dst="$REPO_ROOT/.windsurf/rules"
    [ -d "$dst" ] || { ok "windsurf: nothing to remove"; return; }
    [ -f "$dst/.installed-by-agent-skills" ] || { warn "windsurf: marker absent; refusing to touch $dst"; return; }
    for s in "${SKILLS[@]}"; do
        rm -f "$dst/$s.md"
    done
    rm -f "$dst/.installed-by-agent-skills"
    rmdir "$dst" 2>/dev/null || true
    ok "windsurf: unmounted"
}

check_windsurf() {
    local dst="$REPO_ROOT/.windsurf/rules"
    local missing=0
    for s in "${SKILLS[@]}"; do
        if [ ! -e "$dst/$s.md" ]; then
            err "windsurf: missing $dst/$s.md"
            missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] && ok "windsurf: all $(( ${#SKILLS[@]} )) rules mounted"
    return "$missing"
}

mount_aider() {
    local conf="$REPO_ROOT/.aider.conf.yml"
    if [ ! -f "$conf" ]; then
        warn "aider: $conf does not exist; create it with at least 'read: []' before re-running"
        return
    fi
    local py
    py="$(python_bin)"
    "$py" - "$conf" "${SKILLS[@]}" <<'PY'
import sys, yaml, pathlib
conf_path = pathlib.Path(sys.argv[1])
skills    = sys.argv[2:]
data = yaml.safe_load(conf_path.read_text()) or {}
existing = data.get("read", []) or []
if isinstance(existing, str):
    existing = [existing]
agent_skills_lines = [f"agent-skills/{s}/SKILL.md" for s in skills]
merged = [r for r in existing if not r.startswith("agent-skills/")] + agent_skills_lines
data["read"] = merged
# Marker so uninstall is safe
data.setdefault("_agent_skills_managed", True)
conf_path.write_text(yaml.safe_dump(data, sort_keys=False))
print(f"  OK   aider: updated read: list in {conf_path}")
PY
}

unmount_aider() {
    local conf="$REPO_ROOT/.aider.conf.yml"
    [ -f "$conf" ] || { ok "aider: nothing to remove"; return; }
    local py
    py="$(python_bin)"
    "$py" - "$conf" <<'PY'
import sys, yaml, pathlib
conf_path = pathlib.Path(sys.argv[1])
data = yaml.safe_load(conf_path.read_text()) or {}
if not data.get("_agent_skills_managed"):
    print(f"  WARN aider: {conf_path} was not managed by agent-skills; refusing to edit")
    sys.exit(0)
existing = data.get("read", []) or []
if isinstance(existing, str):
    existing = [existing]
remaining = [r for r in existing if not r.startswith("agent-skills/")]
if remaining:
    data["read"] = remaining
else:
    data.pop("read", None)
data.pop("_agent_skills_managed", None)
conf_path.write_text(yaml.safe_dump(data, sort_keys=False))
print(f"  OK   aider: removed agent-skills entries from {conf_path}")
PY
}

check_aider() {
    local conf="$REPO_ROOT/.aider.conf.yml"
    [ -f "$conf" ] || { warn "aider: $conf missing; nothing mounted"; return 0; }
    local py
    py="$(python_bin)"
    "$py" - "$conf" "${SKILLS[@]}" <<'PY'
import sys, yaml, pathlib
conf_path = pathlib.Path(sys.argv[1])
skills    = sys.argv[2:]
data = yaml.safe_load(conf_path.read_text()) or {}
existing = data.get("read", []) or []
if isinstance(existing, str):
    existing = [existing]
missing = [s for s in skills if f"agent-skills/{s}/SKILL.md" not in existing]
if missing:
    print(f"  ERR  aider: missing read: entries for {missing}")
    sys.exit(len(missing))
print(f"  OK   aider: all {len(skills)} entries present in {conf_path}")
PY
}

# ---- Main ----

log "agent-skills install.sh ($MODE mode)"
log "agents: $AGENTS"
log "skills source: $SCRIPT_DIR"

EXIT=0
for a in $AGENT_LIST; do
    log ""
    log "--- $a ---"
    case "$a" in
        claude-code) case "$MODE" in install) mount_claude_code ;; uninstall) unmount_claude_code ;; check) check_claude_code || EXIT=$((EXIT + $?)) ;; esac ;;
        codex)       case "$MODE" in install) mount_codex       ;; uninstall) unmount_codex       ;; check) check_codex       || EXIT=$((EXIT + $?)) ;; esac ;;
        cursor)      case "$MODE" in install) mount_cursor      ;; uninstall) unmount_cursor      ;; check) check_cursor      || EXIT=$((EXIT + $?)) ;; esac ;;
        copilot)     case "$MODE" in install) mount_copilot     ;; uninstall) unmount_copilot     ;; check) check_copilot     || EXIT=$((EXIT + $?)) ;; esac ;;
        aider)       case "$MODE" in install) mount_aider       ;; uninstall) unmount_aider       ;; check) check_aider       || EXIT=$((EXIT + $?)) ;; esac ;;
        windsurf)    case "$MODE" in install) mount_windsurf    ;; uninstall) unmount_windsurf    ;; check) check_windsurf    || EXIT=$((EXIT + $?)) ;; esac ;;
        *)           warn "$a: unknown agent"; EXIT=$((EXIT + 1)) ;;
    esac
done

log ""
if [ "$EXIT" -eq 0 ]; then
    log "$MODE complete."
else
    log "$MODE finished with $EXIT issue(s)."
fi
exit "$EXIT"

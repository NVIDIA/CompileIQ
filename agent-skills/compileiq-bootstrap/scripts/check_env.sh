#!/usr/bin/env bash
# compileiq-bootstrap self-test.
# Exit code = number of failed checks. Prints the exact next-step command for each failure.
#
# Usage:
#   bash check_env.sh           # run all checks
#   bash check_env.sh --check   # same; alias for tooling

set -u

FAIL=0

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n        fix: %s\n' "$1" "$2"; FAIL=$((FAIL + 1)); }

step() { printf '\n[%s] %s\n' "$1" "$2"; }
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
    return 1
}
PYTHON_BIN="$(python_bin || true)"

step 1 "CUDA toolkit 13.3+"
if command -v nvcc >/dev/null 2>&1; then
    NVCC_REL="$(nvcc --version 2>/dev/null | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -1)"
    if [ -n "$NVCC_REL" ] && awk -v v="$NVCC_REL" 'BEGIN { split(v,a,"."); exit !(a[1]>13 || (a[1]==13 && a[2]>=3)) }'; then
        pass "nvcc release $NVCC_REL"
    else
        fail "nvcc release '$NVCC_REL' < 13.3" "install or upgrade CUDA toolkit to 13.3+"
    fi
else
    fail "nvcc not in PATH" "export PATH=/usr/local/cuda/bin:\$PATH"
fi

if command -v ptxas >/dev/null 2>&1; then
    PTXAS_REL="$(ptxas --version 2>/dev/null | sed -nE 's/.*V([0-9]+\.[0-9]+).*/\1/p' | head -1)"
    if [ -n "$PTXAS_REL" ] && awk -v v="$PTXAS_REL" 'BEGIN { split(v,a,"."); exit !(a[1]>13 || (a[1]==13 && a[2]>=3)) }'; then
        pass "ptxas V$PTXAS_REL"
    else
        fail "ptxas version '$PTXAS_REL' < 13.3" "ensure the matching CUDA toolkit ships ptxas 13.3+"
    fi
else
    fail "ptxas not in PATH" "export PATH=/usr/local/cuda/bin:\$PATH"
fi

step 2 "GPU visible"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    if [ -n "$GPU_INFO" ]; then
        pass "GPU: $GPU_INFO"
    else
        fail "nvidia-smi found no GPU" "check 'nvidia-smi -L' and driver installation"
    fi
else
    fail "nvidia-smi not in PATH" "install NVIDIA driver; ensure nvidia-smi on PATH"
fi

step 3 "CompileIQ imports resolve"
if [ -n "$PYTHON_BIN" ]; then
    IMPORT_OUTPUT="$("$PYTHON_BIN" -c "
from compileiq.ciq import Search
from compileiq.types import INVALID_SCORE, BASELINE_DNA, WorkerTypes, ProblemType, SearchConfiguration
from compileiq.search_spaces.compilers import PtxasSearchSpace, NvccSearchSpace, LocalSearchSpaceBin
from compileiq.utils.helpers import save_compiler_config, load_compiler_config
from compileiq.worker import MultiProcessWorker, IsoMultiProcessWorker, RayWorker, AsyncWorker
print('imports OK')
" 2>&1)"
else
    IMPORT_OUTPUT="python3 or python not found"
fi
if printf '%s' "$IMPORT_OUTPUT" | grep -q "imports OK"; then
    pass "all public imports resolve"
else
    fail "compileiq import failed" "pip install compileiq   (or 'pip install -e .' from a source checkout)"
    printf '        diagnostic: %s\n' "$(printf '%s' "$IMPORT_OUTPUT" | tail -3)"
fi

step 4 "Search-space resolution round-trip"
if [ -n "$PYTHON_BIN" ] && command -v timeout >/dev/null 2>&1; then
    RESOLVE_OUTPUT="$(timeout 60 "$PYTHON_BIN" -c "
from compileiq.search_spaces.compilers import PtxasSearchSpace
p = PtxasSearchSpace().retrieve()
assert p.exists() and p.stat().st_size > 0, p
print(f'resolved: {p}')
" 2>&1)"
elif [ -n "$PYTHON_BIN" ]; then
    RESOLVE_OUTPUT="$("$PYTHON_BIN" -c "
from compileiq.search_spaces.compilers import PtxasSearchSpace
p = PtxasSearchSpace().retrieve()
assert p.exists() and p.stat().st_size > 0, p
print(f'resolved: {p}')
" 2>&1)"
else
    RESOLVE_OUTPUT="python3 or python not found"
fi
if printf '%s' "$RESOLVE_OUTPUT" | grep -q "^resolved:"; then
    pass "$(printf '%s' "$RESOLVE_OUTPUT" | grep '^resolved:')"
else
    fail "PtxasSearchSpace().retrieve() failed or timed out" "set CIQ_SEARCH_SPACES_DIR=/path/to/local/mirror if air-gapped; otherwise check network to github.com"
    printf '        diagnostic: %s\n' "$(printf '%s' "$RESOLVE_OUTPUT" | tail -3)"
fi

printf '\n'
if [ "$FAIL" -eq 0 ]; then
    printf 'All checks passed.\n'
    exit 0
else
    printf '%d check(s) failed.\n' "$FAIL"
    exit "$FAIL"
fi

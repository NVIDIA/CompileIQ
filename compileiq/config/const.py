import os
from pathlib import Path


ENV_VARS_TRUE_VALUES = {"1", "YES", "TRUE"}


def _is_true(val: str):
    return val.upper() in ENV_VARS_TRUE_VALUES


## Manager required files
_CACHE_DIR = os.path.join(Path.home(), ".cache", "compileiq")

KEEP_CACHE_FILES = _is_true(os.getenv("CIQ_KEEP_CACHE", "0"))


def unsafe_allow_checkpoint_provenance_mismatch() -> bool:
    """Read the development-only checkpoint provenance override at call time."""

    return _is_true(os.getenv("CIQ_UNSAFE_ALLOW_CHECKPOINT_PROVENANCE_MISMATCH", "0"))

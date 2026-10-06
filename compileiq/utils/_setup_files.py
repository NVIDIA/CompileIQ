import pathlib
import shutil
import warnings
from compileiq.config.const import _CACHE_DIR


def clear_cache():
    """
    Warning: This will remove the entire .cache/compileiq data
    """
    warnings.warn(
        "This function removes the entire .cache folder for compileiq. "
        "You may lose other experiments data or break running experiments.",
        stacklevel=2,
    )
    if pathlib.Path(_CACHE_DIR).exists():
        shutil.rmtree(_CACHE_DIR)

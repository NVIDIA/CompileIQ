import os
from pathlib import Path


ENV_VARS_TRUE_VALUES = {"1", "YES", "TRUE"}


def _is_true(val: str):
    return val.upper() in ENV_VARS_TRUE_VALUES


## Manager required files
_ROOT_PATH = Path(__file__).parent.parent.parent
_CACHE_DIR = os.path.join(Path.home(), ".cache", "compileiq")

MAIN_CONFIG_FILENAME = "main_config.json"
SEARCH_SPACE_CONFIG_FILENAME = "search_space.json"
CORE_PACKAGES = ["sci", "graph"]

KEEP_CACHE_FILES = _is_true(os.getenv("CIQ_KEEP_CACHE", "0"))

## Socket Values
SOCKET_TIMEOUT = int(os.getenv("CIQ_SOCKET_TIMEOUT", "20"))  # in seconds
MAX_RETRIES = 10
MAX_BYTES: int = 4096 * 10

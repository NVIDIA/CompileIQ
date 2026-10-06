"""Private: combine encrypted ACFs inside ``libciq``. Not a public API.

It combines per-compiler encrypted ACFs, such as one for NVVM and one for PTXAS, into the
single controls file that ``nvcc --apply-controls`` reads. A later change adds a public way
to produce that file. Its names and signatures can change in any release.

``libciq`` is the closed library that reads protected search spaces and formats encrypted
samples. Complete wheels link it into the compiled native runtime on every platform, and
the merge runs there. ``CIQ_LIBCIQ_PATH`` names a shared ``libciq`` for development and
takes precedence over the runtime. Inputs and output are always ciphertext, so no protected
plaintext reaches Python.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import os
import threading
import warnings
from pathlib import Path
from typing import Callable, Sequence

LIBCIQ_PATH_ENV_VAR = "CIQ_LIBCIQ_PATH"
NATIVE_RUNTIME_MODULE = "compileiq_native_runtime"

# libciq merge_controls return codes.
MERGE_SUCCESS = 0
MERGE_ERRORS = {
    -20: (
        "invalid arguments (no inputs, more than 64 inputs, an empty or oversized input, "
        "or an unsupported component in ensure_components)"
    ),
    -21: (
        "an input is not an encrypted ACF this CompileIQ build can read, or its CUDA "
        "version cannot be used to version an ensured section"
    ),
    -22: "two inputs carry controls for the same compiler",
    -23: "inputs use different ACF interface versions",
    -24: "inputs target different CUDA toolkit versions",
    -25: "re-encrypting the merged ACF failed",
    -26: "internal error in libciq",
}

_BYTE_POINTER = ctypes.POINTER(ctypes.c_ubyte)
# Calls are serialized: older libciq builds race on concurrent first use of the cipher.
_CALL_LOCK = threading.Lock()

_UNAVAILABLE_MESSAGE = (
    "This CompileIQ installation cannot merge ACFs: it has no compiled native runtime "
    "with merge_controls. Install a complete CompileIQ wheel of a release that supports "
    "merging."
)


class AcfMergeError(ValueError):
    """The ACFs cannot be merged. ``code`` is the libciq ``merge_controls`` return code."""

    def __init__(self, code: int):
        detail = MERGE_ERRORS.get(code, "unknown error")
        super().__init__(f"Cannot merge ACFs: {detail} (libciq code {code}).")
        self.code = code


class LibciqUnavailableError(RuntimeError):
    """This installation has no usable ``libciq`` for the requested operation."""


def _configure(library: ctypes.CDLL) -> ctypes.CDLL:
    try:
        merge_controls = library.merge_controls
        free_buffer = library.free_buffer
    except AttributeError as exc:
        raise LibciqUnavailableError(
            f"The libciq named by {LIBCIQ_PATH_ENV_VAR} does not export merge_controls."
        ) from exc
    merge_controls.restype = _BYTE_POINTER
    merge_controls.argtypes = [
        ctypes.POINTER(_BYTE_POINTER),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int),
    ]
    free_buffer.restype = None
    free_buffer.argtypes = [_BYTE_POINTER]
    return library


@functools.lru_cache(maxsize=4)
def _load_shared_libciq(path: str) -> ctypes.CDLL:
    return _configure(ctypes.CDLL(path))


def load_libciq() -> ctypes.CDLL:
    """Load the developer ``libciq`` named by ``CIQ_LIBCIQ_PATH``."""

    override = os.environ.get(LIBCIQ_PATH_ENV_VAR)
    if not override:
        raise LibciqUnavailableError(f"{LIBCIQ_PATH_ENV_VAR} is not set.")
    path = Path(override).expanduser()
    if not path.is_file():
        raise LibciqUnavailableError(f"{LIBCIQ_PATH_ENV_VAR} points to a missing file: {path}")
    warnings.warn(
        f"{LIBCIQ_PATH_ENV_VAR} is set; merging ACFs with a developer libciq.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _load_shared_libciq(str(path))


def _native_merge() -> Callable[..., tuple[int, bytes]] | None:
    """Return the native runtime's ``merge_controls`` when this installation has one."""

    try:
        runtime = importlib.import_module(NATIVE_RUNTIME_MODULE)
    except ImportError:
        return None
    return getattr(runtime, "merge_controls", None)


def require_merge_support() -> None:
    """Raise ``LibciqUnavailableError`` unless ``merge_controls`` can run here.

    Checks without merging, so a search can fail at setup rather than in every
    objective call.
    """

    if os.environ.get(LIBCIQ_PATH_ENV_VAR):
        load_libciq()
    elif _native_merge() is None:
        raise LibciqUnavailableError(_UNAVAILABLE_MESSAGE)


def merge_controls(
    acfs: list[bytes],
    library: ctypes.CDLL | None = None,
    ensure_components: Sequence[str] = (),
) -> bytes:
    """Merge encrypted ACFs into one encrypted ACF through libciq ``merge_controls``.

    ``ensure_components`` names components to hold with an empty section when no input
    carries them. Uses the compiled native runtime unless ``library`` is given or
    ``CIQ_LIBCIQ_PATH`` names a developer libciq.
    """

    if not acfs:
        raise AcfMergeError(-20)
    ensure = [str(name) for name in ensure_components]
    if library is None and not os.environ.get(LIBCIQ_PATH_ENV_VAR):
        native = _native_merge()
        if native is None:
            raise LibciqUnavailableError(_UNAVAILABLE_MESSAGE)
        with _CALL_LOCK:
            code, merged = native(list(acfs), ensure)
        if code != MERGE_SUCCESS or not merged:
            raise AcfMergeError(code if code else -26)
        return bytes(merged)
    library = load_libciq() if library is None else library
    buffers = [(ctypes.c_ubyte * len(acf)).from_buffer_copy(acf) for acf in acfs]
    pointers = (_BYTE_POINTER * len(buffers))(
        *[ctypes.cast(buffer, _BYTE_POINTER) for buffer in buffers]
    )
    lengths = (ctypes.c_size_t * len(acfs))(*[len(acf) for acf in acfs])
    names = (ctypes.c_char_p * max(len(ensure), 1))(*[name.encode("utf-8") for name in ensure])
    out_len = ctypes.c_size_t(0)
    code = ctypes.c_int(MERGE_SUCCESS)

    with _CALL_LOCK:
        result = library.merge_controls(
            pointers,
            lengths,
            len(acfs),
            names if ensure else None,
            len(ensure),
            ctypes.byref(out_len),
            ctypes.byref(code),
        )
        if not result:
            raise AcfMergeError(code.value if code.value else -26)
        try:
            merged = ctypes.string_at(result, out_len.value)
        finally:
            library.free_buffer(result)
    if code.value != MERGE_SUCCESS or not merged:
        raise AcfMergeError(code.value if code.value else -26)
    return merged


def merge_acfs(
    acfs: Sequence[str | bytes | os.PathLike[str]],
    ensure_components: Sequence[str] = ("nvvm",),
) -> str:
    """Merge per-compiler ACFs into one ACF that carries every compiler's controls.

    Private plumbing; not part of the public API.

    The inputs must carry controls for different compilers and target the same CUDA
    toolkit major.minor version. Each compiler's controls, including its exact tool
    version, are copied unchanged. A compiler whose knobs were all knocked out keeps an
    empty section, which compilers treat as "use defaults".

    Args:
        acfs: ACFs as hex strings (the form objective functions receive and results
            store), raw bytes, or paths to ``.acf`` files. One input is enough.
        ensure_components: Compilers to add as an empty section when no input carries
            controls for them. Only ``"nvvm"`` is supported; the section carries the NVVM
            version that the inputs' CUDA toolkit expects, which NVVM in CTK 13.3 and
            13.4 requires. An input's own NVVM section is never changed. Pass ``()`` to
            add nothing.

    Returns:
        The merged ACF as a hex string.

    Raises:
        AcfMergeError: The inputs cannot be merged, for example two inputs carry the
            same compiler or target different CUDA versions.
        LibciqUnavailableError: This installation has no libciq that can merge.
    """

    if isinstance(acfs, (str, bytes, bytearray, os.PathLike)):
        raise TypeError("acfs must be a sequence of ACFs, not a single ACF")
    if isinstance(ensure_components, str):
        raise TypeError("ensure_components must be a sequence of names, not a single name")

    blobs: list[bytes] = []
    for index, acf in enumerate(acfs):
        if isinstance(acf, (bytes, bytearray)):
            blobs.append(bytes(acf))
        elif isinstance(acf, str):
            try:
                blobs.append(bytes.fromhex(acf))
            except ValueError as exc:
                raise ValueError(
                    f"acfs[{index}] is not a hex-encoded ACF; pass a pathlib.Path to "
                    "read an ACF file"
                ) from exc
        elif isinstance(acf, os.PathLike):
            with open(acf, "rb") as f:
                blobs.append(f.read())
        else:
            raise TypeError(
                f"acfs[{index}] must be a hex string, bytes, or a path, "
                f"not {type(acf).__name__}"
            )
    return merge_controls(blobs, ensure_components=list(ensure_components)).hex()

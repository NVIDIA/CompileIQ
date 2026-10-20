"""
End-to-end ACF translation against a real libciq and real search spaces.

Runs only when both are available:

    CIQ_LIBCIQ_PATH=/path/to/libciq.so       # built with translate_acf
    CIQ_SEARCH_SPACES_DIR=/path/to/mirror    # manifest.json + ptxas 13.3 and 13.4 .bin files

If a ``ptxas`` matching the target CTK is on PATH, the translated ACF is also applied
with ``ptxas --apply-controls``.
"""

import ctypes
import json
import os
import re
import shutil
import subprocess

import pytest

from compileiq import acf
from compileiq.utils._libciq import LIBCIQ_PATH_ENV_VAR
from compileiq.search_spaces.resolver import resolve

SOURCE_CTK = "13.3"
TARGET_CTK = "13.4"

PTX = """\
.version 8.0
.target sm_80
.address_size 64

.visible .entry noop()
{
    ret;
}
"""


def _libciq():
    path = os.environ.get(LIBCIQ_PATH_ENV_VAR)
    if not path:
        pytest.skip(f"{LIBCIQ_PATH_ENV_VAR} is not set")
    lib = ctypes.CDLL(path)
    if not hasattr(lib, "translate_acf"):
        pytest.skip(f"{path} does not export translate_acf")
    lib.extract_search_space.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.extract_search_space.restype = ctypes.c_void_p
    lib.format_sample.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.format_sample.restype = ctypes.c_void_p
    lib.free_buffer.argtypes = [ctypes.c_void_p]
    return lib


def _search_space(compiler, version, variant="default"):
    if not os.environ.get("CIQ_SEARCH_SPACES_DIR"):
        pytest.skip("CIQ_SEARCH_SPACES_DIR is not set")
    try:
        return resolve(compiler, version, variant)
    except LookupError:
        pytest.skip(f"No {compiler} {version} {variant} search space in CIQ_SEARCH_SPACES_DIR")


def _generate_acf(lib, search_space_path) -> bytes:
    """Builds an ACF that sets every control in the search space to its lowest value."""
    rc = ctypes.c_int(0)
    flat = lib.extract_search_space(os.fsencode(search_space_path), ctypes.pointer(rc))
    assert flat, f"extract_search_space failed with {rc.value}"
    try:
        space = json.loads(ctypes.string_at(flat).decode())
    finally:
        lib.free_buffer(flat)

    sample = {}
    for name, spec in space.items():
        if spec["type"] == "range":
            sample[name] = spec["low"]
        elif spec["type"] == "enum":
            sample[name] = spec["values"][0]
        else:
            sample[name] = spec["value"]

    out_len = ctypes.c_size_t(0)
    buffer = lib.format_sample(
        os.fsencode(search_space_path),
        json.dumps(sample).encode(),
        ctypes.pointer(out_len),
        ctypes.pointer(rc),
    )
    assert buffer, f"format_sample failed with {rc.value}"
    try:
        return ctypes.string_at(buffer, out_len.value)
    finally:
        lib.free_buffer(buffer)


def _ptxas_version():
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return None
    output = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout
    match = re.search(r"release (\d+\.\d+)", output)
    return match.group(1) if match else None


@pytest.fixture(scope="module")
def source_acf():
    lib = _libciq()
    return _generate_acf(lib, _search_space("ptxas", SOURCE_CTK))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_translate_ptxas_acf_forward_and_back(source_acf):
    _search_space("ptxas", TARGET_CTK)

    forward = acf.translate_compiler_config(source_acf.hex(), TARGET_CTK)
    assert forward and forward != source_acf.hex()

    back = acf.translate_compiler_config(forward, SOURCE_CTK)
    assert back


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_translate_rejects_garbage():
    _libciq()
    _search_space("ptxas", TARGET_CTK)

    with pytest.raises(acf.AcfTranslationError) as exc_info:
        acf.translate_compiler_config((b"\x41" * 64).hex(), TARGET_CTK)
    assert exc_info.value.code == -30


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ptxas_accepts_translated_acf(source_acf, tmp_path):
    if _ptxas_version() != TARGET_CTK:
        pytest.skip(f"ptxas {TARGET_CTK} is not on PATH")

    translated = bytes.fromhex(acf.translate_compiler_config(source_acf.hex(), TARGET_CTK))
    acf_path = tmp_path / "translated.acf"
    acf_path.write_bytes(translated)
    ptx_path = tmp_path / "noop.ptx"
    ptx_path.write_text(PTX)

    result = subprocess.run(
        [
            "ptxas",
            "-arch=sm_80",
            "--apply-controls",
            str(acf_path),
            str(ptx_path),
            "-o",
            str(tmp_path / "noop.cubin"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # The untranslated 13.3 ACF is rejected by ptxas 13.4, which is why translation exists
    source_path = tmp_path / "source.acf"
    source_path.write_bytes(source_acf)
    result = subprocess.run(
        ["ptxas", "-arch=sm_80", "--apply-controls", str(source_path), str(ptx_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_translate_through_the_native_runtime(source_acf, monkeypatch):
    """The production path: no developer libciq, translation runs in the native runtime."""
    runtime = pytest.importorskip("compileiq_native_runtime")
    if not hasattr(runtime, "translate_acf"):
        pytest.skip("the installed native runtime does not export translate_acf")
    _search_space("ptxas", TARGET_CTK)
    monkeypatch.delenv(LIBCIQ_PATH_ENV_VAR, raising=False)

    forward = acf.translate_compiler_config(source_acf.hex(), TARGET_CTK)
    assert forward and forward != source_acf.hex()
    assert acf.translate_compiler_config(forward, SOURCE_CTK)

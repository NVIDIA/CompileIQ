"""
Tests for the private ACF merge plumbing in compileiq.utils._libciq.

A fake libciq built from ctypes callbacks exercises the real C calling convention
(the CIQ_LIBCIQ_PATH development path) without native code or encrypted fixtures. A
fake native runtime covers the default path. The optional real-runtime test runs only
where the compiled native runtime exports merge_controls.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
import types

import pytest

import compileiq.utils._libciq as libciq
from compileiq.utils._libciq import AcfMergeError, LibciqUnavailableError, merge_acfs
from compileiq.utils.helpers import save_compiler_config

_BYTE_POINTER = ctypes.POINTER(ctypes.c_ubyte)
_MERGE_FN = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(_BYTE_POINTER),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
)
_FREE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class FakeLibciq:
    """Concatenates inputs plus "+<ensured names>"; rejects b"BAD" (-21) and repeats (-22)."""

    def __init__(self):
        self.live: dict[int, ctypes.Array] = {}
        self.calls: list[list[bytes]] = []
        self.ensured: list[list[bytes]] = []
        self.freed: list[int] = []
        self.merge_controls = _MERGE_FN(self._merge)
        self.free_buffer = _FREE_FN(self._free)

    def _merge(self, acfs, lens, count, names, name_count, out_len, code):
        blobs = [ctypes.string_at(acfs[i], lens[i]) for i in range(count)]
        ensured = [names[i] for i in range(name_count)]
        self.calls.append(blobs)
        self.ensured.append(ensured)
        out_len[0] = 0
        if any(blob.startswith(b"BAD") for blob in blobs):
            code[0] = -21
            return None
        if len(set(blobs)) != len(blobs):
            code[0] = -22
            return None
        merged = b"MERGED:" + b"|".join(blobs) + (b"+" + b",".join(ensured) if ensured else b"")
        buffer = ctypes.create_string_buffer(merged, len(merged))
        address = ctypes.addressof(buffer)
        self.live[address] = buffer
        out_len[0] = len(merged)
        code[0] = 0
        return address

    def _free(self, address):
        self.freed.append(address)
        self.live.pop(address, None)


@pytest.fixture
def fake_libciq(monkeypatch):
    """A developer libciq selected through CIQ_LIBCIQ_PATH."""

    fake = FakeLibciq()
    monkeypatch.setenv(libciq.LIBCIQ_PATH_ENV_VAR, "fake-libciq")
    monkeypatch.setattr(libciq, "_native_merge", lambda: None)
    monkeypatch.setattr(libciq, "load_libciq", lambda: fake)
    return fake


class FakeNativeRuntime(types.ModuleType):
    """Stands in for the compiled runtime's merge_controls -> (code, bytes) contract."""

    def __init__(self):
        super().__init__(libciq.NATIVE_RUNTIME_MODULE)
        self.calls: list[list[bytes]] = []
        self.ensured: list[list[str]] = []

    def merge_controls(self, acfs, ensure_components=()):
        self.calls.append(list(acfs))
        self.ensured.append(list(ensure_components))
        if len(set(acfs)) != len(acfs):
            return (-22, b"")
        suffix = ("+" + ",".join(ensure_components)).encode() if ensure_components else b""
        return (0, b"NATIVE:" + b"|".join(acfs) + suffix)


class TestMergeCompilerConfigs:
    def test_accepts_hex_bytes_and_paths_in_order(self, fake_libciq, tmp_path):
        acf_file = tmp_path / "ptxas.acf"
        save_compiler_config(str(acf_file), b"ptxas".hex())

        merged = merge_acfs([b"nvvm".hex(), b"tileir", acf_file])

        assert bytes.fromhex(merged) == b"MERGED:nvvm|tileir|ptxas+nvvm"
        assert fake_libciq.calls == [[b"nvvm", b"tileir", b"ptxas"]]
        assert fake_libciq.ensured == [[b"nvvm"]]

    def test_single_ptxas_acf_gets_an_ensured_nvvm_section(self, fake_libciq):
        merged = merge_acfs([b"ptxas"])
        assert bytes.fromhex(merged) == b"MERGED:ptxas+nvvm"

    def test_ensure_components_can_be_changed_or_disabled(self, fake_libciq):
        assert bytes.fromhex(merge_acfs([b"a", b"b"], ensure_components=())) == (
            b"MERGED:a|b"
        )
        assert bytes.fromhex(
            merge_acfs([b"a"], ensure_components=["nvvm"])
        ) == b"MERGED:a+nvvm"
        with pytest.raises(TypeError, match="single name"):
            merge_acfs([b"a"], ensure_components="nvvm")

    def test_result_round_trips_through_save_compiler_config(self, fake_libciq, tmp_path):
        merged = merge_acfs([b"nvvm", b"ptxas"])
        out = tmp_path / "merged.acf"
        save_compiler_config(str(out), merged)
        assert out.read_bytes() == b"MERGED:nvvm|ptxas+nvvm"

    def test_frees_the_library_buffer(self, fake_libciq):
        merge_acfs([b"nvvm", b"ptxas"])
        assert len(fake_libciq.freed) == 1
        assert fake_libciq.live == {}

    @pytest.mark.parametrize(
        ("acfs", "code", "text"),
        [
            ([b"nvvm", b"BADptxas"], -21, "not an encrypted ACF"),
            ([b"nvvm", b"nvvm"], -22, "same compiler"),
        ],
    )
    def test_maps_libciq_errors(self, fake_libciq, acfs, code, text):
        with pytest.raises(AcfMergeError) as excinfo:
            merge_acfs(acfs)
        assert excinfo.value.code == code
        assert text in str(excinfo.value)
        assert fake_libciq.freed == []

    def test_error_is_a_value_error(self):
        assert issubclass(AcfMergeError, ValueError)

    def test_empty_input_fails_without_calling_libciq(self, fake_libciq):
        with pytest.raises(AcfMergeError) as excinfo:
            merge_acfs([])
        assert excinfo.value.code == -20
        assert fake_libciq.calls == []

    def test_rejects_non_hex_strings(self, fake_libciq):
        with pytest.raises(ValueError, match="pathlib.Path"):
            merge_acfs(["/tmp/not-hex.acf", b"ptxas".hex()])

    @pytest.mark.parametrize("single", ["abcd", b"abcd", pathlib.Path("x.acf")])
    def test_rejects_a_single_acf_instead_of_a_sequence(self, fake_libciq, single):
        with pytest.raises(TypeError, match="sequence"):
            merge_acfs(single)

    def test_rejects_unsupported_items(self, fake_libciq):
        with pytest.raises(TypeError, match=r"acfs\[1\]"):
            merge_acfs([b"nvvm", 42])  # pyright: ignore[reportArgumentType]


class TestNativeRuntimePath:
    @pytest.fixture
    def native(self, monkeypatch):
        runtime = FakeNativeRuntime()
        monkeypatch.delenv(libciq.LIBCIQ_PATH_ENV_VAR, raising=False)
        monkeypatch.setitem(sys.modules, libciq.NATIVE_RUNTIME_MODULE, runtime)

        def no_shared_library():
            raise AssertionError("the shared libciq must not load when the runtime merges")

        monkeypatch.setattr(libciq, "load_libciq", no_shared_library)
        return runtime

    def test_prefers_the_native_runtime(self, native):
        merged = merge_acfs([b"nvvm", b"ptxas".hex()])
        assert bytes.fromhex(merged) == b"NATIVE:nvvm|ptxas+nvvm"
        assert native.calls == [[b"nvvm", b"ptxas"]]
        assert native.ensured == [["nvvm"]]

    def test_maps_native_error_codes(self, native):
        with pytest.raises(AcfMergeError) as excinfo:
            merge_acfs([b"nvvm", b"nvvm"])
        assert excinfo.value.code == -22

    def test_developer_libciq_override_skips_the_runtime(self, native, monkeypatch, tmp_path):
        fake = FakeLibciq()
        monkeypatch.setenv(libciq.LIBCIQ_PATH_ENV_VAR, str(tmp_path / "libciq.so"))
        monkeypatch.setattr(libciq, "load_libciq", lambda: fake)
        assert bytes.fromhex(merge_acfs([b"a", b"b"])) == b"MERGED:a|b+nvvm"
        assert native.calls == []

    def test_runtime_without_merge_is_unavailable(self, monkeypatch):
        """There is no fallback to a shared libciq bundled with the core."""

        monkeypatch.delenv(libciq.LIBCIQ_PATH_ENV_VAR, raising=False)
        monkeypatch.setitem(
            sys.modules, libciq.NATIVE_RUNTIME_MODULE, types.ModuleType("old_runtime")
        )
        with pytest.raises(LibciqUnavailableError, match="no compiled native runtime"):
            merge_acfs([b"a", b"b"])
        with pytest.raises(LibciqUnavailableError, match="no compiled native runtime"):
            libciq.require_merge_support()

    def test_require_merge_support_accepts_the_runtime(self, native):
        libciq.require_merge_support()
        assert native.calls == []


class TestDeveloperLibciq:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        libciq._load_shared_libciq.cache_clear()
        yield
        libciq._load_shared_libciq.cache_clear()

    def test_requires_the_override(self, monkeypatch):
        monkeypatch.delenv(libciq.LIBCIQ_PATH_ENV_VAR, raising=False)
        with pytest.raises(LibciqUnavailableError, match="not set"):
            libciq.load_libciq()

    def test_override_must_exist(self, monkeypatch, tmp_path):
        monkeypatch.setenv(libciq.LIBCIQ_PATH_ENV_VAR, str(tmp_path / "missing.so"))
        with pytest.raises(LibciqUnavailableError, match="missing file"):
            libciq.load_libciq()
        with pytest.raises(LibciqUnavailableError, match="missing file"):
            libciq.require_merge_support()

    def test_override_loads_and_warns(self, monkeypatch, tmp_path):
        library = tmp_path / "libciq.so"
        library.write_bytes(b"stand-in")
        fake = FakeLibciq()
        monkeypatch.setenv(libciq.LIBCIQ_PATH_ENV_VAR, str(library))
        with monkeypatch.context() as patch:
            patch.setattr(libciq, "_load_shared_libciq", lambda path: fake)
            with pytest.warns(RuntimeWarning, match="developer libciq"):
                assert libciq.load_libciq() is fake

    def test_library_without_merge_controls(self):
        class OldLibciq:
            free_buffer = object()

        with pytest.raises(LibciqUnavailableError, match="does not export merge_controls"):
            libciq._configure(OldLibciq())  # pyright: ignore[reportArgumentType]


def _native_merge_available() -> bool:
    return libciq._native_merge() is not None


@pytest.mark.skipif(
    not _native_merge_available(), reason="no compiled native runtime with merge_controls"
)
def test_native_runtime_rejects_non_acf_bytes(monkeypatch):
    monkeypatch.delenv(libciq.LIBCIQ_PATH_ENV_VAR, raising=False)
    with pytest.raises(AcfMergeError) as excinfo:
        merge_acfs([b"\x00" * 64, b"\x01" * 64])
    assert excinfo.value.code == -21

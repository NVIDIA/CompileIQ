"""
Tests for ACF translation: compileiq.acf and its plumbing in compileiq.utils._libciq.

A fake libciq built from ctypes callbacks exercises the real C calling convention (the
CIQ_LIBCIQ_PATH development path) without native code or encrypted fixtures. A fake
native runtime covers the default path. Search spaces are resolved through a fake
resolver, so no network or search-space files are needed.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
import types

import pytest

import compileiq.utils._libciq as libciq
from compileiq import acf
from compileiq.utils._libciq import AcfTranslationError, LibciqUnavailableError

_BYTE_POINTER = ctypes.POINTER(ctypes.c_ubyte)
_TRANSLATE_FN = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    _BYTE_POINTER,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
)
_FREE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class FakeLibciq:
    """Returns b"TRANSLATED:" + the input; returns ``error`` instead when it is set."""

    def __init__(self, error: int = 0):
        self.error = error
        self.live: dict[int, ctypes.Array] = {}
        self.calls: list[tuple[bytes, list[str]]] = []
        self.freed: list[int] = []
        self.translate_acf = _TRANSLATE_FN(self._translate)
        self.free_buffer = _FREE_FN(self._free)

    def _translate(self, acf_bytes, acf_len, paths, n_paths, out_len, code):
        self.calls.append(
            (ctypes.string_at(acf_bytes, acf_len), [paths[i].decode() for i in range(n_paths)])
        )
        out_len[0] = 0
        if self.error:
            code[0] = self.error
            return None
        translated = b"TRANSLATED:" + ctypes.string_at(acf_bytes, acf_len)
        buffer = ctypes.create_string_buffer(translated, len(translated))
        address = ctypes.addressof(buffer)
        self.live[address] = buffer
        out_len[0] = len(translated)
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
    monkeypatch.setattr(libciq, "_native_translate", lambda: None)
    monkeypatch.setattr(libciq, "load_libciq", lambda: fake)
    return fake


class FakeNativeRuntime(types.ModuleType):
    """Stands in for the compiled runtime's translate_acf(acf, paths) -> (code, bytes)."""

    def __init__(self, error: int = 0):
        super().__init__(libciq.NATIVE_RUNTIME_MODULE)
        self.error = error
        self.calls: list[tuple[bytes, list[str]]] = []

    def translate_acf(self, acf_bytes, paths):
        self.calls.append((acf_bytes, paths))
        if self.error:
            return self.error, b""
        return 0, b"NATIVE:" + acf_bytes


@pytest.fixture
def native(monkeypatch):
    runtime = FakeNativeRuntime()
    monkeypatch.delenv(libciq.LIBCIQ_PATH_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, libciq.NATIVE_RUNTIME_MODULE, runtime)
    return runtime


def _fake_resolve(available):
    def resolve(compiler, version, variant, tag):
        key = (compiler, version, variant)
        if key not in available:
            raise LookupError(f"No manifest entry for {key}")
        return pathlib.Path(available[key])

    return resolve


PUBLISHED_13_4 = {
    ("ptxas", "13.4", "default"): "/ss/ptxas13.4.bin",
    ("ptxas", "13.4", "att"): "/ss/ptxas13.4_att.bin",
    ("nvcc", "13.4", "default"): "/ss/nvcc13.4.bin",
}


@pytest.fixture
def published(monkeypatch):
    monkeypatch.setattr(acf, "resolve", _fake_resolve(PUBLISHED_13_4))


class TestTranslateControls:
    def test_passes_acf_and_paths_through_the_c_api(self, fake_libciq):
        paths = ["/ss/ptxas.bin", pathlib.Path("/ss/nvcc.bin")]
        out = libciq.translate_controls(b"\x00acf\x01", paths)

        assert out == b"TRANSLATED:\x00acf\x01"
        assert fake_libciq.calls == [(b"\x00acf\x01", ["/ss/ptxas.bin", "/ss/nvcc.bin"])]
        assert len(fake_libciq.freed) == 1
        assert fake_libciq.live == {}

    @pytest.mark.parametrize("code", sorted(libciq.TRANSLATE_ERRORS))
    def test_maps_error_codes(self, fake_libciq, code):
        fake_libciq.error = code

        with pytest.raises(AcfTranslationError) as exc_info:
            libciq.translate_controls(b"acf", ["/ss/ptxas.bin"])

        assert exc_info.value.code == code
        assert libciq.TRANSLATE_ERRORS[code] in str(exc_info.value)
        assert fake_libciq.freed == []

    def test_reports_unknown_error_codes(self, fake_libciq):
        fake_libciq.error = -7

        with pytest.raises(AcfTranslationError, match="unknown error"):
            libciq.translate_controls(b"acf", ["/ss/ptxas.bin"])

    def test_rejects_empty_input_without_calling_libciq(self, fake_libciq):
        with pytest.raises(AcfTranslationError) as exc_info:
            libciq.translate_controls(b"", ["/ss/ptxas.bin"])

        assert exc_info.value.code == -30
        assert fake_libciq.calls == []

    def test_prefers_the_native_runtime(self, native):
        out = libciq.translate_controls(b"acf", [pathlib.Path("/ss/ptxas.bin")])

        assert out == b"NATIVE:acf"
        assert native.calls == [(b"acf", ["/ss/ptxas.bin"])]

    def test_maps_native_error_codes(self, native):
        native.error = -32

        with pytest.raises(AcfTranslationError) as exc_info:
            libciq.translate_controls(b"acf", ["/ss/ptxas.bin"])
        assert exc_info.value.code == -32

    def test_developer_libciq_override_skips_the_runtime(self, native, monkeypatch):
        fake = FakeLibciq()
        monkeypatch.setenv(libciq.LIBCIQ_PATH_ENV_VAR, "fake-libciq")
        monkeypatch.setattr(libciq, "load_libciq", lambda: fake)

        assert libciq.translate_controls(b"acf", ["/ss/ptxas.bin"]) == b"TRANSLATED:acf"
        assert native.calls == []

    def test_unavailable_without_runtime_or_override(self, monkeypatch):
        monkeypatch.delenv(libciq.LIBCIQ_PATH_ENV_VAR, raising=False)
        monkeypatch.setattr(libciq, "_native_translate", lambda: None)

        with pytest.raises(LibciqUnavailableError, match="cannot translate ACFs"):
            libciq.translate_controls(b"acf", ["/ss/ptxas.bin"])

    def test_developer_libciq_without_translate_acf(self, monkeypatch):
        class OldLibciq:
            pass

        monkeypatch.setenv(libciq.LIBCIQ_PATH_ENV_VAR, "old-libciq")
        monkeypatch.setattr(libciq, "load_libciq", lambda: OldLibciq())

        with pytest.raises(LibciqUnavailableError, match="does not export translate_acf"):
            libciq.translate_controls(b"acf", ["/ss/ptxas.bin"])


class TestTranslateCompilerConfig:
    def test_passes_every_published_compiler(self, fake_libciq, published):
        out = acf.translate_compiler_config(b"acf".hex(), "13.4")

        assert out == b"TRANSLATED:acf".hex()
        assert fake_libciq.calls == [(b"acf", ["/ss/ptxas13.4.bin", "/ss/nvcc13.4.bin"])]

    def test_uses_requested_ptxas_variant(self, fake_libciq, published):
        acf.translate_compiler_config(b"acf".hex(), "13.4", ptxas_variant="att")

        assert fake_libciq.calls[0][1] == ["/ss/ptxas13.4_att.bin", "/ss/nvcc13.4.bin"]

    def test_skips_compilers_without_a_published_search_space(self, fake_libciq, monkeypatch):
        monkeypatch.setattr(
            acf, "resolve", _fake_resolve({("ptxas", "13.5", "default"): "/ss/ptxas13.5.bin"})
        )

        acf.translate_compiler_config(b"acf".hex(), "13.5")

        assert fake_libciq.calls[0][1] == ["/ss/ptxas13.5.bin"]

    def test_rejects_unpublished_ptxas_variant(self, fake_libciq, published):
        with pytest.raises(LookupError):
            acf.translate_compiler_config(b"acf".hex(), "13.4", ptxas_variant="missing")
        assert fake_libciq.calls == []

    def test_rejects_unpublished_ctk(self, fake_libciq, published):
        with pytest.raises(LookupError, match="CTK 9.9"):
            acf.translate_compiler_config(b"acf".hex(), "9.9")
        assert fake_libciq.calls == []


class TestCli:
    def test_translates_and_writes_default_output(self, fake_libciq, published, tmp_path):
        source = tmp_path / "kernel.acf"
        source.write_bytes(b"source-acf")

        assert acf.main(["translate", str(source), "--to", "13.4"]) == 0

        assert (tmp_path / "kernel.13.4.acf").read_bytes() == b"TRANSLATED:source-acf"

    def test_honours_output_and_variant(self, fake_libciq, published, tmp_path):
        source = tmp_path / "kernel.acf"
        source.write_bytes(b"source-acf")
        output = tmp_path / "translated.acf"

        args = ["translate", str(source), "--to", "13.4", "--ptxas-variant", "att"]
        assert acf.main(args + ["-o", str(output)]) == 0

        assert output.read_bytes() == b"TRANSLATED:source-acf"
        assert fake_libciq.calls[0][1][0] == "/ss/ptxas13.4_att.bin"

    def test_reports_translation_errors(self, fake_libciq, published, tmp_path, capsys):
        fake_libciq.error = -30
        source = tmp_path / "kernel.acf"
        source.write_bytes(b"not-an-acf")

        assert acf.main(["translate", str(source), "--to", "13.4"]) == 1

        assert "not an encrypted ACF" in capsys.readouterr().err
        assert not (tmp_path / "kernel.13.4.acf").exists()

    def test_reports_unpublished_ctk(self, fake_libciq, published, tmp_path, capsys):
        source = tmp_path / "kernel.acf"
        source.write_bytes(b"source-acf")

        assert acf.main(["translate", str(source), "--to", "9.9"]) == 1
        assert "CTK 9.9" in capsys.readouterr().err

    def test_reports_missing_input(self, fake_libciq, published, tmp_path, capsys):
        assert acf.main(["translate", str(tmp_path / "missing.acf"), "--to", "13.4"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_requires_target_version(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            acf.main(["translate", str(tmp_path / "kernel.acf")])
        assert exc_info.value.code == 2

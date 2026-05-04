import subprocess
import warnings
from unittest.mock import MagicMock, patch

import pytest

from compileiq.utils.gpu import (
    _call_and_warn,
    ctk_supports_compileiq,
    gpu_benchmark_mode,
    has_nvidia_smi,
)


# ── has_nvidia_smi ──────────────────────────────────────────────────────────


def test_has_nvidia_smi_found():
    with patch("compileiq.utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi"):
        assert has_nvidia_smi() is True


def test_has_nvidia_smi_not_found():
    with patch("compileiq.utils.gpu.shutil.which", return_value=None):
        assert has_nvidia_smi() is False


# ── ctk_supports_compileiq ──────────────────────────────────────────────────


def _ptxas_output(version: str) -> MagicMock:
    result = MagicMock()
    result.stdout = f"Cuda compilation tools, release {version}, V{version}.0\n"
    return result


class TestCtkSupportsCompileIQ:
    def test_minimum_supported_version(self):
        with patch("compileiq.utils.gpu.subprocess.run", return_value=_ptxas_output("13.3")):
            assert ctk_supports_compileiq() is True

    def test_legacy_version_not_supported(self):
        with patch("compileiq.utils.gpu.subprocess.run", return_value=_ptxas_output("12.6")):
            assert ctk_supports_compileiq() is False

    def test_subprocess_failure_propagates(self):
        with patch(
            "compileiq.utils.gpu.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ptxas"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                ctk_supports_compileiq()


# ── _call_and_warn ──────────────────────────────────────────────────────────


class TestCallAndWarn:
    def test_success_does_not_raise_or_warn(self):
        with patch("compileiq.utils.gpu.subprocess.run"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _call_and_warn(["nvidia-smi", "--version"], raise_on_failure=True)
            assert len(w) == 0

    def test_failure_raises_when_raise_on_failure_true(self):
        with patch(
            "compileiq.utils.gpu.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                _call_and_warn(["nvidia-smi", "--version"], raise_on_failure=True)

    def test_failure_warns_when_raise_on_failure_false(self):
        with patch(
            "compileiq.utils.gpu.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            with pytest.warns(RuntimeWarning, match="GPU benchmark mode may not be fully applied"):
                _call_and_warn(["nvidia-smi", "--version"], raise_on_failure=False)


# ── gpu_benchmark_mode ──────────────────────────────────────────────────────


class TestGpuBenchmarkModeValidation:
    def test_raises_when_all_params_none(self):
        with pytest.raises(ValueError, match="At least one of"):
            with gpu_benchmark_mode():
                pass

    def test_warns_when_all_params_none_and_not_raise_on_failure(self):
        with patch("compileiq.utils.gpu.has_nvidia_smi", return_value=True):
            with pytest.warns(UserWarning, match="no effect"):
                with gpu_benchmark_mode(raise_on_failure=False):
                    pass

    def test_warns_when_nvidia_smi_missing_and_not_raise_on_failure(self):
        with patch("compileiq.utils.gpu.has_nvidia_smi", return_value=False):
            with patch("compileiq.utils.gpu.subprocess.run"):
                with pytest.warns(RuntimeWarning, match="nvidia-smi not found"):
                    with gpu_benchmark_mode(
                        clock_mhz=1000, with_sudo=False, raise_on_failure=False
                    ):
                        pass


class TestGpuBenchmarkModeCleanup:
    def test_clock_reset_on_exit(self):
        with patch("compileiq.utils.gpu.subprocess.run") as mock_run:
            with gpu_benchmark_mode(clock_mhz=1980, with_sudo=False):
                pass
        assert mock_run.call_args_list[-1].args[0] == ["nvidia-smi", "--reset-gpu-clocks"]

    def test_mem_clock_reset_on_exit(self):
        with patch("compileiq.utils.gpu.subprocess.run") as mock_run:
            with gpu_benchmark_mode(mem_clock_mhz=9001, with_sudo=False):
                pass
        assert mock_run.call_args_list[-1].args[0] == ["nvidia-smi", "--reset-memory-clocks"]

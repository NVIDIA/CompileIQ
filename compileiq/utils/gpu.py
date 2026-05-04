import os
import warnings
import re
import shutil
import subprocess
from contextlib import contextmanager


def has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def ctk_supports_compileiq() -> bool:
    version_output = subprocess.run(
        ["ptxas", "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
    return float(cuda_version) >= 13.3


def _call_and_warn(cmd: list[str], raise_on_failure: bool):
    try:
        subprocess.run(
            cmd,
            check=True,
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        if raise_on_failure:
            raise
        else:
            warnings.warn(
                f"Command '{' '.join(cmd)}' failed with error: {e}. "
                "GPU benchmark mode may not be fully applied.",
                RuntimeWarning,
                stacklevel=2,
            )


@contextmanager
def gpu_benchmark_mode(
    clock_mhz: int | None = None,
    mem_clock_mhz: int | None = None,
    power_watts: int | None = None,
    gpu_id: int | str | list[str | int] | None = None,
    with_sudo: bool = True,
    raise_on_failure: bool = True,
):
    """
    Context manager to set GPU parameters for benchmarking.
    For time measurements it is recommended to set at least `clock_mhz` to a fixed value to prevent
    large variations during the search.

    At the exit of the context manager, GPU settings will be reset to their default values.
    Args:
        clock_mhz:
            Desired GPU clock speed in MHz. If None, no call is performed.

        mem_clock_mhz:
            Desired GPU memory clock speed in MHz. If None, no call is performed.

        power_watts:
            Desired GPU power limit in watts. If None, no call is performed.

        gpu_id:
            GPU ID(s) to apply settings to. If None, settings are applied to all visible GPUs.

        with_sudo:
            Whether to prefix nvidia-smi commands with sudo. Most commands require elevated
            permissions, unless nvidia-smi is configured to no need it.
            Docker container environments may benefit from setting this to False.

        raise_on_failure:
            Whether to raise an exception if nvidia-smi commands fail.
            If False, any failed commands will issue a warning instead, and the context manager
            will continue.

    """

    # TODO: Add to nvcc and docs example
    if clock_mhz is None and mem_clock_mhz is None and power_watts is None:
        if raise_on_failure:
            raise ValueError(
                "At least one of clock_mhz, mem_clock_mhz, or power_watts must be set."
            )
        else:
            warnings.warn(
                "No GPU parameters specified for benchmark mode. "
                "`gpu_benchmark_mode` will have no effect.",
                UserWarning,
                stacklevel=2,
            )

    if not raise_on_failure and not has_nvidia_smi():
        if raise_on_failure:
            raise EnvironmentError("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
        else:
            warnings.warn(
                "nvidia-smi not found. GPU benchmark mode will be skipped. "
                "Ensure NVIDIA drivers are installed for full functionality.",
                RuntimeWarning,
                stacklevel=2,
            )

    base_command = ["sudo", "nvidia-smi"] if with_sudo else ["nvidia-smi"]
    if gpu_id is not None:
        if isinstance(gpu_id, (int, str)):
            gpu_id = [gpu_id]
        base_command += ["-i", ",".join(map(str, gpu_id))]
    try:
        if clock_mhz is not None:
            _call_and_warn(
                base_command + ["-lgc", f"{clock_mhz},{clock_mhz}"],
                raise_on_failure=raise_on_failure,
            )
        if mem_clock_mhz is not None:
            _call_and_warn(
                base_command + ["-lmc", f"{mem_clock_mhz},{mem_clock_mhz}"],
                raise_on_failure,
            )
        if power_watts is not None:
            _call_and_warn(
                base_command + ["-pl", str(power_watts)],
                raise_on_failure,
            )
        yield
    finally:
        # Don't raise on failure during cleanup
        if clock_mhz is not None:
            _call_and_warn(base_command + ["--reset-gpu-clocks"], False)
        if mem_clock_mhz is not None:
            _call_and_warn(base_command + ["--reset-memory-clocks"], False)


if __name__ == "__main__":
    # Example usage
    with gpu_benchmark_mode(
        clock_mhz=1980, power_watts=750, gpu_id=1, raise_on_failure=False, with_sudo=False
    ):
        print("Running benchmarks with fixed GPU settings...")

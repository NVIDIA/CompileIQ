"""
Public compiler search space providers.

Each provider fetches a pre-built, search space from GitHub.
"""

from __future__ import annotations

import pathlib
import requests
from typing import Protocol, runtime_checkable
from compileiq.config.const import _CACHE_DIR


@runtime_checkable
class SearchSpaceProvider(Protocol):
    """
    Structural interface for search space providers.

    Each provider is a class whose constructor takes whatever it needs
    (auth_token, version, local options, etc.) and whose ``retrieve()``
    method returns the search space.
    """

    def retrieve(self) -> dict | pathlib.Path:
        """Return a search space dict or path to a search space."""
        ...


class PtxasSearchSpace:
    """Fetches a pre-built PTXAS search space from GitHub."""

    def __init__(self, version: str = "latest"):
        self.version = version

    def retrieve(self) -> pathlib.Path:
        filename = "ptxas_search_space.bin"
        return _download_search_space(filename, self.version)


class NvccSearchSpace:
    """Fetches a pre-built NVCC search space from GitHub."""

    def __init__(self, version: str = "latest"):
        self.version = version

    def retrieve(self) -> pathlib.Path:
        filename = "nvcc_search_space.bin"
        return _download_search_space(filename, self.version)


def _download_search_space(filename: str, version: str) -> pathlib.Path:
    """Download a search space binary from GitHub and cache it locally."""
    local_path = pathlib.Path(_CACHE_DIR) / f"{version}_{filename}"

    if local_path.exists():
        return local_path

    url = (
        "https://github.com/NVIDIA/compileiq-search-spaces/releases/download/"
        f"{version}/{filename}"
    )
    response = requests.get(url)
    response.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path

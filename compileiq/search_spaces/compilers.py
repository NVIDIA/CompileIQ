"""
Public compiler search space providers.

Each provider fetches a pre-built search space, by default from a GitHub
release on ``NVIDIA/CompileIQ`` (override with ``CIQ_SEARCH_SPACES_REPO``).
For offline use, set ``CIQ_SEARCH_SPACES_DIR`` to a directory containing a
``manifest.json`` plus its referenced binaries; the providers below will
read from that directory and skip the network. For a single binary you
already have on disk, use ``LocalSearchSpaceBin`` instead.
"""

from __future__ import annotations

import pathlib
from typing import Protocol, runtime_checkable

from compileiq.search_spaces.resolver import SearchSpaceResolutionMetadata, resolve_with_metadata


@runtime_checkable
class SearchSpaceProvider(Protocol):
    """Structural interface for search space providers.

    Each provider is a class whose constructor takes whatever it needs
    (auth_token, version, local options, etc.) and whose ``retrieve()``
    method returns the search space.
    """

    def retrieve(self) -> dict | pathlib.Path: ...


class CompilerSearchSpaceBase:
    """Base class for compiler-specific GitHub-release-backed providers.

    Subclasses set the ``compiler`` class attribute. Network access can be
    bypassed by setting ``CIQ_SEARCH_SPACES_DIR`` to a directory holding a
    ``manifest.json`` plus its referenced binaries. ``version`` is the compiler
    version selector, defaulting to the launch catalog compiler version
    ``"13.3"``; use ``tag`` to pin a specific search-space release.
    """

    compiler: str  # set by subclass

    def __init__(
        self,
        version: str = "13.3",
        variant: str = "default",
        tag: str = "latest",
    ):
        self.version = version
        self.variant = variant
        self.tag = tag
        self.resolution_metadata: SearchSpaceResolutionMetadata | None = None

    def retrieve(self) -> pathlib.Path:
        resolved = resolve_with_metadata(
            compiler=self.compiler,
            compiler_version=self.version,
            variant=self.variant,
            tag=self.tag,
        )
        self.resolution_metadata = resolved.metadata
        return resolved.path


class PtxasSearchSpace(CompilerSearchSpaceBase):
    """Fetches a pre-built PTXAS search space from a GitHub release.

    Honors ``CIQ_SEARCH_SPACES_DIR`` for offline use. See module docstring.
    """

    compiler = "ptxas"


class NvccSearchSpace(CompilerSearchSpaceBase):
    """Fetches a pre-built NVCC search space from a GitHub release.

    Honors ``CIQ_SEARCH_SPACES_DIR`` for offline use. See module docstring.
    """

    compiler = "nvcc"


class LocalSearchSpaceBin:
    """Provider that returns an explicit on-disk binary path with no manifest.

    Use this when you have a single search-space ``.bin`` file in hand (a
    development build, an asset shared out-of-band, etc.) and want to skip
    both the manifest lookup and any network access. For mirrors of a full
    release (multiple variants, manifest-based variant selection), set
    ``CIQ_SEARCH_SPACES_DIR`` instead and use the standard provider classes
    above.
    """

    def __init__(self, path: str | pathlib.Path):
        resolved = pathlib.Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        self._path = resolved

    def retrieve(self) -> pathlib.Path:
        return self._path

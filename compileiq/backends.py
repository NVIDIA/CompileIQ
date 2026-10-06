"""Public backend registration and checkpoint contracts for CompileIQ.

The frontend package owns this compatibility seam. Backend implementations may be
built as separately versioned artifacts before being composed into a complete
CompileIQ wheel. They register at import time or through the
``compileiq.backends`` entry-point group.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from typing import Any

from pydantic import BaseModel, Field

from compileiq.types import ParamArg
from compileiq.utils.validation import Score


class EngineConfig(BaseModel, extra="forbid", frozen=True):
    """A registered engine name plus engine-owned options.

    Most users select an engine with a string such as ``"amber"``, the default.
    ``EngineConfig`` is the equivalent form for callers that also need to supply
    settings owned and validated by that engine.
    """

    name: str = Field(
        min_length=1,
        max_length=128,
        description="Registered engine name.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific options validated by the engine registration.",
    )


class BackendError(RuntimeError):
    """Base class for actionable backend errors."""


class BackendNotFoundError(BackendError):
    """Raised when a requested backend has not been registered."""


def resolve_engine(engine: str | EngineConfig) -> EngineConfig:
    """Normalize one public engine selection into its configured form."""

    if isinstance(engine, EngineConfig):
        name = engine.name.strip().lower()
        options = dict(engine.options)
    else:
        name = engine.strip().lower()
        options = {}
    if not name:
        raise ValueError("Engine name must be a non-empty string.")
    return EngineConfig(name=name, options=options)


class BackendCapabilityError(BackendError):
    """Raised when a backend cannot safely perform a requested operation."""


class SearchSpaceExhaustedError(BackendError, ValueError):
    """Raised when an engine cannot issue another candidate it has not issued before.

    Engines that never repeat a candidate within a search raise this once a small
    discrete search space, or a converged search, leaves too few unseen candidates for
    the next batch. ``Search.start()`` ends the search with the results so far instead
    of failing, unless no candidate has been evaluated yet.
    """


class CheckpointValidationError(BackendError):
    """Raised when a checkpoint cannot be validated or restored safely."""


class PendingCandidatePolicy(StrEnum):
    """How a restored search handles candidates that were still pending when saved.

    ``RESUBMIT`` keeps each saved pending candidate's identity and parameters and
    re-evaluates it through the newly configured worker (at-least-once
    evaluation). ``DISCARD`` drops the saved pending work from both public and
    optimizer state, leaving one ordinary unfilled target position per discarded
    candidate for a later ask to fill.
    """

    RESUBMIT = "resubmit"
    DISCARD = "discard"


class BackendCandidate(BaseModel, frozen=True):
    """One issued candidate: a parameter set plus the stable ID used to score it.

    ``param_id`` is the authoritative association between an issued candidate and
    any later score submitted for it, so scores can be submitted in any order and in
    non-empty subsets. ``params`` is the exact parameter set the backend issued.
    """

    param_id: int | str = Field(
        description="Stable identifier that associates later scores with this candidate."
    )
    params: ParamArg = Field(description="Parameter set issued for this candidate.")


class BackendCapabilities(BaseModel, frozen=True):
    """Optional backend operations that differ among supported providers."""

    arbitrary_batch_size: bool = Field(
        default=False,
        description="Accepts an ask size other than the configured pool size.",
    )
    checkpoint: bool = Field(
        default=False,
        description="Can save and restore optimizer execution state.",
    )
    sealed_checkpoint: bool = Field(
        default=False,
        description=(
            "Returns an opaque checkpoint blob already sealed by the provider. A provider "
            "may derive sealing from build-embedded key material; such sealing provides "
            "corruption detection and obfuscation, not confidentiality or unforgeability "
            "against a holder of the distributed provider library."
        ),
    )
    multiple_outstanding_asks: bool = Field(
        default=False,
        description="Can issue another candidate batch while earlier IDs remain pending.",
    )


class BackendArtifactInfo(BaseModel, frozen=True):
    """Version and provenance data recorded in checkpoint envelopes.

    These fields pin a checkpoint to the exact backend build that sealed it, so a
    restore can fail closed when the current build does not match.
    """

    backend_version: str = Field(description="Provider build/version identifier.")
    abi_version: int = Field(description="Sealed-state ABI version the provider understands.")
    artifact_sha256: str = Field(description="Hex SHA-256 of the provider's runtime artifact.")


class BackendRestoreResult(BaseModel, frozen=True):
    """Provider options reconstructed by an opaque backend checkpoint restore."""

    effective_options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider options actually in force after the restore, as read back from the "
            "sealed state. These become the search's active options, so a caller cannot "
            "silently diverge from the options the checkpoint was created with."
        ),
    )


class SearchBackend(ABC):
    """Interface implemented by separately owned search backends."""

    @property
    @abstractmethod
    def current_generation(self) -> int:
        """Return the current zero-based generation count."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pending_candidates(self) -> tuple[BackendCandidate, ...]:
        """Return a detached snapshot of candidates still awaiting scores.

        Providers own the canonical pending lifecycle. Every returned candidate,
        including nested parameter dictionaries and lists, must be independent of
        provider-private state so callers cannot mutate the backend through this
        public snapshot.
        """
        raise NotImplementedError

    @abstractmethod
    def ask(self, num_samples: int | None = None) -> list[BackendCandidate]:
        """Return detached candidates that remain pending until submitted with ``tell()``."""
        raise NotImplementedError

    @abstractmethod
    def tell(self, scores: list[Score]) -> None:
        """Accept scores associated to pending candidates by ``Score.param_id``.

        Score order is not significant. The public facade accepts non-empty
        subsets, rejects duplicate and unknown IDs, and supplies canonical
        parameters from the provider's pending snapshot. An adapter may forward the
        subset immediately or buffer it for a more restrictive transport.
        """
        raise NotImplementedError

    def checkpoint(self) -> bytes:
        """Return provider-owned execution state as opaque bytes.

        The facade stores and checksums the bytes but never parses or interprets
        them. A provider with textual state can return its UTF-8 encoding. A
        provider advertising ``sealed_checkpoint`` must seal and authenticate
        these bytes before returning them.
        """
        raise BackendCapabilityError(
            f"Backend {type(self).__name__!r} does not support checkpointing."
        )

    def restore(
        self,
        sealed_blob: bytes,
        *,
        pending_policy: PendingCandidatePolicy = PendingCandidatePolicy.RESUBMIT,
    ) -> BackendRestoreResult:
        """Rebuild provider state from the provider's opaque checkpoint bytes.

        ``sealed_blob`` is the exact opaque output of :meth:`checkpoint`.
        ``pending_policy`` decides how candidates that were outstanding at save
        time are handled (see :class:`PendingCandidatePolicy`).
        """
        raise BackendCapabilityError(
            f"Backend {type(self).__name__!r} does not support checkpoint restore."
        )

    @property
    def is_complete(self) -> bool:
        """Return whether the backend has emitted its terminal state."""
        return False

    @property
    def has_accepted_results_in_current_generation(self) -> bool:
        """Return whether any successful evaluations exist in the current generation.

        This gates the first-generation "all evaluations failed" fail-fast: if a
        successful subset was already accepted this generation, a later all-failed
        subset is not treated as a total first-generation failure. Backends that
        accept a generation across multiple tell calls should override this from
        their execution-owned state. The default is conservatively ``False`` so
        providers that do not track partial success keep the strict fail-fast.
        """
        return False

    @property
    def max_refill_batches_per_generation(self) -> int | None:
        """Bound failed-evaluation refill batches in managed ``start()``.

        The default 50-batch ceiling is a fail-safe against a provider that
        consumes work forever without advancing. It is not an optimizer or
        manual ask/tell limit; providers may override it or return ``None``.
        """
        return 50

    def close(self) -> None:
        """Release backend-owned processes, sockets, and other resources."""


BackendFactory = Callable[..., SearchBackend]


@dataclass(frozen=True)
class BackendRegistration:
    """One backend factory plus its capabilities and artifact provenance."""

    name: str
    factory: BackendFactory
    capabilities: BackendCapabilities
    artifact: BackendArtifactInfo
    options_model: type[BaseModel] | None = None
    search_config_validator: Callable[[Any], None] | None = None


_BACKENDS: dict[str, BackendRegistration] = {}
_ENTRY_POINTS_LOADED = False
_PROVIDER_INSTALL_HINTS = {
    "amber": (
        "The default 'amber' engine is unavailable because this installation does not contain "
        "its native payload. Install a complete platform-specific 'compileiq' wheel for this "
        "version and retry. In a source checkout, run 'make install', which materializes "
        "the native runtime checked in under native-runtime-wheels/."
    ),
}


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    capabilities: BackendCapabilities,
    artifact: BackendArtifactInfo,
    options_model: type[BaseModel] | None = None,
    search_config_validator: Callable[[Any], None] | None = None,
    replace: bool = False,
) -> None:
    """Register a runtime backend without the frontend importing provider code.

    Providers call this from an entry-point registrar so the frontend package
    never has to import a provider module directly.
    """

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Backend name must be non-empty.")
    if normalized_name in _BACKENDS and not replace:
        raise ValueError(f"Backend {normalized_name!r} is already registered.")
    _BACKENDS[normalized_name] = BackendRegistration(
        name=normalized_name,
        factory=factory,
        capabilities=capabilities,
        artifact=artifact,
        options_model=options_model,
        search_config_validator=search_config_validator,
    )


def unregister_backend(name: str) -> None:
    """Remove a registration. Primarily useful for isolated tests and plugin reloads."""

    _BACKENDS.pop(name.strip().lower(), None)


def _load_entry_point_backends() -> None:
    """Load installed provider registrars once, only when a provider is first requested.

    Lazy discovery keeps ``import compileiq`` from importing the composed native payload
    and its dependencies until a search selects an engine.
    """

    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return

    entry_points = metadata.entry_points()
    selected = (
        entry_points.select(group="compileiq.backends")
        if hasattr(entry_points, "select")
        else entry_points.get("compileiq.backends", [])
    )
    registrations_before_load = _BACKENDS.copy()
    try:
        for entry_point in selected:
            registrar = entry_point.load()
            if not callable(registrar):
                raise BackendError(
                    "Backend entry point "
                    f"{entry_point.name!r} must resolve to a callable registrar."
                )
            try:
                registrar()
            except Exception as exc:
                distribution = getattr(entry_point, "dist", None)
                distribution_name = getattr(distribution, "name", None)
                distribution_version = getattr(distribution, "version", None)
                owner = (
                    f" from {distribution_name} {distribution_version}"
                    if distribution_name and distribution_version
                    else ""
                )
                raise BackendError(
                    f"Backend entry point {entry_point.name!r}{owner} failed to register: {exc}"
                ) from exc
    except Exception:
        _BACKENDS.clear()
        _BACKENDS.update(registrations_before_load)
        raise
    _ENTRY_POINTS_LOADED = True


def get_backend_registration(name: str) -> BackendRegistration:
    """Resolve one registered backend, loading artifact entry points once on demand."""

    normalized_name = name.strip().lower()
    if normalized_name not in _BACKENDS:
        _load_entry_point_backends()
    try:
        return _BACKENDS[normalized_name]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS)) or "none"
        install_hint = _PROVIDER_INSTALL_HINTS.get(normalized_name)
        detail = f" {install_hint}" if install_hint is not None else ""
        raise BackendNotFoundError(
            f"Engine {normalized_name!r} is not registered. Available engines: {available}."
            f"{detail}"
        ) from None


def create_backend(name: str, **kwargs: Any) -> SearchBackend:
    """Create an instance of one registered engine backend."""

    registration = get_backend_registration(name)
    backend = registration.factory(**kwargs)
    if not isinstance(backend, SearchBackend):
        raise TypeError(
            f"Backend factory {registration.name!r} returned {type(backend).__name__}, "
            "expected SearchBackend."
        )
    return backend


def is_protected_search_space(search_space: Any) -> bool:
    """Return whether any resolved search-space entry is file-backed."""

    if isinstance(search_space, (str, pathlib.Path)):
        return True
    if isinstance(search_space, list):
        return any(is_protected_search_space(item) for item in search_space)
    return False


def _canonical_search_space(value: Any) -> Any:
    """Build deterministic identity material without retaining protected paths or bytes."""

    if isinstance(value, pathlib.Path):
        return {
            "kind": "file",
            "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
        }
    if isinstance(value, str):
        path = pathlib.Path(value)
        if path.is_file():
            return {
                "kind": "file",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        return {"kind": "string", "value": value}
    if isinstance(value, BaseModel):
        return _canonical_search_space(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_search_space(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_search_space(item) for item in value]
    return value


def search_space_identity(search_space: Any) -> str:
    """Return a stable identity without storing protected schema or file paths."""

    canonical = json.dumps(
        _canonical_search_space(search_space),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

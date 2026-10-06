import pathlib
import os
import shutil
import json
import warnings
import base64
import hashlib
import hmac
import math
from datetime import datetime
from tqdm.auto import tqdm
from uuid import uuid4
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    ConfigDict,
    PrivateAttr,
    StrictFloat,
    StrictInt,
    field_validator,
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeAlias,
    Literal,
    cast,
)
from compileiq.utils.validation import BASELINE_CONFIG, Score, validate_scores
from compileiq.tracker import (
    _TRACKER_TYPES_TO_CLASSES,
    _TRACKER_TYPES_TO_CONFIG,
)
from compileiq.config.const import (
    _CACHE_DIR,
    KEEP_CACHE_FILES,
    unsafe_allow_checkpoint_provenance_mismatch,
)
from compileiq.types import (
    BaseTracker,
    ParamArg,
    ProblemType,
    TrackerTypes,
    Worker,
    WorkerTypes,
    SearchConfiguration,
    InternalSearchConfiguration,
    TrackerConfig,
    DefaultTrackerConfig,
)
from compileiq.search_spaces.compilers import SearchSpaceProvider
from compileiq.results import SearchResult
from compileiq.backends import (
    BackendCandidate,
    BackendCapabilityError,
    BackendRegistration,
    BackendRestoreResult,
    CheckpointValidationError,
    EngineConfig,
    PendingCandidatePolicy,
    SearchBackend,
    create_backend,
    get_backend_registration,
    is_protected_search_space,
    resolve_engine,
    search_space_identity,
)

SearchSpaceInput: TypeAlias = (
    Dict[str, Any]
    | pathlib.Path
    | List[Dict | pathlib.Path | SearchSpaceProvider]
    | SearchSpaceProvider
)
SearchConfigInput: TypeAlias = Dict[str, Any] | SearchConfiguration | pathlib.Path
ResolvedSearchSpaceInput: TypeAlias = Dict[str, Any] | pathlib.Path | List[Dict | pathlib.Path]
SearchSpaceResolutionMetadataDict: TypeAlias = dict[str, str | int | None]

# Largest magnitude that fits in a signed 64-bit integer. Visible checkpoint result
# integers must round-trip across languages, so they are bounded to this range.
_SIGNED_64_BIT_BOUND = 2**63

# Maximum nesting depth allowed in a visible checkpoint result parameter payload.
_MAX_CHECKPOINT_NESTING_DEPTH = 64

# Checkpoint v1 resource limits shared by envelope validation and managed preflight.
_MAX_CHECKPOINT_RESULT_ROWS = 262_144
_MAX_CHECKPOINT_POPULATION = 4_096
_MAX_CHECKPOINT_CANDIDATE_HISTORY = 262_144

# The only checks the unsafe debug override may downgrade to warnings. Both are
# build provenance that a rebuild legitimately changes. Format, ABI, engine
# identity, search-space identity, search configuration, and integrity stay hard.
_OVERRIDABLE_PROVENANCE_KEYS = frozenset({"artifact_sha256", "provider_version"})


class CheckpointCompatibilityWarning(UserWarning):
    """A build-provenance check was bypassed via the unsafe debug override."""


class CheckpointSearchConfig(BaseModel):
    """Bounded public search shape needed before the sealed state is restored."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    problem_type: Literal["min", "max"]
    normalize: bool
    num_objectives: int = Field(ge=1, le=64)
    generations: int = Field(ge=1, le=2**32 - 1)
    pool_size: int = Field(ge=6, le=_MAX_CHECKPOINT_POPULATION)


CheckpointScoreValue: TypeAlias = StrictInt | StrictFloat | Literal["*"]


def _validate_checkpoint_json_value(value: Any) -> Any:
    """Reject non-JSON or pathologically nested values in visible result parameters."""

    nodes = 0

    def validate(item: Any, depth: int) -> None:
        """Walk one visible parameter payload while enforcing resource limits."""

        nonlocal nodes
        nodes += 1
        if nodes > 262144:
            raise ValueError("checkpoint result parameters contain too many values")
        if depth > _MAX_CHECKPOINT_NESTING_DEPTH:
            raise ValueError("checkpoint result parameters exceed the nesting limit")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int):
            if not -_SIGNED_64_BIT_BOUND <= item < _SIGNED_64_BIT_BOUND:
                raise ValueError("checkpoint result integers must fit in signed 64 bits")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("checkpoint result numbers must be finite")
            return
        if isinstance(item, str):
            if len(item.encode("utf-8")) > 16 * 1024 * 1024:
                raise ValueError("checkpoint result strings cannot exceed 16 MiB")
            return
        if isinstance(item, list):
            for child in item:
                validate(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("checkpoint result parameter keys must be strings")
                if len(key.encode("utf-8")) > 4096:
                    raise ValueError("checkpoint result parameter keys cannot exceed 4 KiB")
                validate(child, depth + 1)
            return
        raise ValueError(
            f"checkpoint result parameters contain unsupported {type(item).__name__} values"
        )

    validate(value, 0)
    return value


class CheckpointResultRow(BaseModel):
    """One bounded user-visible SearchResult row stored outside optimizer state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    metadata: str = Field(max_length=1024 * 1024)
    generation: int = Field(ge=0, le=2**32 - 1)
    scores: list[CheckpointScoreValue] = Field(min_length=1, max_length=64)
    params: ParamArg
    normalized_scores: list[CheckpointScoreValue] | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )

    @field_validator("params")
    @classmethod
    def validate_params(cls, value):
        """Accept only bounded JSON values in the owner-visible result ledger."""

        return _validate_checkpoint_json_value(value)

    @field_validator("scores", "normalized_scores")
    @classmethod
    def validate_score_values(cls, values):
        """Reject non-finite scores that strict numeric parsing alone permits."""

        if values is not None and any(
            isinstance(value, float) and not math.isfinite(value) for value in values
        ):
            raise ValueError("checkpoint result scores must be finite")
        return values


class CheckpointResultLedger(BaseModel):
    """Validated user-visible history carried in the single checkpoint artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rows: list[CheckpointResultRow] = Field(
        default_factory=list,
        max_length=_MAX_CHECKPOINT_RESULT_ROWS,
    )


class CheckpointEnvelope(BaseModel):
    """Safe outer metadata required to reconstruct a Search before unsealing."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format: Literal["compileiq-checkpoint-v1"]
    engine: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=128)
    provider_version: str = Field(min_length=1, max_length=128)
    abi_version: int = Field(ge=0, le=2**32 - 1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_space_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_config: CheckpointSearchConfig
    exit_on_failure: bool
    worker_type_name: str | None = Field(default=None, min_length=1, max_length=128)
    tracker_type_name: str | None = Field(default=None, min_length=1, max_length=128)
    result_ledger: CheckpointResultLedger
    sealed_blob_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sealed_blob: str = Field(min_length=1, max_length=24 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_result_shape(self):
        """Keep every ledger row consistent with the saved objective configuration."""

        for row in self.result_ledger.rows:
            if len(row.scores) != self.search_config.num_objectives:
                raise ValueError("checkpoint result score count does not match search config")
            if self.search_config.normalize:
                if (
                    row.normalized_scores is None
                    or len(row.normalized_scores) != self.search_config.num_objectives
                ):
                    raise ValueError(
                        "checkpoint normalized score count does not match search config"
                    )
            elif row.normalized_scores is not None:
                raise ValueError("checkpoint has normalized results for an unnormalized search")
        return self


def _load_checkpoint_envelope(path: pathlib.Path) -> CheckpointEnvelope:
    """Read a validated envelope while rejecting duplicate JSON object keys."""

    def unique_object(pairs):
        """Reject duplicate keys instead of letting the JSON parser silently choose one."""

        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate checkpoint envelope key: {key!r}.")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        return CheckpointEnvelope.model_validate(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        raise CheckpointValidationError("Checkpoint envelope is malformed.") from None


class Search(BaseModel):
    """
    Your main class to start a CompileIQ search.
    Instantiate this class with your objective function, search space, and search configuration,
    then call `start()` to run the search and retrieve the results.
    """

    ## User defined (Public) Attributes
    objective_function: Callable = Field(
        description=(
            "A Python function that runs the task and returns score(s). "
            "The function must have all imports and objects declared inside."
        )
    )
    search_space: SearchSpaceInput = Field(
        description=(
            "The user search space for CompileIQ to explore. "
            "The objective function will receive a single set following this declaration. "
            "Accepted values: a dict mapping string keys to compileiq search_spaces functions, "
            "a path to an existing search-space file, or a SearchSpaceProvider instance."
        )
    )
    search_config: SearchConfigInput = Field(
        description=(
            "Search configuration parameters such as generation number and mutation rate. "
            "Accepted values: a SearchConfiguration object, a dict with SearchConfiguration keys, "
            "or a path to an existing .config file."
        )
    )
    worker_type: WorkerTypes | type[Worker] = Field(
        default=WorkerTypes.DEFAULT,
        description=(
            "Selects which worker implementation runs your objective function. "
            "Built-in options via WorkerTypes: NATIVE (default, local multiprocessing), "
            "ISOLATED (fresh local process per task), RAY (distributed via Ray), "
            "ASYNC (asyncio concurrency), or MANUAL (external ask/tell execution only). "
            "A Worker subclass can also be passed directly for custom implementations."
        ),
    )
    tracker_config: TrackerConfig = Field(
        default_factory=DefaultTrackerConfig,
        description=(
            "A TrackerConfig that defines how experiment tracking will be handled. "
            "Refer to TrackerTypes for available options."
        ),
    )
    debug: bool = Field(
        default=False,
        description=(
            "Passed to the selected engine, which may emit additional diagnostics "
            "when enabled."
        ),
    )
    cache_folder: Optional[pathlib.Path] = Field(
        default=None,
        description=(
            "Base directory for cache files created during the run. "
            "Cleaned up at the end unless `CIQ_KEEP_CACHE=1` is set. "
            "Defaults to `~/.cache/compileiq` if not provided."
        ),
    )
    dump_results: Optional[pathlib.Path] = Field(
        default=None,
        description=(
            "If set, the results CSV is written to this path after every evaluation batch "
            "(typically one batch per pool_size evaluations). No file is written if None."
        ),
    )
    disable_progress_bar: bool = Field(
        default=False,
        description="Disables the TQDM progress bar.",
    )
    exit_on_failure: bool = Field(
        default=True,
        description=(
            "When True, execution terminates with a RuntimeError if all objectives fail "
            "in the first generation. Set to False if your search has an inherently high "
            "failure rate."
        ),
    )
    engine: str | EngineConfig = Field(
        default="amber",
        description=(
            "Execution engine. The default 'amber' selects the native NSGA-III runtime "
            "included in complete platform wheels. Use EngineConfig(name=..., options={...}) "
            "when passing engine-owned settings."
        ),
    )

    ## Private Attributes
    _create_new_id: Callable = lambda _: (
        datetime.now().strftime("%Y-%m-%d-%H_%M_%S-") + str(uuid4())
    )
    _worker: Worker = PrivateAttr()
    _tracker: BaseTracker = PrivateAttr()
    _worker_type_name: str | None = PrivateAttr(default=None)
    _tracker_type_name: str | None = PrivateAttr(default=None)
    current_generation: int = Field(
        0, init=False, description="Current generation of the search, starting at 0."
    )
    _search_config: InternalSearchConfiguration = PrivateAttr()
    _result: SearchResult = PrivateAttr()
    _using_file_backed_search_space: bool | list[bool] = False
    _search_space_resolution_metadata: list[SearchSpaceResolutionMetadataDict] | None = PrivateAttr(
        default=None
    )

    # Cache directory management
    _base_cache_dir: Optional[pathlib.Path] = PrivateAttr(default=None)

    _backend_impl: SearchBackend | None = PrivateAttr(default=None)
    _backend_registration: BackendRegistration | None = PrivateAttr(default=None)
    _reported_backend_candidate_ids: set[str] = PrivateAttr(default_factory=set)
    _engine_identity: str = PrivateAttr(default="amber")
    _provider_identity: str = PrivateAttr(default="amber")
    _engine_options: dict[str, Any] = PrivateAttr(default_factory=dict)
    _engine_options_were_declared: bool = PrivateAttr(default=False)
    _has_activity: bool = PrivateAttr(default=False)

    # Pydantic config
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("search_config", mode="after")
    def normalize_search_config(cls, value: SearchConfigInput) -> SearchConfigInput:
        if isinstance(value, SearchConfiguration):
            return value
        if isinstance(value, dict):
            return SearchConfiguration(**value)
        return SearchConfiguration.from_legacy(str(value))

    @field_validator("engine", mode="after")
    @classmethod
    def normalize_engine(cls, value: str | EngineConfig) -> str | EngineConfig:
        """Normalize string names while preserving a validated EngineConfig."""

        if isinstance(value, EngineConfig):
            return value
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("engine must be a non-empty string")
        return normalized

    def _resolve_engine_selection(self) -> tuple[str, str, dict[str, Any]]:
        """Return engine identity, backend registration name, and engine options."""

        engine_config = resolve_engine(self.engine)
        return (
            engine_config.name,
            engine_config.name,
            dict(engine_config.options),
        )

    def _do_init_folders(self) -> None:
        # Determine the base cache directory (only on first call)
        if self._base_cache_dir is None:
            if self.cache_folder is None:
                self._base_cache_dir = pathlib.Path(_CACHE_DIR)
            else:
                self._base_cache_dir = pathlib.Path(self.cache_folder)

        # Always create a flat path: base / new_id (never nest deeper)
        #  (this is crucial for Win32 where filepath lengths are bounded)
        self.cache_folder = self._base_cache_dir / str(self._create_new_id())
        self.cache_folder.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def _init_folders(self):
        self._do_init_folders()
        return self

    def _do_setup_search_config(self) -> None:
        """
        Converting user-defined configuration to internal representation
        """

        search_config = cast(SearchConfiguration, self.search_config)
        self._search_config = InternalSearchConfiguration(**search_config.model_dump())

    @model_validator(mode="after")
    def _setup_search_config(self):
        self._do_setup_search_config()
        return self

    def _do_setup(self) -> None:
        """
        Using this as a secondary __init__ to perform validation and start variables
        that depend on user-defined values, perform additional validation and create
        required files in `cache_folder`.
        """

        # Preparing search space - resolve SearchSpaceProvider instances.
        metadata_records: list[SearchSpaceResolutionMetadataDict] = []

        def resolve_provider(
            search_space: Dict[str, Any] | pathlib.Path | SearchSpaceProvider,
        ) -> Dict[str, Any] | pathlib.Path:
            if not isinstance(search_space, SearchSpaceProvider):
                return search_space
            resolved = search_space.retrieve()
            metadata = getattr(search_space, "resolution_metadata", None)
            if metadata is not None:
                metadata_records.append(metadata.as_dict())
            return resolved

        if isinstance(self.search_space, list):
            # Multi-config searches may mix raw dict/path search spaces with
            # provider-backed entries. Resolve only the provider entries.
            self.search_space = [
                resolve_provider(search_space) for search_space in self.search_space
            ]
        else:
            # Common case: a single raw search space or one provider-backed
            # search space that resolves to a local binary path.
            self.search_space = resolve_provider(self.search_space)

        if metadata_records:
            self._search_space_resolution_metadata = metadata_records
        if isinstance(self.search_space, list):
            self._using_file_backed_search_space = [
                isinstance(sspace, pathlib.Path) for sspace in self.search_space
            ]
        else:
            self._using_file_backed_search_space = isinstance(self.search_space, pathlib.Path)

        engine_identity, provider, backend_options = self._resolve_engine_selection()
        self._engine_identity = engine_identity
        self._provider_identity = provider
        self._engine_options = backend_options
        self._engine_options_were_declared = bool(backend_options)

        registration = get_backend_registration(self.provider)
        if registration.search_config_validator is not None:
            registration.search_config_validator(cast(SearchConfiguration, self.search_config))
        self._backend_registration = registration
        backend_options = dict(self._engine_options)
        if registration.options_model is not None:
            backend_options = registration.options_model(**backend_options).model_dump()
        elif backend_options:
            raise BackendCapabilityError(
                f"Engine {registration.name!r} does not accept engine options."
            )
        self._engine_options = dict(backend_options)
        self._backend_impl = create_backend(
            self.provider,
            search_space=cast(ResolvedSearchSpaceInput, self.search_space),
            search_config=self._search_config,
            debug=self.debug,
            options=backend_options,
        )

        if self.tracker_config.type is None:
            raise ValueError("Tracker is not initialized")

        if not isinstance(self.tracker_config, TrackerConfig):
            raise ValueError(
                f"Tracker configuration is not a TrackerConfig, got {type(self.tracker_config)}"
            )
        tracker_type = TrackerTypes(self.tracker_config.type)
        self._tracker_type_name = tracker_type.value
        self._tracker = _TRACKER_TYPES_TO_CLASSES[tracker_type](self.tracker_config)

        wt = self.worker_type
        if isinstance(wt, WorkerTypes):
            worker_cls = wt.worker_type()
        elif isinstance(wt, type) and issubclass(wt, Worker):
            worker_cls = wt
        else:
            raise RuntimeError(f"Expected a WorkerTypes or Worker subclass, but found {wt}")
        self._worker_type_name = self._canonical_worker_type_name(wt)

        assert self.cache_folder is not None, "_init_folders populates cache_folder before _setup"
        self._worker = worker_cls.create(
            cache_folder=self.cache_folder,
            normalize=self._search_config.normalize,
            tracker=self._tracker,
        )
        # Installing a new backend starts a new execution identity. Internal
        # compatibility flows such as reduction() and legacy resume() call this
        # setup path on an existing Search, so facade-owned reported IDs from the
        # previous backend must not collide with the fresh backend's IDs.
        self.current_generation = 0
        self._reported_backend_candidate_ids.clear()
        self._has_activity = False
        self._result = SearchResult._initialize_empty(
            num_scores=self._search_config.num_objectives,
            problem_type=self._search_config.problem_type,
            norm_scores=self._search_config.normalize,
        )

    @model_validator(mode="after")
    def _setup(self):
        self._do_setup()
        return self

    @property
    def provider(self) -> str:
        """Resolved backend provider for diagnostics and checkpoint provenance."""

        return self._provider_identity

    @property
    def engine_options(self) -> dict[str, Any]:
        """Validated provider-owned settings active for this search."""

        return dict(self._engine_options)

    @property
    def pending_candidates(self) -> tuple[BackendCandidate, ...]:
        """Outstanding candidates, including work restored for resubmission."""

        if self._backend_impl is None:
            return ()
        return self._backend_impl.pending_candidates

    def _pending_candidate_map(self) -> dict[str, BackendCandidate]:
        """Return one provider snapshot keyed by public candidate ID."""

        return {str(candidate.param_id): candidate for candidate in self.pending_candidates}

    @property
    def result(self) -> SearchResult:
        """Accumulated accepted results, including history restored from a checkpoint."""

        return self._result

    @property
    def best_result(self) -> Dict | List[Dict]:
        """Return the best completed result or multi-objective Pareto front so far."""

        return self._result.get_best_result()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del self

    def __del__(self):
        private_attributes = getattr(self, "__pydantic_private__", None) or {}
        backend = private_attributes.get("_backend_impl")
        if backend is not None:
            backend.close()
        if hasattr(self, "_worker") and self._worker is not None:
            del self._worker
        if (
            hasattr(self, "cache_folder")
            and self.cache_folder is not None
            and os.path.exists(self.cache_folder)
            and not KEEP_CACHE_FILES
        ):
            self._clean_files()

    def sample(self, num_samples: int = 1) -> List[ParamArg]:
        """
        Instead of performing a full search, this function will just sample `num_samples`
        from the search space provided by the user.

        Args:
            num_samples (`int`): Number of samples to retrieve from the search space.
        Returns:
            A list of parameter sets sampled from the search space, in the same format as the
        parameters sent to the objective function during a normal search.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        assert self._backend_impl is not None
        assert self._backend_registration is not None
        hijacked_config = self._search_config.model_copy(deep=True)
        hijacked_config.num_objectives = 1
        hijacked_config.pool_size = max(num_samples, 6)

        preview_backend: SearchBackend = create_backend(
            self.provider,
            search_space=cast(ResolvedSearchSpaceInput, self.search_space),
            search_config=hijacked_config,
            options=self._engine_options,
            debug=self.debug,
        )
        try:
            candidates = preview_backend.ask(num_samples=num_samples)
        finally:
            preview_backend.close()

        return [candidate.params for candidate in candidates[:num_samples]]

    def ask(self, num_samples: int | None = None) -> list[BackendCandidate]:
        """Ask a registered backend for candidates without running the objective function.

        Batch sizing and repeated-ask behavior are provider-owned capabilities.
        ``num_samples`` requests that exact batch size from providers supporting
        arbitrary batches. When omitted, the provider selects its next natural
        batch from its current state. Providers separately declare whether a new
        batch may be issued while earlier candidates remain pending.
        """

        if self._backend_impl is None or self._backend_registration is None:
            raise BackendCapabilityError("The search backend is not initialized.")
        if self._backend_impl.is_complete:
            raise BackendCapabilityError(
                f"Engine {self._engine_identity!r} has completed the configured search."
            )
        capabilities = self._backend_registration.capabilities
        pending_before = self._pending_candidate_map()
        if pending_before and not capabilities.multiple_outstanding_asks:
            raise BackendCapabilityError(
                "The previous candidate batch is still pending. Submit its scores with "
                "tell() before requesting another batch."
            )
        if num_samples is not None:
            if num_samples < 1:
                raise ValueError("num_samples must be a positive integer.")
            if (
                not capabilities.arbitrary_batch_size
                and num_samples != self._search_config.pool_size
            ):
                raise BackendCapabilityError(
                    f"Engine {self._engine_identity!r} does not support arbitrary ask batch sizes; "
                    f"use pool_size={self._search_config.pool_size}."
                )

        candidates = self._backend_impl.ask(num_samples=num_samples)
        # A backend exception leaves Search pristine. Once a backend returns, it
        # may already have advanced private state even if facade validation below
        # rejects the returned batch, so restoration must treat that as activity.
        self._has_activity = True
        candidate_ids = [str(candidate.param_id) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError("Backend returned duplicate candidate identifiers in one ask batch.")
        collisions = sorted(set(candidate_ids) & pending_before.keys())
        if collisions:
            raise RuntimeError(f"Backend reused pending candidate identifiers: {collisions}.")
        reused = sorted(set(candidate_ids) & self._reported_backend_candidate_ids)
        if reused:
            raise RuntimeError(
                f"Backend reused previously reported candidate identifiers: {reused}."
            )
        return list(candidates)

    def _validate_evaluation_candidates(
        self,
        candidates: list[BackendCandidate],
    ) -> list[BackendCandidate]:
        """Validate a non-empty subset against the authoritative pending ledger."""

        if not candidates:
            raise ValueError("evaluate requires at least one pending candidate.")
        if any(not isinstance(candidate, BackendCandidate) for candidate in candidates):
            raise TypeError(
                "evaluate candidates must be BackendCandidate instances returned by ask()."
            )

        candidate_ids = [str(candidate.param_id) for candidate in candidates]
        unique_ids = set(candidate_ids)
        duplicates = len(candidate_ids) - len(unique_ids)
        pending_by_id = self._pending_candidate_map()
        pending_ids = set(pending_by_id)
        already_reported_ids = unique_ids & self._reported_backend_candidate_ids
        unknown_ids = unique_ids - pending_ids - already_reported_ids
        if duplicates or unknown_ids or already_reported_ids:
            raise ValueError(
                "Evaluation candidates contain invalid candidate IDs "
                f"(duplicates={duplicates}, unknown={sorted(unknown_ids)}, "
                f"already_reported={sorted(already_reported_ids)})."
            )

        mismatched_params = [
            candidate_id
            for candidate_id, candidate in zip(candidate_ids, candidates, strict=True)
            if candidate.params != pending_by_id[candidate_id].params
        ]
        if mismatched_params:
            raise ValueError(
                "Evaluation candidate parameters do not match the issued candidates for IDs: "
                f"{mismatched_params}."
            )
        return candidates

    def _evaluate_candidate_batch(
        self,
        candidates: list[BackendCandidate],
        *,
        num_workers: int | None,
        task_timeout: int | float | None,
        worker_kwargs: dict[str, Any],
    ) -> list[Score]:
        """Evaluate one validated candidate subset through the configured worker."""

        self._require_managed_execution("evaluate")
        candidates = self._validate_evaluation_candidates(candidates)
        if num_workers is not None:
            if not self._worker.respects_num_workers:
                warnings.warn(
                    f"num_workers is not supported by {type(self._worker).__name__}",
                    stacklevel=3,
                )
            elif num_workers < 1:
                raise ValueError("num_workers must be a positive integer.")
        if task_timeout is not None and not self._worker.supports_timeout:
            warnings.warn(
                f"task_timeout is not supported by {type(self._worker).__name__}",
                stacklevel=3,
            )

        assert self._backend_impl is not None
        generation = self._backend_impl.current_generation
        self._worker.current_generation = generation
        scores = self._worker.run(
            function=self.objective_function,
            tracker=self._tracker,
            params_pool=[candidate.params for candidate in candidates],
            params_ids=[candidate.param_id for candidate in candidates],
            num_workers=num_workers if isinstance(num_workers, int) else 1,
            num_function_returns=self._search_config.num_objectives,
            task_timeout=task_timeout,
            **worker_kwargs,
        )

        expected_ids = [str(candidate.param_id) for candidate in candidates]
        returned_ids = [str(score.param_id) for score in scores if not score.is_baseline]
        if len(returned_ids) != len(expected_ids) or set(returned_ids) != set(expected_ids):
            raise RuntimeError(
                "Worker must return exactly one non-baseline score for every evaluated "
                "candidate ID."
            )

        return scores

    def _require_managed_execution(self, operation: str) -> None:
        """Reject objective execution when the configured worker is manual-only."""

        if not self._worker.supports_managed_execution:
            raise RuntimeError(
                f"{operation}() is unavailable with {type(self._worker).__name__}. "
                "Use ask() and tell() to evaluate candidates externally."
            )

    def evaluate(
        self,
        candidates: list[BackendCandidate],
        num_workers: int | None = None,
        task_timeout: int | float | None = None,
        **worker_kwargs: Any,
    ) -> list[Score]:
        """Evaluate pending candidates without submitting them to the optimizer.

        ``candidates`` may be any non-empty subset returned by :meth:`ask`, in
        any order. Candidate IDs and parameters are validated before the worker
        runs. The returned scores remain caller-owned until passed to
        :meth:`tell`; this method does not apply optimizer failure policy,
        advance backend state, remove pending candidates, or append results.
        """

        return self._evaluate_candidate_batch(
            candidates,
            num_workers=num_workers,
            task_timeout=task_timeout,
            worker_kwargs=worker_kwargs,
        )

    def tell(self, scores: list[Score]) -> None:
        """Submit scores for pending candidates using their stable ``param_id`` values.

        List order is irrelevant, and every provider accepts any non-empty
        subset of outstanding IDs. ``Score.params`` is an optional compatibility
        echo: when supplied it is validated, but the provider's detached pending
        snapshot is authoritative for backend submissions and stored result rows.
        """

        if self._backend_impl is None or self._backend_registration is None:
            raise BackendCapabilityError("The search backend is not initialized.")
        pending_by_id = self._pending_candidate_map()
        pending_ids = set(pending_by_id)
        non_baseline_scores = [score for score in scores if not score.is_baseline]
        if not non_baseline_scores:
            raise ValueError("tell requires at least one non-baseline candidate score.")
        score_id_list = [str(score.param_id) for score in non_baseline_scores]
        score_ids = set(score_id_list)
        duplicates = len(score_id_list) - len(score_ids)
        already_reported_ids = score_ids & self._reported_backend_candidate_ids
        unknown_ids = score_ids - pending_ids - already_reported_ids
        if duplicates or unknown_ids or already_reported_ids:
            raise ValueError(
                "Reported scores contain invalid candidate IDs "
                f"(duplicates={duplicates}, unknown={sorted(unknown_ids)}, "
                f"already_reported={sorted(already_reported_ids)})."
            )
        wrong_objective_counts = [
            str(score.param_id)
            for score in scores
            if score.num_objectives != self._search_config.num_objectives
        ]
        if wrong_objective_counts:
            raise ValueError(
                "Reported score objective counts do not match this search for candidate IDs: "
                f"{wrong_objective_counts}."
            )
        mismatched_params = [
            str(score.param_id)
            for score in non_baseline_scores
            if score.params is not None
            and score.params != pending_by_id[str(score.param_id)].params
        ]
        if mismatched_params:
            raise ValueError(
                "Reported score parameters do not match the issued candidates for IDs: "
                f"{mismatched_params}."
            )
        invalid_normalized_scores: list[str] = []
        if self._search_config.normalize:
            for score in non_baseline_scores:
                if score.norm_score is None:
                    if not score.failed:
                        invalid_normalized_scores.append(str(score.param_id))
                    continue
                try:
                    validate_scores(score.norm_score, self._search_config.num_objectives)
                except ValueError:
                    invalid_normalized_scores.append(str(score.param_id))
        if invalid_normalized_scores:
            raise ValueError(
                "Reported normalized scores are missing or do not match this search for "
                f"candidate IDs: {invalid_normalized_scores}."
            )
        generation = self._backend_impl.current_generation
        evaluating_all_pending = score_ids == pending_ids
        if (
            self.exit_on_failure
            and generation == 0
            and evaluating_all_pending
            and self._check_fail_count(non_baseline_scores)
            and not self._backend_impl.has_accepted_results_in_current_generation
        ):
            raise RuntimeError(
                "All objective functions failed in the first gen. Are you sure this "
                "is expected behavior? You can disable this error by setting "
                "`exit_on_failure=False`."
            )
        canonical_scores = []
        for score in scores:
            if score.is_baseline:
                canonical_scores.append(
                    score.model_copy(update={"params": BASELINE_CONFIG})
                    if score.params is None
                    else score
                )
                continue
            updates: dict[str, Any] = {"params": pending_by_id[str(score.param_id)].params}
            if self._search_config.normalize and score.failed and score.norm_score is None:
                updates["norm_score"] = Worker.invalidate_score(self._search_config.num_objectives)
            canonical_scores.append(score.model_copy(update=updates))
        canonical_non_baseline_scores = [
            score for score in canonical_scores if not score.is_baseline
        ]
        result_rows = [
            self._result._row_for_score(score, generation, self._search_config.normalize)
            for score in canonical_scores
        ]
        self._has_activity = True
        self._backend_impl.tell(canonical_non_baseline_scores)
        self.current_generation = self._backend_impl.current_generation
        self._reported_backend_candidate_ids.update(score_ids)
        self._result._append_rows(result_rows)
        if self.dump_results is not None:
            self._result.save(self.dump_results)

    def _checkpoint_result_ledger(self) -> CheckpointResultLedger:
        """Validate the API-owned visible history before combining it with sealed state."""

        rows = self._result._checkpoint_rows()
        if len(rows) > _MAX_CHECKPOINT_RESULT_ROWS:
            raise CheckpointValidationError(
                "Checkpoint result history exceeds the supported limit of "
                f"{_MAX_CHECKPOINT_RESULT_ROWS} rows. This search has accumulated "
                f"{len(rows)} result rows; reduce the retained history or "
                "continue without creating a v1 checkpoint."
            )
        return CheckpointResultLedger(
            rows=[CheckpointResultRow.model_validate(row) for row in rows]
        )

    def _validate_checkpoint_result_boundary(
        self,
        ledger: CheckpointResultLedger,
    ) -> None:
        """Keep protected result positions opaque even in an owner-editable ledger."""

        file_backed = self._using_file_backed_search_space
        if file_backed is False:
            return
        for row in ledger.rows:
            if self._search_config.normalize and row.params == BASELINE_CONFIG:
                continue
            if file_backed is True:
                if not isinstance(row.params, str):
                    raise CheckpointValidationError(
                        "Protected checkpoint result parameters must remain opaque strings."
                    )
                continue
            if not isinstance(row.params, list) or len(row.params) != len(file_backed):
                raise CheckpointValidationError(
                    "Multi-configuration checkpoint results do not match the search-space shape."
                )
            if any(
                is_protected and not isinstance(params, str)
                for is_protected, params in zip(file_backed, row.params, strict=True)
            ):
                raise CheckpointValidationError(
                    "Protected checkpoint result parameters must remain opaque strings."
                )

    @staticmethod
    def _result_from_checkpoint(envelope: CheckpointEnvelope) -> SearchResult:
        """Rebuild reporting history without using it to influence optimizer state."""

        return SearchResult._from_checkpoint_rows(
            [row.model_dump() for row in envelope.result_ledger.rows],
            num_scores=envelope.search_config.num_objectives,
            problem_type=envelope.search_config.problem_type,
            normalize=envelope.search_config.normalize,
        )

    def _require_pristine_restore_target(self) -> None:
        """Forbid implicit merging of a checkpoint into unrelated Search activity."""

        assert self._backend_impl is not None
        if (
            self._has_activity
            or self.pending_candidates
            or len(self._result.get_results()) > 0
            or self._backend_impl.current_generation != 0
        ):
            raise CheckpointValidationError(
                "restore_checkpoint requires a fresh Search with no results or active state. "
                "Use Search.from_checkpoint() for canonical restoration."
            )
        if self._engine_options_were_declared:
            raise CheckpointValidationError(
                "restore_checkpoint restores provider options from the checkpoint and does not "
                "accept predeclared EngineConfig options. Use Search.from_checkpoint()."
            )

    def _require_checkpoint_support(self) -> None:
        """Capability-gate checkpoints and require sealing for protected spaces."""

        if self._backend_impl is None or self._backend_registration is None:
            raise BackendCapabilityError("The search backend is not initialized.")
        capabilities = self._backend_registration.capabilities
        if not capabilities.checkpoint:
            raise BackendCapabilityError(
                f"Engine {self._engine_identity!r} does not support checkpointing."
            )
        if is_protected_search_space(self.search_space) and not capabilities.sealed_checkpoint:
            raise BackendCapabilityError(
                "Protected search spaces require a backend that seals checkpoint state."
            )

    def _preflight_checkpoint_capacity(self) -> None:
        """Reject a managed checkpoint request whose configured projection exceeds v1 caps."""

        assert self._backend_impl is not None
        pool_size = cast(int, self._search_config.pool_size)
        if pool_size > _MAX_CHECKPOINT_POPULATION:
            raise CheckpointValidationError(
                f"pool_size={pool_size} exceeds the checkpointable population limit "
                f"({_MAX_CHECKPOINT_POPULATION}); reduce pool_size or run without "
                "checkpoint_path."
            )

        generation_limit = self._search_config.generations
        current_generation = self._backend_impl.current_generation
        remaining_generations = max(generation_limit - current_generation, 0)
        existing_rows = len(self._result.get_results())
        projected_rows = existing_rows + remaining_generations * pool_size
        if self._search_config.normalize and remaining_generations:
            projected_rows += 1
        if projected_rows > _MAX_CHECKPOINT_RESULT_ROWS:
            raise CheckpointValidationError(
                "This managed run's checkpoint result-history projection is "
                f"{projected_rows} rows ({existing_rows} existing + "
                f"{remaining_generations} remaining generations x pool_size={pool_size}"
                f"{' + 1 normalization baseline' if self._search_config.normalize else ''}), "
                f"which exceeds the v1 limit of {_MAX_CHECKPOINT_RESULT_ROWS}. Reduce "
                "generations or pool_size, or run without checkpoint_path."
            )

        refill_limit = self._backend_impl.max_refill_batches_per_generation
        if refill_limit is not None:
            existing_candidates = existing_rows + len(self.pending_candidates)
            projected_candidates = existing_candidates + remaining_generations * pool_size
            if projected_candidates > _MAX_CHECKPOINT_CANDIDATE_HISTORY:
                raise CheckpointValidationError(
                    "This managed run's checkpoint candidate-history projection is "
                    f"{projected_candidates} candidates ({existing_candidates} existing or "
                    f"pending + {remaining_generations} remaining generations x "
                    f"pool_size={pool_size}), which exceeds "
                    f"the v1 limit of {_MAX_CHECKPOINT_CANDIDATE_HISTORY}. Reduce "
                    "generations or pool_size, or run without checkpoint_path."
                )

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        require_no_pending: bool = False,
    ) -> pathlib.Path:
        """Atomically save visible results and backend-sealed execution state.

        Set ``require_no_pending`` to fail before sealing or artifact creation
        when any candidate remains outstanding.
        """

        self._require_checkpoint_support()
        assert self._backend_impl is not None
        assert self._backend_registration is not None
        pending = self.pending_candidates
        if require_no_pending and pending:
            raise CheckpointValidationError(
                "Checkpoint creation requires no pending candidates, but "
                f"{len(pending)} remain outstanding."
            )

        result_ledger = self._checkpoint_result_ledger()
        self._validate_checkpoint_result_boundary(result_ledger)
        sealed_blob = self._backend_impl.checkpoint()
        if not isinstance(sealed_blob, bytes) or not sealed_blob:
            raise RuntimeError("Backend returned an invalid sealed checkpoint blob.")

        artifact = self._backend_registration.artifact
        envelope = CheckpointEnvelope(
            format="compileiq-checkpoint-v1",
            engine=self._engine_identity,
            provider=self.provider,
            provider_version=artifact.backend_version,
            abi_version=artifact.abi_version,
            artifact_sha256=artifact.artifact_sha256,
            search_space_identity=search_space_identity(self.search_space),
            search_config=CheckpointSearchConfig(
                problem_type=ProblemType(self._search_config.problem_type).value,
                normalize=self._search_config.normalize,
                num_objectives=self._search_config.num_objectives,
                generations=self._search_config.generations,
                pool_size=cast(int, self._search_config.pool_size),
            ),
            exit_on_failure=self.exit_on_failure,
            worker_type_name=self._worker_type_name,
            tracker_type_name=self._tracker_type_name,
            result_ledger=result_ledger,
            sealed_blob_sha256=hashlib.sha256(sealed_blob).hexdigest(),
            sealed_blob=base64.b64encode(sealed_blob).decode("ascii"),
        )

        destination = pathlib.Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as checkpoint_file:
                os.chmod(temporary, 0o600)
                checkpoint_file.write(envelope.model_dump_json(exclude_none=True))
                checkpoint_file.write("\n")
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def restore_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        pending_policy: PendingCandidatePolicy | str = PendingCandidatePolicy.RESUBMIT,
        allow_unsafe_provenance_mismatch: bool = False,
    ) -> None:
        """Restore compatible state, optionally accepting unsafe build-provenance drift."""

        # Capability-gate first, before reading, parsing, or validating the supplied
        # artifact and before any pristine-target checks. An engine that cannot restore
        # must fail with the capability error without touching the caller's file or
        # creating Search activity.
        self._require_checkpoint_support()
        assert self._backend_impl is not None
        assert self._backend_registration is not None
        self._require_pristine_restore_target()
        pending_policy = PendingCandidatePolicy(pending_policy)

        try:
            envelope = _load_checkpoint_envelope(pathlib.Path(path))
            sealed_blob = base64.b64decode(envelope.sealed_blob, validate=True)
            self._validate_checkpoint_result_boundary(envelope.result_ledger)
            restored_result = self._result_from_checkpoint(envelope)
        except CheckpointValidationError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise CheckpointValidationError("Checkpoint envelope is malformed.") from None

        artifact = self._backend_registration.artifact
        expected = {
            "format": "compileiq-checkpoint-v1",
            "engine": self._engine_identity,
            "provider": self.provider,
            "provider_version": artifact.backend_version,
            "abi_version": artifact.abi_version,
            "artifact_sha256": artifact.artifact_sha256,
            "search_space_identity": search_space_identity(self.search_space),
        }
        allow_provenance_mismatch = (
            allow_unsafe_provenance_mismatch or unsafe_allow_checkpoint_provenance_mismatch()
        )
        bypassed_provenance: list[str] = []
        for key, expected_value in expected.items():
            actual_value = getattr(envelope, key)
            if hmac.compare_digest(str(actual_value), str(expected_value)):
                continue
            if allow_provenance_mismatch and key in _OVERRIDABLE_PROVENANCE_KEYS:
                bypassed_provenance.append(
                    f"{key}: checkpoint={actual_value!r} current={expected_value!r}"
                )
                continue
            raise CheckpointValidationError(
                f"Checkpoint {key.replace('_', ' ')} does not match this search."
            )
        expected_search_config = CheckpointSearchConfig(
            problem_type=ProblemType(self._search_config.problem_type).value,
            normalize=self._search_config.normalize,
            num_objectives=self._search_config.num_objectives,
            generations=self._search_config.generations,
            pool_size=cast(int, self._search_config.pool_size),
        )
        if envelope.search_config != expected_search_config:
            raise CheckpointValidationError(
                "Checkpoint search configuration does not match this search."
            )
        if not hmac.compare_digest(
            envelope.sealed_blob_sha256, hashlib.sha256(sealed_blob).hexdigest()
        ):
            raise CheckpointValidationError("Checkpoint sealed blob checksum is invalid.")
        if bypassed_provenance:
            warnings.warn(
                "Attempting checkpoint restore with allow_unsafe_provenance_mismatch / "
                "CIQ_UNSAFE_ALLOW_CHECKPOINT_PROVENANCE_MISMATCH. Bypassed provenance "
                "checks -> "
                + "; ".join(bypassed_provenance)
                + ". All other compatibility, ABI, identity, configuration, and integrity "
                "checks still ran, and the selected backend will still validate its own "
                "state. The restored run will come from a non-matching build.",
                CheckpointCompatibilityWarning,
                stacklevel=2,
            )

        try:
            restore_result = self._backend_impl.restore(
                sealed_blob,
                pending_policy=pending_policy,
            )
        except Exception:
            raise CheckpointValidationError(
                "Backend rejected the sealed checkpoint. It may be tampered or incompatible."
            ) from None
        if not isinstance(restore_result, BackendRestoreResult):
            raise CheckpointValidationError(
                "Backend returned an invalid checkpoint restore result."
            )
        self.current_generation = self._backend_impl.current_generation
        restored = self.pending_candidates
        restored_ids = [str(candidate.param_id) for candidate in restored]
        if len(restored_ids) != len(set(restored_ids)):
            raise CheckpointValidationError(
                "Backend restored duplicate pending candidate identifiers."
            )
        self._reported_backend_candidate_ids.clear()
        self._engine_options = dict(restore_result.effective_options)
        self._engine_options_were_declared = False
        self._result = restored_result
        self._has_activity = True

    @classmethod
    def _canonical_worker_type_name(
        cls,
        worker_type: WorkerTypes | type[Worker],
    ) -> str | None:
        """Return the public built-in name for an enum value or concrete worker class."""

        if isinstance(worker_type, WorkerTypes):
            return worker_type.value
        for candidate in WorkerTypes:
            if candidate.worker_type() is worker_type:
                return candidate.value
        return None

    @classmethod
    def _resolve_worker_type_name(cls, name: str) -> WorkerTypes | type[Worker] | None:
        """Resolve a saved public worker name without loading objects from the artifact."""

        try:
            return WorkerTypes(name)
        except ValueError:
            return None

    @classmethod
    def _resolve_tracker_type_name(cls, name: str) -> TrackerConfig | None:
        """Create the saved built-in tracker type with safe default configuration."""

        try:
            tracker_type = TrackerTypes(name)
        except ValueError:
            return None
        return _TRACKER_TYPES_TO_CONFIG[tracker_type]()

    @classmethod
    def _search_config_from_checkpoint(
        cls,
        envelope: CheckpointEnvelope,
    ) -> SearchConfiguration:
        """Rebuild the public search configuration recorded by the envelope."""

        return SearchConfiguration(**envelope.search_config.model_dump())

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        objective_function: Callable,
        search_space: SearchSpaceInput,
        worker_type: WorkerTypes | type[Worker] | None = None,
        tracker_config: TrackerConfig | None = None,
        pending_policy: PendingCandidatePolicy | str = PendingCandidatePolicy.RESUBMIT,
        exit_on_failure: bool | None = None,
        allow_unsafe_provenance_mismatch: bool = False,
        **search_kwargs: Any,
    ) -> "Search":
        """Restore a search, reconstructing safe built-in execution types when possible."""

        envelope = _load_checkpoint_envelope(pathlib.Path(path))
        if {
            "engine",
            "backend",
            "algorithm",
            "backend_options",
            "engine_options",
            "search_config",
        } & search_kwargs.keys():
            raise TypeError(
                "from_checkpoint derives search configuration, engine, and provider state "
                "from the checkpoint."
            )
        if worker_type is None:
            worker_type = (
                cls._resolve_worker_type_name(envelope.worker_type_name)
                if envelope.worker_type_name is not None
                else None
            )
            if worker_type is None:
                saved_worker = (
                    f" (saved worker {envelope.worker_type_name!r})"
                    if envelope.worker_type_name is not None
                    else ""
                )
                raise ValueError(
                    "This checkpoint was saved with a worker that cannot be reconstructed "
                    f"automatically{saved_worker}. Pass worker_type= explicitly to "
                    "Search.from_checkpoint()."
                )
        if tracker_config is None:
            tracker_config = (
                cls._resolve_tracker_type_name(envelope.tracker_type_name)
                if envelope.tracker_type_name is not None
                else None
            )
            if tracker_config is None:
                saved_tracker = (
                    f" (saved tracker {envelope.tracker_type_name!r})"
                    if envelope.tracker_type_name is not None
                    else ""
                )
                raise ValueError(
                    "This checkpoint was saved with a tracker that cannot be reconstructed "
                    f"automatically{saved_tracker}. Pass tracker_config= explicitly to "
                    "Search.from_checkpoint()."
                )
        search = cls(
            objective_function=objective_function,
            search_space=search_space,
            search_config=cls._search_config_from_checkpoint(envelope),
            worker_type=worker_type,
            tracker_config=tracker_config,
            exit_on_failure=(
                envelope.exit_on_failure if exit_on_failure is None else exit_on_failure
            ),
            engine=envelope.engine,
            **search_kwargs,
        )
        search.restore_checkpoint(
            path,
            pending_policy=pending_policy,
            allow_unsafe_provenance_mismatch=allow_unsafe_provenance_mismatch,
        )
        return search

    def start(
        self,
        num_workers: int | None = None,
        task_timeout: int | float | None = None,
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
        **additional_worker_kwargs,
    ) -> SearchResult:
        """
        Run the selected engine through its managed ask, evaluate, and tell lifecycle.

        Restored pending candidates are evaluated first through the newly configured worker.
        Otherwise the engine produces each candidate batch, the worker evaluates it, and the
        accepted scores are submitted until the configured generation limit is reached.
        The returned result contains both restored history and newly accepted evaluations.

        Args:
            num_workers:
                The maximum number of processes spawned to run parallel searches.
                This parameter is ignored by workers where `respects_num_workers`
                is `False`.

            task_timeout:
                The maximum time (in seconds) allowed for a single execution of the objective.
                If the timeout is reached, the worker will treat it as a failed execution and return
                a failed Score. This is useful to prevent workers from hanging indefinitely on
                certain parameter sets.

            checkpoint_path:
                When set, atomically replace this checkpoint after each successfully accepted
                managed batch. The selected engine must support checkpointing. No automatic
                checkpoint is created when this parameter is None.

            **additional_worker_kwargs:
                Additional keyword arguments forwarded to the worker's ``run()`` method.
                Each worker accepts the kwargs it cares about. For example:

                - RayWorker: accepts Ray task resource options
                - Custom workers: accept any kwargs they define

        Returns (`SearchResult`):
            An object with the search results.
        """

        self._require_managed_execution("start")
        checkpoint_destination = None
        if checkpoint_path is not None:
            checkpoint_destination = pathlib.Path(checkpoint_path)
            self._require_checkpoint_support()
            self._preflight_checkpoint_capacity()

        return self._start_backend(
            num_workers,
            task_timeout=task_timeout,
            checkpoint_path=checkpoint_destination,
            **additional_worker_kwargs,
        )

    def _start_backend(
        self,
        num_workers: int | None,
        task_timeout: int | float | None = None,
        checkpoint_path: pathlib.Path | None = None,
        **additional_worker_kwargs,
    ) -> SearchResult:
        """Run the registered backend through its managed lifecycle."""

        assert self._backend_impl is not None
        self._has_activity = True
        if self.cache_folder is None or not self.cache_folder.exists():
            self._do_init_folders()
        refill_counts: dict[int, int] = {}
        progress = tqdm(
            total=self._search_config.generations,
            initial=min(
                self._backend_impl.current_generation,
                self._search_config.generations,
            ),
            ascii="░▒█",
            colour="green",
            disable=self.disable_progress_bar,
            smoothing=0.8,
            bar_format="{desc} {n_fmt}/{total_fmt}|{bar}| [elapsed: {elapsed} · eta: {remaining}]"
            " {postfix}",
        )
        progress.set_description("🧬 Generation")
        try:
            self._tracker.search_starts(
                search_space_resolution_metadata=self._search_space_resolution_metadata
            )
            while (
                not self._backend_impl.is_complete
                and self._backend_impl.current_generation < self._search_config.generations
            ):
                generation_before_ask = self._backend_impl.current_generation
                batch_count = refill_counts.get(generation_before_ask, 0)
                refill_limit = self._backend_impl.max_refill_batches_per_generation
                # Guard before ask() so a provider cannot request unbounded refill
                # work without advancing its generation.
                if refill_limit is not None and batch_count >= refill_limit:
                    raise RuntimeError(
                        "Backend did not advance within max_refill_batches_per_generation "
                        f"({refill_limit}) for generation {generation_before_ask}."
                    )
                pending = self.pending_candidates
                candidates = list(pending) if pending else self.ask()
                if not candidates:
                    if self._backend_impl.is_complete:
                        self._advance_progress(progress, self._search_config.generations)
                        break
                    raise RuntimeError("Backend returned no candidates before search completion.")

                # Some providers only learn the generation number from the candidate batch
                # returned by ask(). Use that authoritative value for the worker, tracker,
                # failure policy, and result rows.
                generation = self._backend_impl.current_generation
                batch_count = refill_counts.get(generation, 0)
                # Some providers only reveal the authoritative generation in
                # their returned batch, so enforce the same bound again against
                # that post-ask value before running customer work.
                if refill_limit is not None and batch_count >= refill_limit:
                    raise RuntimeError(
                        "Backend did not advance within max_refill_batches_per_generation "
                        f"({refill_limit}) for generation {generation}."
                    )
                refill_counts[generation] = batch_count + 1
                self._advance_progress(progress, generation)
                self.current_generation = generation
                self._tracker.generation_starts(generation)

                scores = self._evaluate_candidate_batch(
                    candidates,
                    num_workers=num_workers,
                    task_timeout=task_timeout,
                    worker_kwargs=additional_worker_kwargs,
                )

                self.tell(scores)
                if checkpoint_path is not None:
                    # Managed batches evaluate every candidate they issue. A
                    # successful tell therefore leaves no public pending work,
                    # even when the backend records a replacement deficit for a
                    # refill batch in the same generation.
                    self.save_checkpoint(checkpoint_path, require_no_pending=True)
                self._tracker.generation_ends(generation)
                next_generation = self._backend_impl.current_generation
                self._advance_progress(progress, next_generation)

            return self._result
        finally:
            progress.close()
            self._tracker.search_ends()
            self.current_generation = 0
            self._backend_impl.close()
            if not KEEP_CACHE_FILES:
                self._clean_files()

    def _set_progress_best_score(self, progress) -> None:
        """Display the best completed single-objective result without affecting the run."""

        if self._search_config.num_objectives != 1:
            return
        try:
            best = self._result.get_best_result()
            if not isinstance(best, dict):
                return
            generation = int(best["generation"])
            value = best["norm_score_1" if self._search_config.normalize else "score_1"]
            progress.set_postfix(
                {
                    "🏆 best_score": f"{value:.4f}",
                    "at_gen": generation,
                }
            )
        except (KeyError, TypeError, ValueError):
            return

    def _advance_progress(self, progress, generation: int) -> None:
        """Advance the bar to the backend's authoritative completed generation."""

        if generation <= progress.n:
            return
        self._set_progress_best_score(progress)
        progress.update(generation - progress.n)

    @property
    def search_space_resolution_metadata(self) -> list[SearchSpaceResolutionMetadataDict] | None:
        """One metadata record per resolved provider, or None if no provider was used."""
        if self._search_space_resolution_metadata is None:
            return None
        return [dict(metadata) for metadata in self._search_space_resolution_metadata]

    def _clean_files(self):
        """Deletes cache files from this run"""
        if hasattr(self, "_tracker") and self._tracker is not None:
            self._tracker.cleanup()
        if self.cache_folder is not None and pathlib.Path(self.cache_folder).exists():
            shutil.rmtree(self.cache_folder, ignore_errors=True)

    def _check_fail_count(self, scores: List[Score]):
        """
        In case all non-baseline scores are failures, this function returns True.
        """
        return all([score.failed for score in scores if not score.is_baseline])

import os
import re
import pathlib
import ast
import warnings
from math import comb, ceil
from enum import StrEnum
from abc import ABC, abstractmethod
from importlib.metadata import version
from pydantic import BaseModel, Field, model_validator, field_validator, SkipValidation
from typing import (
    Callable,
    Optional,
    List,
    Literal,
    Sequence,
    TextIO,
    TypeAlias,
    cast,
    overload,
)
from compileiq.utils.validation import (  # noqa: F401 (re-exported)
    SingleScore,
    MultiScore,
    Score,
    INVALID_SCORE,
    BASELINE_CONFIG,
)


ParamArg: TypeAlias = dict | str | list[dict | str]


class ProblemType(StrEnum):
    """Supported problem types, min or max"""

    MIN = "min"
    MAX = "max"


class MultiScoreComparison(StrEnum):
    """
    Possible ways to aggregate multi-score at the end of the search.

    Attributes:
        AVERAGE:
            Averages all multi-scores
        STDDEV:
            Finds the standard deviation from all scores
        PARETO:
            This is the preferred method. Calculate the pareto front
            for all scores.
    """

    AVERAGE = "avg"
    STDDEV = "stddev"
    PARETO = "pareto_front"


class TrackerTypes(StrEnum):
    """
    Chooses between the available tracker types for experiment tracking.
    Trackers are responsible for logging and tracking experiment data during the search.

    Attributes:
        LOGURU:
            Uses the Loguru Python logging library for tracking experiment events.
            Provides simple and structured logging to an output file.
            This is the default tracker type.

        MLFLOW:
            Uses MLflow for ML experiment tracking and logging.
            Provides comprehensive experiment tracking including parameters, metrics, and artifacts.

        DISABLED:
            Disables experiment tracking. No data will be logged or tracked.
    """

    LOGURU = "loguru"
    MLFLOW = "mlflow"
    DISABLED = "disabled"
    DEFAULT = DISABLED


# We use extra="allow" to allow any extra fields to be passed to the tracker config.
# Useful for passing additional arguments to the trackers.
# All tracker configs must inherit from this class, and also set the extra="allow" flag.
class TrackerConfig(BaseModel, extra="allow"):
    type: TrackerTypes | Literal["loguru", "mlflow", "disabled"]
    enqueue: bool = True


class BaseTracker(ABC):
    """
    Base class for experiment trackers.

    Trackers receive lifecycle callbacks during a search, allowing implementations
    to log parameters, scores, and metadata to any backend (files, MLflow, etc.).

    Lifecycle (in call order):
        setup → search_starts → [generation_starts → [pre_objective → post_objective]* →
        generation_ends]* → search_ends → cleanup
    """

    tracker_type: TrackerTypes
    tracker_config: TrackerConfig
    kwargs: dict
    search_title: Optional[str] = None

    def __init__(self, tracker_config: TrackerConfig):
        """
        Initialize the tracker from a config, storing a defensive copy.
        Calls ``setup()`` automatically after initialization.
        """
        self.tracker_type = TrackerTypes(tracker_config.type)
        self.tracker_config = tracker_config.model_copy()

        self.setup()

    def setup(self, **kwargs):
        """Initialize tracker resources (logging sinks, connections, etc.).

        Called once during ``__init__`` and again inside worker subprocesses
        that need to re-establish tracker state (e.g., RayWorker remote tasks).
        """
        pass

    def cleanup(self, **kwargs):
        """Release tracker resources. Called once after the search completes."""
        pass

    @abstractmethod
    def search_starts(self, **kwargs):
        """Called when the search begins, before any generations run."""
        pass

    @abstractmethod
    def search_ends(self, **kwargs):
        """Called when the search finishes, after all generations complete."""
        pass

    @abstractmethod
    def generation_starts(self, generation_number: int, **kwargs):
        """Called at the start of each generation."""
        pass

    @abstractmethod
    def generation_ends(self, generation_number: int, **kwargs):
        """Called at the end of each generation."""
        pass

    @abstractmethod
    def pre_objective(self, config: ParamArg, **kwargs):
        """Called before each objective function evaluation.

        Args:
            config: The parameters being evaluated.
            **kwargs: Worker-specific metadata (e.g., ``task_id``, ``node_id``).
        """
        pass

    @abstractmethod
    def post_objective(self, scores, **kwargs):
        """Called after each objective function evaluation.

        Args:
            scores: JSON-serialized Score from the evaluation.
            **kwargs: Worker-specific metadata (e.g., ``task_id``).
        """
        pass


class Worker(ABC):
    """
    Minimal specification for a worker.
    """

    def __init__(
        self,
        cache_folder: str | os.PathLike,
        normalize: bool = False,
        tracker: BaseTracker | None = None,
        respects_num_workers: bool = False,
        supports_timeout: bool = False,
    ):
        # Initialize internal properties here, but @property methods are the
        # primary public interface so concrete implementations can override
        # functionality as needed.
        from compileiq.tracker import DisabledTracker

        self._normalize = normalize
        self._baseline_score = None
        self._cache_dir = cache_folder
        self._current_generation = 0
        self._tracker: BaseTracker = tracker if tracker is not None else DisabledTracker()
        self._respects_num_workers = respects_num_workers
        self._supports_timeout = supports_timeout

    @property
    def normalize(self) -> bool:
        """Whether normalization should be applied."""
        return self._normalize

    @normalize.setter
    def normalize(self, new_value: bool):
        """Set whether normalization should be applied."""
        self._normalize = new_value

    @property
    def baseline_score(self):
        """The baseline score used for normalization."""
        return self._baseline_score

    @baseline_score.setter
    def baseline_score(self, new_value):
        """Set the baseline score."""
        self._baseline_score = new_value

    @property
    def cache_dir(self) -> str | os.PathLike:
        """The cache directory for this worker."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, new_value: str | os.PathLike):
        """Set the cache directory."""
        self._cache_dir = new_value

    @property
    def current_generation(self) -> int:
        """The current generation number."""
        return self._current_generation

    @current_generation.setter
    def current_generation(self, new_value: int):
        """Set the current generation number."""
        self._current_generation = new_value

    @property
    def tracker(self) -> BaseTracker:
        """The tracker instance."""
        return self._tracker

    @tracker.setter
    def tracker(self, new_value: BaseTracker):
        """Set the tracker instance."""
        self._tracker = new_value

    @property
    def respects_num_workers(self) -> bool:
        """Whether this worker respects the `num_workers` parameter on its `run` method."""
        return self._respects_num_workers

    @property
    def supports_timeout(self) -> bool:
        """Whether this worker supports the `task_timeout` parameter on its `run` method."""
        return self._supports_timeout

    @abstractmethod
    def run(
        self,
        *,
        function: Callable,
        params_pool: Sequence[ParamArg],
        params_ids: Sequence[int],
        num_function_returns: int = 1,
        num_workers: int = 1,
        **kwargs,
    ) -> list[Score]:
        """Execute the objective function across a pool of parameters."""
        pass

    @classmethod
    @abstractmethod
    def create(
        cls,
        cache_folder: str | os.PathLike,
        normalize: bool,
        tracker: BaseTracker | None,
    ) -> "Worker":
        """Construct a worker instance with the given configuration."""
        pass

    @overload
    @staticmethod
    def normalize_scores(
        current_score: SingleScore, baseline_score: SingleScore
    ) -> SingleScore: ...

    @overload
    @staticmethod
    def normalize_scores(current_score: MultiScore, baseline_score: MultiScore) -> MultiScore: ...

    @overload
    @staticmethod
    def normalize_scores(
        current_score: SingleScore | MultiScore,
        baseline_score: SingleScore | MultiScore,
    ) -> SingleScore | MultiScore: ...

    @staticmethod
    def normalize_scores(
        current_score: SingleScore | MultiScore,
        baseline_score: SingleScore | MultiScore,
    ) -> SingleScore | MultiScore:
        """
        We normalize the scores when `normalize=True` as there may be run-to-run variations.

        Args:
            current_score:
                The current score(s) to be normalized.
            baseline_score:
                The baseline score(s) used for normalization.
        Returns:
            The normalized score(s).
        """
        if baseline_score is None:
            raise RuntimeError("Trying to normalize score without configuring a baseline value.")

        if isinstance(current_score, (int, float)):
            assert isinstance(baseline_score, (int, float, str))
            return Worker._norm(current_score, baseline_score)
        if isinstance(current_score, (list, tuple)):
            assert isinstance(baseline_score, (list, tuple))
            return [Worker._norm(score, baseline_score[i]) for i, score in enumerate(current_score)]
        raise ValueError("Score type not recognized for normalization.")

    @staticmethod
    def _norm(score: SingleScore, baseline_score: SingleScore) -> SingleScore:
        if (
            isinstance(score, (int, float))
            and isinstance(baseline_score, (int, float))
            and baseline_score != 0
        ):
            return score / baseline_score
        return INVALID_SCORE

    @staticmethod
    def invalidate_score(num_objectives: int) -> SingleScore | MultiScore:
        """Return an invalid score for the given number of objectives."""
        if num_objectives == 1:
            return INVALID_SCORE
        else:
            return cast(MultiScore, [INVALID_SCORE] * num_objectives)


class WorkerTypes(StrEnum):
    """
    Chooses between the available classes for workers.
    Workers are responsible for running the user objective function
    Different Worker types will use different libs and configurations for
    executing the function in parallel, locally or distributed

    Attributes:
        DEFAULT:
            This is set to be the same as WorkerTypes.NATIVE

        NATIVE:
            Uses the native Python multiprocessing library to spawn new processes that will
            pick 'work' from a queue until all generations are complete.
            Supports: Only local parallelism

        ISOLATED:
            Same as `NATIVE`, but each evaluation spawns a new process. Useful if your objective
            function has memory leaks or other side effects that may affect other evaluations.
            If a timeout is set, it will kill the entire process.
            This is the most robust execution type, but also the slowest.
            Supports: Only local parallelism

        RAY:
            Uses RayTune Cluster and remote features to run your workload.
            Allows for seamless executing in clusters as long as you have the RayCluster
            properly configured.
            It is your responsibility to have a Ray Cluster configured to enable distributed
            execution. `Supports: Local and Distributed Parallelism`
            Ray Documentation:
                [Get Started](https://docs.ray.io/en/latest/cluster/getting-started.html)
                [Deploy On-Premise](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)

        ASYNC:
            Uses Python's asyncio library to run your workload. The objective function must be
            of type `async` and be prepared to take advantage of concurrency.
    """

    RAY = "ray"
    NATIVE = "native"
    ISOLATED = "isolated"
    ASYNC = "async"
    DEFAULT = NATIVE

    def worker_type(self) -> type[Worker]:
        """Return the Worker subclass for this type."""
        from compileiq.worker import (
            MultiProcessWorker,
            RayWorker,
            AsyncWorker,
            IsoMultiProcessWorker,
        )

        _MAP = {
            "native": MultiProcessWorker,
            "ray": RayWorker,
            "isolated": IsoMultiProcessWorker,
            "async": AsyncWorker,
        }
        return _MAP[self.value]


class DisabledTrackerConfig(TrackerConfig):
    """
    DisabledTrackerConfig is a configuration to disable tracking.

    Attributes:
        type: The type of tracker it creates.
    """

    type: Literal[TrackerTypes.DISABLED] = TrackerTypes.DISABLED  # pyright: ignore[reportIncompatibleVariableOverride]


class DefaultTrackerConfig(DisabledTrackerConfig):
    """
    DefaultTrackerConfig is a configuration to use the default tracker.
    """

    type: Literal[TrackerTypes.DEFAULT] = TrackerTypes.DEFAULT


class LoguruTrackerConfig(TrackerConfig, extra="allow", arbitrary_types_allowed=True):
    """
    LoguruTrackerConfig is a configuration for the LoguruTracker.
    By default, it will log to stderr.

    Attributes:
        type: The type of tracker it creates.
    """

    type: Literal[TrackerTypes.LOGURU] = TrackerTypes.LOGURU  # pyright: ignore[reportIncompatibleVariableOverride]
    sink: str | SkipValidation[TextIO] | SkipValidation[List] | None = None
    enqueue: bool = True
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[task_id]} | {message}"

    @field_validator("sink", mode="after")
    def force_list(cls, v):
        if not isinstance(v, list):
            return [v]
        return v


class MLflowTrackerConfig(TrackerConfig, extra="allow"):
    """
    MLflowTrackerConfig is a configuration for the MLflowTracker.

    Attributes:
        type: The type of tracker it creates.
        experiment_name: The name of the experiment to use.
        run_name: The name of the run to use.
        tracking_uri: The URI of the tracking server to use.
        description: The description of the run to use.
        score_names: The names of the scores to use.
        log_config: Whether to log the config for each objective function call.
    """

    type: Literal[TrackerTypes.MLFLOW] = TrackerTypes.MLFLOW  # pyright: ignore[reportIncompatibleVariableOverride]
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    description: Optional[str] = None
    score_names: list = []
    log_config: bool = True


class SearchConfiguration(BaseModel, extra="forbid"):
    """
    Supported values to configure your search behavior.
    """

    problem_type: ProblemType | Literal["min", "max"] = Field(
        default=ProblemType.MIN, description="If it is minimization or maximization problem"
    )
    normalize: bool = Field(
        default=False,
        description="Controls if the scores are normalized by CompileIQ. "
        "Mostly recommended if using multi-machine or multi-gpu setups.",
    )
    num_objectives: int = Field(
        default=1,
        gt=0,
        description="Should match the number of returns from your objective function",
    )
    generations: int = Field(
        gt=0,
        description="The number of search iterations to run before halting. "
        "The larger this value, the more solutions will be evaluated, but it may also lead "
        "to better results.",
    )
    pool_size: int | None = Field(
        default=None,
        gt=5,
        description="The batch size of evaluations for each search iteration. "
        "If set to None, we calculate it based on your number of objectives.",
    )
    cull_size: int | None = Field(
        default=None,
        gt=1,
        multiple_of=2,
        description="The number of parents in a given generation that will become the progenitors "
        "of the next generation. The difference between pool size and cull size are the survivors "
        "of the next generation. If set to None, we calculate it based on pool size.",
    )
    mutate_rate: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="The chance a sampled candidate will be perturbed between iterations.",
    )
    objective_weights: Optional[list[float]] = Field(
        default=None,
        description="If you are doing multiobjective search you may want to weight-in more scores. "
        "This will affect how we generate the algorithm reference directions.",
    )
    init_with_true_random_threshold: Optional[float] = Field(
        default=0.9,
        description="Controls what sampling percentage is performed at seed-low "
        "and seed-high vs normal range. Defaults to 0.9, so 90% will be sampled from the "
        "`seed-high` and `seed-low` range, and 10% from `start` and `end`.",
    )
    enable_large_fail_pool: bool = Field(
        default=True,
        description="When a generation does not achieve at least 20pct of passing solutions it"
        " will not move into the next. The subsequent sample batch will only contain "
        "the minimal number of passing samples required for this generation to pass. "
        "If this is set to True, it will resubmit an entire pool size length batch.",
    )

    @model_validator(mode="after")
    def validate_weights(self):
        if self.objective_weights is not None:
            if len(self.objective_weights) != self.num_objectives:
                raise ValueError(
                    f"Length of objective weights ({self.objective_weights}) "
                    f"must be equal to num_objectives ({self.num_objectives})"
                )
            if sum(self.objective_weights) != 1.0:
                raise ValueError("Objective Weights must add up to 1.0")
        return self

    @model_validator(mode="after")
    def set_pool_and_cull_sizes(self):
        target = (2 * self.num_objectives) + 1
        cull_pct = 0.75
        if self.pool_size is None:
            pool_size = max(ceil(target / (1 - cull_pct)), 32)
            self.pool_size = pool_size + pool_size % 2

        if self.pool_size is None:
            raise RuntimeError("pool_size must be populated before deriving cull_size")

        if self.cull_size is None:
            min_survivors = 1 + 2 * self.num_objectives
            cull_size = int(self.pool_size * cull_pct)
            cull_size = cull_size - cull_size % 2  # round down to even
            # Cap so that pool_size - cull_size >= min_survivors
            max_cull = self.pool_size - min_survivors
            max_cull = max_cull - max_cull % 2  # round down to even
            self.cull_size = max(2, min(cull_size, max_cull))

        return self

    @model_validator(mode="after")
    def validate_pool_and_cull_sizes(self):
        if self.pool_size is None or self.cull_size is None:
            raise RuntimeError("pool_size and cull_size must be populated before validation")

        if self.cull_size >= self.pool_size:
            raise ValueError(
                f"cull_size ({self.cull_size}) must be less than pool_size ({self.pool_size})"
            )

        num_survivors = self.pool_size - self.cull_size
        min_survivors = 1 + 2 * self.num_objectives
        if num_survivors < min_survivors:
            raise ValueError(
                f"Number of survivors (pool_size ({self.pool_size}) - cull_size ({self.cull_size}))"
                f" is too small. Need at least {min_survivors}."
            )

        def _auto_find_num_dir_vectors(
            num_survivors: int,
            num_objectives: int,
        ) -> int:
            outer_size = comb(num_objectives, 1)
            remaining = num_survivors - outer_size - 1
            if remaining <= 0:
                return outer_size
            inner_size = 0
            for p in range(1, num_survivors + 1):
                count = comb(num_objectives + p - 1, p)
                if count > remaining:
                    break
                inner_size = count
            if inner_size == outer_size:
                return outer_size
            return outer_size + inner_size

        # There are constraints between pool, cull and the way we build reference directions
        n_dir = _auto_find_num_dir_vectors(num_survivors, self.num_objectives)
        if self.pool_size < n_dir:
            raise ValueError(
                f"pool_size ({self.pool_size}) is too small. Needs to be at least "
                f"{n_dir} for {self.num_objectives} objectives."
            )

        return self

    def to_legacy(self) -> str:
        """Convert configuration into expected legacy format"""
        class_dict = self.model_dump()

        legacy_string = f";THIS FILE WAS GENERATED USING COMPILEIQ {version('compileiq')}\n\n"
        for key, val in class_dict.items():
            if val is not None:
                if "problem_type" in key:
                    value = "#t" if ProblemType(val) == ProblemType.MIN else "#f"
                    legacy_string += f"(seek_minimum . {value})\n"
                    continue
                elif isinstance(val, bool):
                    # Core source expects #t and #f for bool
                    val = "#t" if val else "#f"
                elif isinstance(val, str):
                    # We need to explicit add quotes for strings
                    val = f'"{val}"'
                elif isinstance(val, tuple) or isinstance(val, list):
                    if isinstance(val[0], str):
                        val = '("' + '" "'.join(val) + '")'
                    else:
                        val = "((" + " ".join(map(str, val)) + "))"

                if "normalize" in key:
                    value = "#f" if self.normalize else "#t"
                    legacy_string += f"(qualitative . {value})\n"
                else:
                    legacy_string += f"({key} . {val})\n"

        return legacy_string

    def to_json_dict(self) -> dict:
        """Convert configuration into JSON dict for core consumption."""
        class_dict = self.model_dump(mode="json")
        return {k: v for k, v in class_dict.items() if v is not None}

    @classmethod
    def from_legacy(cls, legacy: str):
        """
        Converts legacy lisp-like format into the SearchConfiguration Object.

        Warning:
            It will ignore all fields that are not available at SearchConfiguration

        Warning:
            It will raise exceptions if using fields that are unsupported in CompileIQ
        """
        if legacy.endswith(".config") and pathlib.Path(legacy).exists():
            with open(legacy, "r") as f:
                legacy_str = f.read()
        else:
            raise ValueError(
                "Search Configuration file is missing the extension .config or does not exist"
            )

        # Extracting keys and values while ignoring comments
        legacy_values: List[str] = re.findall(
            r"^(?!\;)(?:\(\s*)(.+) \. (.+)(?:\s*\))",
            legacy_str,
            flags=re.RegexFlag.MULTILINE,
        )
        available_fields = cls.model_fields.keys()

        # Building dictionary with pairs for SearchConfiguration, ignoring non-available fields
        config_dict = {}
        for key, val in legacy_values:
            val = val.replace("#t", str(True)).replace("#f", str(False))
            if "seek_minimum" in key:
                config_dict["problem_type"] = ProblemType.MIN if val == "True" else ProblemType.MAX
            elif key in available_fields:
                config_dict[key] = ast.literal_eval(val)
            else:
                warnings.warn(f"Configuration {key} is going to be ignored")

        return cls(**config_dict)


class InternalSearchConfiguration(SearchConfiguration):
    """
    Used internally to construct the core search-space config.
    These are fields CompileIQ overwrites and the user should not set by themselves.
    """

    dna_config: str | List[str] = ""
    enable_result_file: bool = False

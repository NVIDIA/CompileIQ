import pathlib
import os
import sys
import shutil
import socket
import json
import warnings
from datetime import datetime
from tqdm.auto import tqdm
from uuid import uuid4
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    ConfigDict,
    PrivateAttr,
    field_validator,
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeAlias,
    cast,
)
from compileiq.utils.validation import Score, SingleScore, MultiScore
from compileiq.tracker import (
    _TRACKER_TYPES_TO_CLASSES,
)
from compileiq.config.const import _CACHE_DIR, KEEP_CACHE_FILES
from compileiq.types import (
    BaseTracker,
    ParamArg,
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
from compileiq.core.core_comms import CoreIPC, initialize_socket
from compileiq.core.core_types import (
    ParameterSet,
    CompletionMessage,
    EvaluatedParamResponse,
    ResponseTemplate,
)
from compileiq.utils._setup_files import (
    setup_legacy_search_config,
    setup_search_space,
    get_core_filepaths,
)
from compileiq.utils.helpers import (
    restore_nested_search_space,
    _decode_from_core,
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


def _expand_objective_values(
    value: SingleScore | MultiScore, num_objectives: int
) -> list[int | float | str]:
    if num_objectives > 1:
        assert isinstance(
            value, (list, tuple)
        ), f"num_objectives > 1 requires list/tuple score; got {type(value).__name__}"
        return cast(list[int | float | str], list(value))
    assert not isinstance(value, (list, tuple))
    return [value]


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
            "RAY (distributed via Ray), or ASYNC (asyncio concurrency). "
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
            "When enabled, the cached log from the core subprocess is not deleted, "
            "allowing inspection of its output."
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

    ## Private Attributes
    _create_new_id: Callable = lambda _: (
        datetime.now().strftime("%Y-%m-%d-%H_%M_%S-") + str(uuid4())
    )
    # CompileIQ Core will give us the id once generation starts
    run_id: int | None = Field(None, init=False)
    _worker: Worker = PrivateAttr()
    _tracker: BaseTracker = PrivateAttr()
    current_generation: int = Field(
        0, init=False, description="Current generation of the search, starting at 0."
    )
    _search_config: InternalSearchConfiguration = PrivateAttr()
    _result: SearchResult = PrivateAttr()
    _using_file_backed_search_space: bool | list[bool] = False
    _multi_config: bool = False
    _search_space_resolution_metadata: list[SearchSpaceResolutionMetadataDict] | None = (
        PrivateAttr(default=None)
    )

    # Cache directory management
    _base_cache_dir: Optional[pathlib.Path] = PrivateAttr(default=None)

    # Communication with Core
    _listen_socket: socket.socket = PrivateAttr(default_factory=initialize_socket)
    _core_socket: Optional[socket.socket] = PrivateAttr(default=None)
    _core_ipc: CoreIPC = PrivateAttr()

    # Pydantic config
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("search_space", mode="after")
    def validate_windows(cls, value):
        if sys.platform == "win32":
            if isinstance(value, list):
                # If using multiple configs
                raise ValueError("Windows does not support multiple config search spaces")

        return value

    @field_validator("search_config", mode="after")
    def normalize_search_config(cls, value: SearchConfigInput) -> SearchConfigInput:
        if isinstance(value, SearchConfiguration):
            return value
        if isinstance(value, dict):
            return SearchConfiguration(**value)
        return SearchConfiguration.from_legacy(str(value))

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

        _, search_space_config_filepath = get_core_filepaths(str(self.cache_folder))

        # Windows workaround with paths
        if sys.platform == "win32":
            search_space_config_filepath = search_space_config_filepath.replace("\\", "\\\\")

        self._search_config.dna_config = search_space_config_filepath

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
        self._multi_config = isinstance(self.search_space, list)
        if isinstance(self.search_space, list):
            self._using_file_backed_search_space = [
                isinstance(sspace, pathlib.Path) for sspace in self.search_space
            ]
        else:
            self._using_file_backed_search_space = isinstance(self.search_space, pathlib.Path)

        # Initializing Core IPC
        self._core_ipc = CoreIPC()

        if self.tracker_config.type is None:
            raise ValueError("Tracker is not initialized")

        if not isinstance(self.tracker_config, TrackerConfig):
            raise ValueError(
                f"Tracker configuration is not a TrackerConfig, got {type(self.tracker_config)}"
            )
        self._tracker = _TRACKER_TYPES_TO_CLASSES[TrackerTypes(self.tracker_config.type)](
            self.tracker_config
        )

        wt = self.worker_type
        if isinstance(wt, WorkerTypes):
            worker_cls = wt.worker_type()
        elif isinstance(wt, type) and issubclass(wt, Worker):
            worker_cls = wt
        else:
            raise RuntimeError(f"Expected a WorkerTypes or Worker subclass, but found {wt}")

        assert self.cache_folder is not None, "_init_folders populates cache_folder before _setup"
        self._worker = worker_cls.create(
            cache_folder=self.cache_folder,
            normalize=self._search_config.normalize,
            tracker=self._tracker,
        )

    @model_validator(mode="after")
    def _setup(self):
        self._do_setup()
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del self

    def __del__(self):
        if hasattr(self, "_listen_socket") and self._listen_socket is not None:
            self._listen_socket.close()
        if hasattr(self, "_worker") and self._worker is not None:
            del self._worker
        if hasattr(self, "_core_ipc") and self._core_ipc is not None:
            del self._core_ipc
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
        assert self.cache_folder is not None
        main_config_filepath, search_space_config_filepath = get_core_filepaths(self.cache_folder)

        hijacked_config = self._search_config.model_copy(deep=True)
        hijacked_config.num_objectives = 1
        hijacked_config.pool_size = max(num_samples, 6)
        hijacked_config.cull_size = 2
        try:
            hijacked_config.dna_config = setup_search_space(
                cast(ResolvedSearchSpaceInput, self.search_space), search_space_config_filepath
            )
            setup_legacy_search_config(hijacked_config, main_config_filepath)

            # Starting Core as a subprocess
            _ = self._core_ipc.start(
                server_socket=self._listen_socket,
                main_config_filepath=main_config_filepath,
                silent=not self.debug,
            )

            # Wait for core to connect (returns accepted socket and its addr)
            self._core_socket, _ = self._listen_socket.accept()

            # Executing function throughout the generations
            parameter_sets = self._core_ipc.receive_from_core(self._core_socket)
            if isinstance(parameter_sets, CompletionMessage):
                raise RuntimeError(
                    "Something went wrong with the core, enable `debug=True` for debugging."
                )

            func_args = self._load_params(parameter_sets)

        finally:
            self._core_ipc.stop()
            if not KEEP_CACHE_FILES:
                if isinstance(hijacked_config.dna_config, list):
                    for path in hijacked_config.dna_config:
                        os.remove(path)
                else:
                    os.remove(hijacked_config.dna_config)
                os.remove(main_config_filepath)

        return func_args[:num_samples]

    def start(
        self,
        num_workers: int | None = None,
        task_timeout: int | float | None = None,
        **additional_worker_kwargs,
    ) -> SearchResult:
        """
        The CompileIQ core is started as a subprocess through here.

        The communication between python process and the subprocess is done through sockets:

        1. During __init__ we prepare the python socket server for communication with the
           core subprocess.
        2. The `search_space.json` and `main_config.json` files are created inside the cache
           folder. (Core needs these.)
        3. We start the core process with the correct environment variables and wait for
           communication.
        4. The core process will start sending parameter candidates which we will execute
           using multiprocess.
        5. The Python process returns the scores through the socket.
        6. At some point the core process sends a completion flag indicating the end of the
           tune process.

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

            **additional_worker_kwargs:
                Additional keyword arguments forwarded to the worker's ``run()`` method.
                Each worker accepts the kwargs it cares about. For example:

                - RayWorker: accepts Ray task resource options
                - Custom workers: accept any kwargs they define

        Returns (`SearchResult`):
            An object with the search results.
        """

        if num_workers is not None:
            if not self._worker.respects_num_workers:
                warnings.warn(
                    f"num_workers is not supported by {type(self._worker).__name__}", stacklevel=2
                )
            elif num_workers < 1:
                raise ValueError("num_workers must be a positive integer.")

        if task_timeout is not None and not self._worker.supports_timeout:
            warnings.warn(
                f"task_timeout is not supported by {type(self._worker).__name__}", stacklevel=2
            )

        worker_count = num_workers if isinstance(num_workers, int) else 1

        if self.cache_folder is None or not self.cache_folder.exists():
            self._do_init_folders()
        assert self.cache_folder is not None

        # Initializing Result df
        self._result = SearchResult._initialize_empty(
            num_scores=self._search_config.num_objectives,
            problem_type=self._search_config.problem_type,
            norm_scores=self._search_config.normalize,
        )

        main_config_filepath, search_space_config_filepath = get_core_filepaths(self.cache_folder)
        try:
            # Configure the core input files.
            self._search_config.dna_config = setup_search_space(
                cast(ResolvedSearchSpaceInput, self.search_space), search_space_config_filepath
            )
            setup_legacy_search_config(self._search_config, main_config_filepath)

            # Starting Core as a subprocess
            _ = self._core_ipc.start(
                server_socket=self._listen_socket,
                main_config_filepath=main_config_filepath,
                silent=not self.debug,
            )

            # Wait for core to connect (returns accepted socket and its addr)
            self._core_socket, _ = self._listen_socket.accept()

            self._tracker.search_starts(
                search_space_resolution_metadata=self._search_space_resolution_metadata
            )

            # Executing function throughout the generations
            self._result = self._process_candidates(
                worker_count,
                task_timeout=task_timeout,
                **additional_worker_kwargs,
            )

        finally:
            self._tracker.search_ends()
            self.current_generation = 0
            self._core_ipc.stop()
            if not KEEP_CACHE_FILES:
                self._clean_files()

        return self._result

    @property
    def search_space_resolution_metadata(self) -> list[SearchSpaceResolutionMetadataDict] | None:
        """One metadata record per resolved provider, or None if no provider was used."""
        if self._search_space_resolution_metadata is None:
            return None
        return [dict(metadata) for metadata in self._search_space_resolution_metadata]

    def _process_candidates(
        self,
        num_workers: int,
        task_timeout: Optional[int | float] = None,
        **additional_worker_kwargs,
    ) -> SearchResult:
        """
        We receive the parameter set, call upon the worker to execute the objective function, and
        send the score back through the socket, until we receive a completion message.
        """
        pbar = tqdm(
            total=self._search_config.generations,
            ascii="░▒█",
            colour="green",
            disable=self.disable_progress_bar,
            smoothing=0.8,
            bar_format="{desc} {n_fmt}/{total_fmt}|{bar}| [elapsed: {elapsed} · eta: {remaining}]"
            " {postfix}",
        )
        pbar.set_description("🧬 Generation")
        while True:
            try:
                # receiving params ('knobs') from core subprocess
                assert self._core_socket is not None
                parameter_sets = self._core_ipc.receive_from_core(self._core_socket)
            except Exception as e:
                pbar.close()
                raise e

            # Verify if the message signals the end of the run
            if isinstance(parameter_sets, CompletionMessage):
                if not (parameter_sets.complete):
                    raise RuntimeError(
                        "Something went wrong with the core, enable `debug=True` for debugging."
                    )

                pbar.update()
                self._result.clear_duplicates()
                return self._result

            else:
                # Standard flow receiving knobs
                self.run_id = parameter_sets.invocation_id

                # Updating Progress Bar and Tracker
                if parameter_sets.generation_num != self.current_generation:
                    self.current_generation = parameter_sets.generation_num
                    self._worker.current_generation = self.current_generation

                    if self._search_config.num_objectives == 1 and self.current_generation > 0:
                        # We limit best score display to single objective because multi-objective is
                        # a pareto front and it's not straightforward to define a single "best"
                        try:
                            best_score = self._result.get_best_result()
                        except Exception:
                            best_score = None

                        if isinstance(best_score, dict):
                            gen_from_best = int(best_score["generation"])
                            best_score_value = (
                                best_score["score_1"]
                                if not self._search_config.normalize
                                else best_score["norm_score_1"]
                            )
                            pbar.set_postfix(
                                {
                                    "🏆 best_score": f"{best_score_value:.4f}",
                                    "at_gen": gen_from_best,
                                }
                            )

                    pbar.update()
                    self._tracker.generation_starts(self.current_generation)

                elif self.current_generation == 0:
                    self._tracker.generation_starts(0)

                # Processing parameters into dictionaries (if possible)
                func_args = self._load_params(parameter_sets)
                param_ids = [single_param.id for single_param in parameter_sets.params]
                scores = self._worker.run(
                    function=self.objective_function,
                    tracker=self._tracker,
                    params_pool=func_args,
                    params_ids=param_ids,
                    num_workers=num_workers,
                    num_function_returns=self._search_config.num_objectives,
                    task_timeout=task_timeout,
                    **additional_worker_kwargs,
                )

                if len(scores) < len(parameter_sets.params):
                    raise RuntimeError(
                        "The worker returned less scores than the number of parameter sets "
                        "requested by the core."
                    )

                if (
                    self.exit_on_failure
                    and self.current_generation == 0
                    and self._check_fail_count(scores)
                ):
                    raise RuntimeError(
                        "All objective functions failed in the first gen. Are you sure this "
                        "is expected behavior? You can disable this error by setting"
                        "`exit_on_failure=False`."
                    )

                # Storing worker results and preparing response to Core
                scores_response = ResponseTemplate(evaluated_params=[])
                assert isinstance(scores_response.evaluated_params, list)
                num_objectives = self._search_config.num_objectives
                for score in scores:
                    self._result.add_result(
                        score, parameter_sets.generation_num, self._search_config.normalize
                    )

                    # Preparing response to Core
                    # Baselines are not reported back to core
                    if not score.is_baseline:
                        raw_score_value = (
                            score.norm_score if self._search_config.normalize else score.score
                        )
                        assert raw_score_value is not None
                        score_value = _expand_objective_values(raw_score_value, num_objectives)
                        assert isinstance(score.param_id, int), (
                            f"non-baseline score must have int param_id; got "
                            f"{type(score.param_id).__name__}"
                        )

                        response = EvaluatedParamResponse(id=score.param_id, scores=score_value)
                        scores_response.evaluated_params.append(response)

                # Verifying all scores are returned
                if len(param_ids) != len(scores_response.evaluated_params):
                    raise RuntimeError(
                        "Worker did not return all scores passed down for calculation."
                        f"Sent {len(param_ids)} and only returned "
                        f"{len(scores_response.evaluated_params)}"
                    )

                # Checkpointing intermediate results every batch
                if self.dump_results is not None:
                    self._result.save(self.dump_results)

                self._tracker.generation_ends(self.current_generation)

                # Sending scores to the Core subprocess
                self._core_ipc.send_to_core(
                    self._core_socket,
                    scores_response,
                )

    def _load_params(self, parameter_sets: ParameterSet) -> List[ParamArg]:
        """
        We verify if the params (knobs) received from core are json-compatible
        otherwise we will send them as strings and is the responsibility of the user to
        manage the string at its own objective function.
        """

        def parse_param_payload(param: str, is_file_backed: bool = False):
            try:
                param_set = json.loads(param)
            except (ValueError, json.JSONDecodeError):
                try:
                    import json5

                    param_set = json5.loads(param)
                except Exception:
                    return param

            # We only support nested data for dictionary-defined search spaces.
            if not is_file_backed and isinstance(param_set, dict):
                # Nested restoration only applies to dictionary-defined search spaces.
                param_set = restore_nested_search_space(param_set)

            return param_set

        params_from_generation: List[ParamArg] = []
        for param in parameter_sets.params:
            if self._multi_config:
                assert isinstance(self._using_file_backed_search_space, list)
                # When specifying multiple configs, a list of base64 strings is returned
                # Each string is the representation from one config (in the same order as
                # provided by the user)
                decoded_param = list(map(_decode_from_core, json.loads(param.knobs)))
                single_pset = [
                    parse_param_payload(single_param, self._using_file_backed_search_space[i])
                    for i, single_param in enumerate(decoded_param)
                ]
            else:
                assert isinstance(self._using_file_backed_search_space, bool)
                single_pset = parse_param_payload(
                    param.knobs, self._using_file_backed_search_space
                )

            params_from_generation.append(cast(ParamArg, single_pset))

        return params_from_generation

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

import base64
import hashlib
import json
import pathlib
from unittest.mock import Mock

import pytest

import compileiq.search_spaces.base as ss
import compileiq.backends as backends_module
import compileiq.ciq as ciq_module
from compileiq.backends import (
    BackendArtifactInfo,
    BackendCandidate,
    BackendCapabilities,
    BackendCapabilityError,
    BackendRestoreResult,
    BackendNotFoundError,
    CheckpointValidationError,
    EngineConfig,
    PendingCandidatePolicy,
    SearchBackend,
    register_backend,
    get_backend_registration,
    unregister_backend,
)
from compileiq.ciq import CheckpointCompatibilityWarning, Search
from compileiq.types import (
    LoguruTrackerConfig,
    SearchConfiguration,
    TrackerConfig,
    TrackerTypes,
    Worker,
    WorkerTypes,
)
from compileiq.utils.validation import BASELINE_CONFIG, INVALID_SCORE, Score
from compileiq.worker import IsoMultiProcessWorker, ManualWorker


class FakeSealedBackend(SearchBackend):
    """Deterministic backend double for public lifecycle and checkpoint delegation tests."""

    def __init__(self, *, search_config, **kwargs):
        """Capture only the state needed to observe orchestration decisions."""

        search_space = kwargs.pop("search_space", None)
        del kwargs
        self.pool_size = search_config.pool_size
        self._protected = isinstance(search_space, pathlib.Path)
        self._generation = 0
        self._pending = []
        self._replacement_deficit = 0
        self.ask_calls = 0
        self.checkpoint_calls = 0
        self.state = b"protected-state-marker"
        self._accepted_in_generation = False

    @property
    def current_generation(self) -> int:
        """Expose generation advancement controlled by accepted reports."""

        return self._generation

    @property
    def pending_candidates(self) -> tuple[BackendCandidate, ...]:
        """Return a detached snapshot of provider-owned pending work."""

        return tuple(candidate.model_copy(deep=True) for candidate in self._pending)

    @property
    def has_accepted_results_in_current_generation(self) -> bool:
        """Expose whether an earlier subset contained a successful evaluation."""

        return self._accepted_in_generation

    def ask(self, num_samples=None):
        """Issue stable candidates, honoring any deficit created by discard restore."""

        self.ask_calls += 1
        count = num_samples or self._replacement_deficit or self.pool_size
        self._replacement_deficit = max(0, self._replacement_deficit - count)
        self._pending = [
            BackendCandidate(
                param_id=f"candidate-{self._generation}-{idx}",
                params=f"opaque-{idx}" if self._protected else self._candidate_params(idx),
            )
            for idx in range(count)
        ]
        self._accepted_in_generation = False
        return [candidate.model_copy(deep=True) for candidate in self._pending]

    def _candidate_params(self, index):
        """Build ordinary candidate parameters, with a test seam for nested values."""

        return {"x": index}

    def tell(self, scores):
        """Accept a pending subset and advance only after the batch is complete."""

        reported_ids = {str(score.param_id) for score in scores}
        pending_ids = {str(candidate.param_id) for candidate in self._pending}
        assert reported_ids <= pending_ids
        self._accepted_in_generation = self._accepted_in_generation or any(
            not score.failed for score in scores
        )
        self._pending = [
            candidate for candidate in self._pending if str(candidate.param_id) not in reported_ids
        ]
        if not self._pending:
            self._generation += 1
            self._accepted_in_generation = False

    def checkpoint(self) -> bytes:
        """Serialize provider-owned state without simulating production sealing."""

        self.checkpoint_calls += 1
        return json.dumps(
            {
                "generation": self._generation,
                "pending": [candidate.model_dump(mode="json") for candidate in self._pending],
                "replacement_deficit": self._replacement_deficit,
                "state": self.state.hex(),
            },
            sort_keys=True,
        ).encode()

    def restore(
        self,
        sealed_blob: bytes,
        *,
        pending_policy: PendingCandidatePolicy = PendingCandidatePolicy.RESUBMIT,
    ) -> BackendRestoreResult:
        """Restore pending work or convert it into a one-for-one replacement deficit."""

        state = json.loads(sealed_blob)
        self._generation = state["generation"]
        self._pending = [BackendCandidate.model_validate(item) for item in state["pending"]]
        self._replacement_deficit = state["replacement_deficit"]
        self.state = bytes.fromhex(state["state"])
        if pending_policy == PendingCandidatePolicy.DISCARD:
            self._replacement_deficit += len(self._pending)
            self._pending = []
        return BackendRestoreResult()


class SynchronousWorker(Worker):
    """Evaluate inline so managed-search tests are deterministic and process-free."""

    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        """Construct the test worker through the same factory contract as real workers."""

        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(
        self,
        *,
        function,
        params_pool,
        params_ids,
        num_function_returns=1,
        **kwargs,
    ):
        """Return one correctly associated Score for every requested candidate."""

        del kwargs
        return [
            Score(
                score=function(params),
                params=params,
                param_id=param_id,
                num_objectives=num_function_returns,
            )
            for params, param_id in zip(params_pool, params_ids, strict=True)
        ]


class GenerationRecordingWorker(SynchronousWorker):
    """Record which backend generation Search propagates to the worker."""

    seen_generations = []

    def run(self, **kwargs):
        """Capture the generation immediately before evaluating the batch."""

        self.seen_generations.append(self.current_generation)
        return super().run(**kwargs)


class ArgumentRecordingWorker(SynchronousWorker):
    """Capture public evaluation controls while otherwise evaluating synchronously."""

    calls = []

    def __init__(self, *, cache_folder, normalize, tracker):
        """Advertise support for both standard worker controls."""

        super().__init__(
            cache_folder=cache_folder,
            normalize=normalize,
            tracker=tracker,
            respects_num_workers=True,
            supports_timeout=True,
        )

    def run(self, **kwargs):
        """Record the exact arguments shared by evaluate() and start()."""

        self.calls.append(dict(kwargs))
        return super().run(**kwargs)


class BaselineInjectingWorker(SynchronousWorker):
    """Model normalized execution, including the public baseline sentinel row."""

    def run(
        self,
        *,
        function,
        params_pool,
        params_ids,
        num_function_returns=1,
        **kwargs,
    ):
        """Return a baseline plus opaque evaluated candidates for protected-ledger tests."""

        del kwargs
        baseline = Score(
            score=10.0,
            norm_score=1.0,
            params=BASELINE_CONFIG,
            param_id="baseline",
            num_objectives=num_function_returns,
            is_baseline=True,
        )
        evaluated = [
            Score(
                score=function(params),
                norm_score=0.5,
                params=params,
                param_id=param_id,
                num_objectives=num_function_returns,
            )
            for params, param_id in zip(params_pool, params_ids, strict=True)
        ]
        return [baseline, *evaluated]


@pytest.fixture
def fake_backend():
    """Register an isolated provider and expose instances for state-transition assertions."""

    name = "test-sealed-backend"
    instances = []

    def factory(**kwargs):
        """Track each constructed backend so tests can inspect the active instance."""

        backend = FakeSealedBackend(**kwargs)
        instances.append(backend)
        return backend

    register_backend(
        name,
        factory,
        capabilities=BackendCapabilities(
            arbitrary_batch_size=True,
            checkpoint=True,
            sealed_checkpoint=True,
        ),
        artifact=BackendArtifactInfo(
            backend_version="1.2.3",
            abi_version=1,
            artifact_sha256="a" * 64,
        ),
    )
    try:
        yield name, instances
    finally:
        unregister_backend(name)


@pytest.fixture
def uncheckpointed_backend():
    """Register a provider that, like many plugins, cannot checkpoint."""

    name = "test-uncheckpointed-backend"
    register_backend(
        name,
        FakeSealedBackend,
        capabilities=BackendCapabilities(arbitrary_batch_size=True),
        artifact=BackendArtifactInfo(
            backend_version="0.0.1",
            abi_version=0,
            artifact_sha256="b" * 64,
        ),
    )
    try:
        yield name
    finally:
        unregister_backend(name)


@pytest.fixture
def stand_in_amber(monkeypatch):
    """Serve the default engine name from a fake so tests need no native payload."""

    registration = backends_module.BackendRegistration(
        name="amber",
        factory=FakeSealedBackend,
        capabilities=BackendCapabilities(arbitrary_batch_size=True, checkpoint=True),
        artifact=BackendArtifactInfo(
            backend_version="stand-in",
            abi_version=1,
            artifact_sha256="c" * 64,
        ),
    )
    monkeypatch.setitem(backends_module._BACKENDS, "amber", registration)
    return registration


def make_search(
    provider=None,
    engine=None,
    engine_options=None,
    search_space=None,
    *,
    generations=1,
    pool_size=6,
    num_objectives=1,
    normalize=False,
    objective=None,
    worker_type: WorkerTypes | type[Worker] = SynchronousWorker,
    tracker_config: TrackerConfig | None = None,
    exit_on_failure=True,
    dump_results=None,
):
    """Build a minimal Search while keeping curated and advanced selection unambiguous."""

    if engine is not None and (provider is not None or engine_options):
        raise AssertionError("test helper selection is ambiguous")
    if engine_options and provider is None:
        raise AssertionError("engine_options need an explicit provider")
    if provider is not None:
        engine = EngineConfig(
            name=provider,
            options=engine_options or {},
        )
    search_kwargs = {}
    if tracker_config is not None:
        search_kwargs["tracker_config"] = tracker_config
    if engine is not None:
        # Omitting engine= exercises the Search default.
        search_kwargs["engine"] = engine
    return Search(
        objective_function=objective or (lambda config: config["x"]),
        search_space=search_space or {"x": ss.range(0, 10)},
        search_config=SearchConfiguration(
            generations=generations,
            pool_size=pool_size,
            num_objectives=num_objectives,
            normalize=normalize,
        ),
        worker_type=worker_type,
        exit_on_failure=exit_on_failure,
        dump_results=dump_results,
        disable_progress_bar=True,
        **search_kwargs,
    )


def score(candidate, value=1.0):
    """Create a score that preserves the candidate identity and parameter association."""

    return Score(
        score=value,
        params=candidate.params,
        param_id=candidate.param_id,
        num_objectives=1,
    )


def rewrite_checkpoint(checkpoint: pathlib.Path, **changes) -> None:
    """Edit owner-visible envelope fields while leaving sealed provider state untouched."""

    envelope = json.loads(checkpoint.read_text())
    envelope.update(changes)
    checkpoint.write_text(json.dumps(envelope))


def test_engine_config_rejects_algorithm_coordinate():
    """Keep algorithm selection out of the public engine configuration."""

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EngineConfig.model_validate({"name": "amber", "algorithm": "evo"})


def test_default_engine_resolves_to_amber(stand_in_amber):
    """Use the native amber engine when the caller does not choose one."""

    search = make_search()

    assert search.engine == "amber"
    assert search.provider == "amber"
    assert search.engine_options == {}
    assert search._backend_registration is stand_in_amber


def test_engine_config_normalizes_registered_name(stand_in_amber):
    """Treat the configured engine name as the only selection coordinate."""

    search = make_search(engine=EngineConfig(name=" AmBeR "))

    assert search._engine_identity == "amber"
    assert search.provider == "amber"


def test_removed_top_level_selectors_are_rejected():
    """Prevent duplicate provider selectors from becoming accidental public API aliases."""

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Search.model_validate(
            {
                "objective_function": lambda config: config["x"],
                "search_space": {"x": ss.range(0, 10)},
                "search_config": SearchConfiguration(generations=1, pool_size=6),
                "worker_type": SynchronousWorker,
                "backend": "amber",
            }
        )


def test_missing_native_runtime_error_requires_complete_compileiq_wheel(monkeypatch):
    """Keep a missing amber payload on the one-package public installation contract."""

    provider = "amber"
    monkeypatch.delitem(backends_module._BACKENDS, provider, raising=False)
    monkeypatch.setattr(backends_module, "_ENTRY_POINTS_LOADED", True)

    with pytest.raises(BackendNotFoundError, match="complete platform-specific 'compileiq' wheel"):
        get_backend_registration(provider)


def test_failed_entry_point_registration_remains_retryable_and_preserves_diagnostic(monkeypatch):
    """Do not cache a failed provider load or retain its partial registrations."""

    attempts = 0

    class Distribution:
        name = "compileiq-native-runtime"
        version = "1.1.0.dev0"

    class EntryPoint:
        name = "amber"
        dist = Distribution()

        @staticmethod
        def load():
            def register():
                nonlocal attempts
                attempts += 1
                raise TypeError("frontend/runtime registration mismatch")

            return register

    class EntryPoints(list):
        def select(self, *, group):
            assert group == "compileiq.backends"
            return self

    monkeypatch.setattr(backends_module, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(
        backends_module.metadata,
        "entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )

    for expected_attempts in (1, 2):
        with pytest.raises(
            backends_module.BackendError,
            match=(
                "amber.*compileiq-native-runtime 1.1.0.dev0.*"
                "frontend/runtime registration mismatch"
            ),
        ):
            get_backend_registration("amber")
        assert attempts == expected_attempts
        assert backends_module._ENTRY_POINTS_LOADED is False


def test_engine_config_selects_registered_backend(fake_backend):
    """Resolve one configured engine directly through the backend registration seam."""

    name, _ = fake_backend
    search = make_search(engine=EngineConfig(name=f" {name.upper()} "))

    assert search._engine_identity == name
    assert search.provider == name


def test_create_backend_normalizes_direct_engine_name(fake_backend):
    """Apply the same engine-name normalization outside the Search facade."""

    name, instances = fake_backend

    backend = backends_module.create_backend(
        f" {name.upper()} ",
        search_config=SearchConfiguration(generations=1, pool_size=6),
    )

    assert backend is instances[-1]


def test_registered_engine_name_needs_no_builtin_catalog():
    """Allow extensions to register a new opaque engine name directly."""

    name = "private-provider"
    register_backend(
        name,
        FakeSealedBackend,
        capabilities=BackendCapabilities(),
        artifact=BackendArtifactInfo(
            backend_version="1",
            abi_version=1,
            artifact_sha256="b" * 64,
        ),
    )
    try:
        search = make_search(engine=name)
        assert search._engine_identity == name
        assert search.provider == name
    finally:
        unregister_backend(name)


def test_tell_records_results_only_after_backend_acceptance(fake_backend):
    """Treat result recording as a commit after backend acceptance, not before it."""

    name, _ = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)
    accepted = [score(candidate, 100 + idx) for idx, candidate in enumerate(candidates)]

    search.tell(accepted)

    rows = search._result.get_results()
    assert list(rows["generation"]) == [0, 0]
    assert list(rows["score_1"]) == [100, 101]
    before = rows.copy(deep=True)
    with pytest.raises(ValueError, match="already_reported=.*candidate-0-0"):
        search.tell(accepted)
    assert search._result.get_results().equals(before)


def test_public_pending_and_result_accessors_track_accepted_state(fake_backend):
    """Expose pending work until acceptance and expose only accepted work as results."""

    name, _ = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)

    assert search.pending_candidates == tuple(candidates)
    assert search.result.get_results().empty

    search.tell([score(candidate) for candidate in candidates])

    assert search.pending_candidates == ()
    assert len(search.result.get_results()) == 2


def test_pending_candidate_snapshots_do_not_expose_provider_state(fake_backend, monkeypatch):
    """Keep mutable nested parameters detached across both public read paths."""

    name, instances = fake_backend
    search = make_search(provider=name)
    backend = instances[-1]
    monkeypatch.setattr(backend, "_candidate_params", lambda index: {"a": {"b": index}})
    candidates = search.ask(num_samples=2)

    assert candidates[0] is not backend._pending[0]
    assert isinstance(candidates[0].params, dict)
    candidates[0].params["a"]["b"] = 99
    assert backend._pending[0].params == {"a": {"b": 0}}

    snapshot = search.pending_candidates
    assert snapshot[0] is not backend._pending[0]
    assert isinstance(snapshot[0].params, dict)
    snapshot[0].params["a"]["b"] = 101
    assert search.pending_candidates[0].params == {"a": {"b": 0}}

    with pytest.raises(ValueError, match="parameters do not match"):
        search.tell([score(candidates[0])])
    assert [candidate.params for candidate in search.pending_candidates] == [
        {"a": {"b": 0}},
        {"a": {"b": 1}},
    ]


def test_best_result_forwards_single_and_multi_objective_result_semantics(fake_backend):
    """Expose the existing best-row/Pareto behavior without defining another ranking policy."""

    name, _ = fake_backend
    single = make_search(provider=name)
    single_candidates = single.ask(num_samples=2)
    single.tell(
        [
            score(single_candidates[0], 5.0),
            score(single_candidates[1], 1.0),
        ]
    )

    assert isinstance(single.best_result, dict)
    assert single.best_result == single.result.get_best_result()

    multi = make_search(provider=name, pool_size=32, num_objectives=2)
    multi_candidates = multi.ask(num_samples=2)
    multi.tell(
        [
            Score(
                score=(1.0, 5.0),
                params=multi_candidates[0].params,
                param_id=multi_candidates[0].param_id,
                num_objectives=2,
            ),
            Score(
                score=(5.0, 1.0),
                params=multi_candidates[1].params,
                param_id=multi_candidates[1].param_id,
                num_objectives=2,
            ),
        ]
    )

    assert isinstance(multi.best_result, list)
    assert multi.best_result == multi.result.get_best_result()


def test_fresh_backend_setup_resets_facade_execution_identity(fake_backend):
    """Start a new execution without retaining candidate IDs from the replaced backend.

    Internal reduction intentionally installs a fresh backend on an existing
    Search. Providers may restart their local candidate sequence, so
    facade-reported IDs from the previous execution must be cleared at that
    boundary while duplicate-ID checks remain strict within each execution.
    """

    name, instances = fake_backend
    search = make_search(provider=name)
    first = search.ask(num_samples=2)
    search.tell([score(candidate) for candidate in first])
    search.ask(num_samples=1)

    search._do_setup()

    assert len(instances) == 2
    assert search.current_generation == 0
    assert not search.pending_candidates
    assert not search._reported_backend_candidate_ids
    assert not search._has_activity
    assert search.result.get_results().empty
    assert [candidate.param_id for candidate in search.ask(num_samples=2)] == [
        "candidate-0-0",
        "candidate-0-1",
    ]


def test_ask_evaluate_tell_matches_managed_execution(fake_backend):
    """Give manual orchestration the same worker and result semantics as start()."""

    name, instances = fake_backend
    manual = make_search(provider=name)
    candidates = manual.ask()
    manual.tell(manual.evaluate(candidates))

    managed = make_search(provider=name)
    managed_result = managed.start()

    assert manual.result.get_results().to_dict(orient="records") == (
        managed_result.get_results().to_dict(orient="records")
    )
    assert instances[0].current_generation == instances[1].current_generation == 1
    assert manual.current_generation == 1
    assert managed.current_generation == 0


def test_manual_worker_allows_external_ask_tell(fake_backend):
    """Keep externally evaluated candidates on the normal ask/tell lifecycle."""

    name, instances = fake_backend
    objective = Mock(return_value=999.0)
    search = make_search(
        provider=name,
        objective=objective,
        worker_type=WorkerTypes.MANUAL,
    )

    candidates = search.ask(num_samples=2)
    search.tell([score(candidate, 100 + index) for index, candidate in enumerate(candidates)])

    objective.assert_not_called()
    assert search.pending_candidates == ()
    assert list(search.result.get_results()["score_1"]) == [100, 101]
    assert instances[-1].current_generation == 1


def test_manual_worker_rejects_evaluate_without_mutating_search(fake_backend, monkeypatch):
    """Reject accidental local execution while preserving externally issued work."""

    name, instances = fake_backend
    objective = Mock(return_value=999.0)
    search = make_search(
        provider=name,
        objective=objective,
        worker_type=WorkerTypes.MANUAL,
    )
    candidates = search.ask(num_samples=2)
    pending_before = search.pending_candidates
    run = Mock(side_effect=AssertionError("manual execution guard was bypassed"))
    monkeypatch.setattr(search._worker, "run", run)

    with pytest.raises(
        RuntimeError,
        match=r"evaluate\(\).*ManualWorker.*ask\(\).*tell\(\)",
    ):
        search.evaluate(candidates)

    run.assert_not_called()
    objective.assert_not_called()
    assert search.pending_candidates == pending_before
    assert search.result.get_results().empty
    assert instances[-1].current_generation == 0


def test_manual_worker_rejects_start_before_asking_or_marking_activity(fake_backend):
    """Leave a manual Search pristine when managed execution is requested by mistake."""

    name, instances = fake_backend
    objective = Mock(return_value=999.0)
    search = make_search(
        provider=name,
        objective=objective,
        worker_type=WorkerTypes.MANUAL,
    )

    with pytest.raises(
        RuntimeError,
        match=r"start\(\).*ManualWorker.*ask\(\).*tell\(\)",
    ):
        search.start()

    objective.assert_not_called()
    assert search._has_activity is False
    assert search.pending_candidates == ()
    assert search.result.get_results().empty
    assert instances[-1].ask_calls == 0


def test_evaluate_alone_leaves_search_state_unchanged(fake_backend):
    """Keep evaluation separate from optimizer acceptance and result recording."""

    name, instances = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)
    pending_before = search.pending_candidates
    generation_before = search.current_generation

    scores = search.evaluate(candidates)

    assert [str(item.param_id) for item in scores] == [
        str(candidate.param_id) for candidate in candidates
    ]
    assert search.pending_candidates == pending_before
    assert search.current_generation == generation_before
    assert instances[-1].current_generation == 0
    assert search.result.get_results().empty


def test_evaluate_accepts_out_of_order_pending_subset(fake_backend):
    """Evaluate only selected pending work without changing the outstanding ledger."""

    name, _ = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=3)

    scores = search.evaluate([candidates[2], candidates[0]])

    assert [str(item.param_id) for item in scores] == [
        str(candidates[2].param_id),
        str(candidates[0].param_id),
    ]
    assert search.pending_candidates == tuple(candidates)


def test_evaluate_rejects_invalid_candidates_before_worker(fake_backend, monkeypatch):
    """Reject duplicate, foreign, stale, and altered candidates before objective execution."""

    name, _ = fake_backend

    def make_observed_search():
        search = make_search(provider=name)
        candidates = search.ask(num_samples=2)
        run = Mock(return_value=[])
        monkeypatch.setattr(search._worker, "run", run)
        return search, candidates, run

    search, candidates, run = make_observed_search()
    with pytest.raises(ValueError, match="duplicates=1"):
        search.evaluate([candidates[0], candidates[0]])
    run.assert_not_called()

    search, candidates, run = make_observed_search()
    foreign = candidates[0].model_copy(update={"param_id": "foreign"})
    with pytest.raises(ValueError, match="unknown=.*foreign"):
        search.evaluate([foreign])
    run.assert_not_called()

    search, candidates, run = make_observed_search()
    search.tell([score(candidates[0])])
    with pytest.raises(ValueError, match="already_reported=.*candidate-0-0"):
        search.evaluate([candidates[0]])
    run.assert_not_called()

    search, candidates, run = make_observed_search()
    altered = candidates[0].model_copy(update={"params": candidates[1].params})
    with pytest.raises(ValueError, match="parameters do not match"):
        search.evaluate([altered])
    run.assert_not_called()


def test_evaluate_forwards_standard_and_worker_specific_arguments(fake_backend):
    """Use the same concurrency, timeout, and extension kwargs as managed execution."""

    name, _ = fake_backend
    ArgumentRecordingWorker.calls = []
    search = make_search(provider=name, worker_type=ArgumentRecordingWorker)
    candidates = search.ask(num_samples=2)

    search.evaluate(
        candidates,
        num_workers=3,
        task_timeout=4.5,
        scheduling_class="latency",
    )

    call = ArgumentRecordingWorker.calls[-1]
    assert call["num_workers"] == 3
    assert call["task_timeout"] == 4.5
    assert call["scheduling_class"] == "latency"
    assert call["params_pool"] == [candidate.params for candidate in candidates]
    assert call["params_ids"] == [candidate.param_id for candidate in candidates]


def test_evaluate_failure_policy_accounts_for_an_accepted_subset(fake_backend):
    """Do not treat a failed remainder as an entirely failed first generation."""

    name, _ = fake_backend
    search = make_search(
        provider=name,
        objective=lambda _config: INVALID_SCORE,
        exit_on_failure=True,
    )
    candidates = search.ask(num_samples=2)
    search.tell([score(candidates[0], 1.0)])

    remaining_scores = search.evaluate([candidates[1]])

    assert remaining_scores[0].failed
    assert search.pending_candidates == (candidates[1],)
    search.tell(remaining_scores)
    assert search.pending_candidates == ()
    assert len(search.result.get_results()) == 2


def test_split_evaluate_applies_first_generation_failure_policy_at_tell(fake_backend):
    """Reject a fully failed generation even when worker evaluation was split.

    ``evaluate()`` is deliberately side-effect free, so neither partial call can
    decide whether the whole outstanding batch failed. ``tell()`` sees the
    complete submitted ID set and applies the fail-fast policy before mutating
    backend, pending, or result state.
    """

    name, instances = fake_backend
    search = make_search(
        provider=name,
        objective=lambda _config: INVALID_SCORE,
        exit_on_failure=True,
    )
    candidates = search.ask(num_samples=2)
    scores = [
        *search.evaluate([candidates[0]]),
        *search.evaluate([candidates[1]]),
    ]

    with pytest.raises(RuntimeError, match="All objective functions failed"):
        search.tell(scores)

    assert search.pending_candidates == tuple(candidates)
    assert search.result.get_results().empty
    assert instances[-1].current_generation == 0


def test_tell_rejects_parameter_mismatch_before_backend_or_result_mutation(fake_backend):
    """Reject swapped or forged parameters before either execution state can change."""

    name, instances = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)
    altered = score(candidates[0]).model_copy(update={"params": candidates[1].params})

    with pytest.raises(ValueError, match="parameters do not match"):
        search.tell([altered, score(candidates[1])])

    assert search.pending_candidates == tuple(candidates)
    assert search.result.get_results().empty
    assert instances[-1].current_generation == 0


def test_tell_rejects_objective_count_mismatch_before_backend_mutation(fake_backend):
    """Keep malformed objective shapes from partially advancing optimizer state."""

    name, instances = fake_backend
    search = make_search(provider=name)
    candidate = search.ask(num_samples=1)[0]
    altered = score(candidate).model_copy(update={"num_objectives": 2, "score": (1.0, 2.0)})

    with pytest.raises(ValueError, match="objective counts"):
        search.tell([altered])

    assert search.pending_candidates == (candidate,)
    assert search.result.get_results().empty
    assert instances[-1].current_generation == 0


def test_tell_rejects_missing_normalized_score_before_backend_mutation(fake_backend):
    """Require normalized values for successful scores before accepting the batch."""

    name, instances = fake_backend
    search = Search(
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        search_config=SearchConfiguration(
            generations=1,
            pool_size=6,
            normalize=True,
        ),
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
        engine=EngineConfig(name=name),
    )
    candidates = search.ask(num_samples=2)
    valid = score(candidates[0]).model_copy(update={"norm_score": 0.5})
    missing = score(candidates[1])

    with pytest.raises(ValueError, match="normalized scores are missing"):
        search.tell([valid, missing])

    assert search.pending_candidates == tuple(candidates)
    assert search.result.get_results().empty
    assert instances[-1].current_generation == 0


def test_tell_accepts_partial_out_of_order_subsets(fake_backend):
    """Accept each outstanding ID independently of tell order.

    This is a facade guarantee, not a backend capability: adapters may forward
    each subset immediately or buffer it for an underlying full-batch protocol.
    """

    name, instances = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=3)

    search.tell([score(candidates[2], 12), score(candidates[0], 10)])

    assert search.pending_candidates == (candidates[1],)
    assert instances[-1].current_generation == 0
    search.tell([score(candidates[1], 11)])
    assert search.pending_candidates == ()
    assert instances[-1].current_generation == 1
    assert list(search.result.get_results()["score_1"]) == [12, 10, 11]


def test_second_ask_requires_resolving_pending_batch(fake_backend):
    """Reject a second ask while IDs are pending unless the provider opts in.

    Multiple outstanding asks are independent of partial tells: every backend
    accepts score subsets, while this backend still permits only one
    in-flight ask batch.
    """

    name, _ = fake_backend
    search = make_search(provider=name)
    search.ask(num_samples=2)

    with pytest.raises(BackendCapabilityError, match="still pending"):
        search.ask(num_samples=2)


def test_tell_rejects_duplicate_candidate_scores(fake_backend):
    """Prevent one candidate from satisfying multiple positions in a pending batch."""

    name, _ = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)

    with pytest.raises(ValueError, match="duplicates=1"):
        search.tell([score(candidates[0]), score(candidates[0])])


def test_tell_rejects_unknown_and_already_reported_ids(fake_backend):
    """Distinguish never-issued IDs from IDs accepted by an earlier subset."""

    name, _ = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)
    search.tell([score(candidates[0])])

    with pytest.raises(ValueError, match="already_reported=.*candidate-0-0"):
        search.tell([score(candidates[0])])

    unknown = score(candidates[1]).model_copy(update={"param_id": "never-issued"})
    with pytest.raises(ValueError, match="unknown=.*never-issued"):
        search.tell([unknown])


def test_tell_without_params_uses_issued_candidate_for_results(fake_backend):
    """Join an ID-only evaluation score with canonical issued parameters.

    ``Score.params`` remains accepted as a compatibility echo, but scores no
    longer need to repeat candidate payloads and result rows never trust that
    optional echo as their source of truth.
    """

    name, _ = fake_backend
    search = make_search(provider=name)
    candidate = search.ask(num_samples=1)[0]

    search.tell([Score(score=17, param_id=candidate.param_id, num_objectives=1)])

    row = search.result.get_results().iloc[0]
    assert row["params"] == candidate.params
    assert row["score_1"] == 17


def test_manual_tell_persists_dump_results_only_after_acceptance(fake_backend, tmp_path):
    """Persist accepted manual scores while leaving the file unchanged on rejection."""

    name, _ = fake_backend
    destination = tmp_path / "results.csv"
    search = make_search(provider=name, dump_results=destination)
    candidates = search.ask(num_samples=2)

    search.tell([score(candidates[0], 17)])

    persisted = destination.read_bytes()
    assert "17" in persisted.decode()
    with pytest.raises(ValueError, match="already_reported"):
        search.tell([score(candidates[0], 99)])
    assert destination.read_bytes() == persisted


def test_rejected_tell_does_not_create_dump_results(fake_backend, tmp_path):
    """Avoid writing a result artifact for scores rejected before backend acceptance."""

    name, _ = fake_backend
    destination = tmp_path / "results.csv"
    search = make_search(provider=name, dump_results=destination)
    candidates = search.ask(num_samples=2)
    altered = score(candidates[0]).model_copy(update={"params": candidates[1].params})

    with pytest.raises(ValueError, match="parameters do not match"):
        search.tell([altered])

    assert not destination.exists()


def test_failed_backend_ask_leaves_search_pristine_for_restore(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Allow direct restore after a backend rejects an ask before changing its state."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "pristine.checkpoint")
    target = make_search(provider=name)
    assert target._backend_impl is not None
    monkeypatch.setattr(
        target._backend_impl,
        "ask",
        Mock(side_effect=ValueError("rejected before mutation")),
    )

    with pytest.raises(ValueError, match="rejected before mutation"):
        target.ask(num_samples=2)

    assert target._has_activity is False
    assert target.pending_candidates == ()
    target.restore_checkpoint(checkpoint)
    assert target.current_generation == 0
    assert target.pending_candidates == ()


def test_checkpoint_envelope_keeps_provider_state_out_of_metadata_and_restores(
    fake_backend, tmp_path
):
    """Keep provider state in its blob while retaining enough safe metadata to restore it."""

    name, instances = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)
    search.tell([score(candidate) for candidate in candidates])
    checkpoint = search.save_checkpoint(tmp_path / "checkpoint.json")

    envelope = json.loads(checkpoint.read_text())
    assert envelope["format"] == "compileiq-checkpoint-v1"
    assert envelope["engine"] == name
    assert envelope["provider"] == name
    assert "algorithm" not in envelope
    assert envelope["provider_version"] == "1.2.3"
    assert envelope["search_config"]["generations"] == 1
    assert envelope["exit_on_failure"] is True
    assert len(envelope["result_ledger"]["rows"]) == 2
    assert "state" not in envelope
    assert "search_space" not in envelope

    restored = make_search(provider=name)
    restored.restore_checkpoint(checkpoint)
    assert instances[-1].current_generation == 1
    assert instances[-1].state == b"protected-state-marker"


def test_from_checkpoint_restores_saved_exit_on_failure_when_omitted(fake_backend, tmp_path):
    """Use the saved execution policy as the continuation default."""

    name, _ = fake_backend
    source = make_search(provider=name, exit_on_failure=False)
    checkpoint = source.save_checkpoint(tmp_path / "saved-policy.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
    )

    assert restored.exit_on_failure is False


def test_from_checkpoint_reconstructs_saved_builtin_worker(fake_backend, tmp_path):
    """Restore a non-default built-in worker without silently selecting the default."""

    name, _ = fake_backend
    source = make_search(provider=name, worker_type=WorkerTypes.ISOLATED)
    checkpoint = source.save_checkpoint(tmp_path / "isolated-worker.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        disable_progress_bar=True,
    )

    assert restored.worker_type is WorkerTypes.ISOLATED
    assert isinstance(restored._worker, IsoMultiProcessWorker)


def test_from_checkpoint_reconstructs_saved_manual_worker(fake_backend, tmp_path):
    """Preserve the execution guard when a manual workflow resumes from a checkpoint."""

    name, _ = fake_backend
    source = make_search(provider=name, worker_type=WorkerTypes.MANUAL)
    checkpoint = source.save_checkpoint(tmp_path / "manual-worker.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        disable_progress_bar=True,
    )

    assert restored.worker_type is WorkerTypes.MANUAL
    assert isinstance(restored._worker, ManualWorker)
    with pytest.raises(RuntimeError, match=r"start\(\).*ManualWorker"):
        restored.start()


def test_from_checkpoint_requires_unresolvable_custom_worker(fake_backend, tmp_path):
    """Require an explicit override when the checkpoint cannot name a built-in worker."""

    name, _ = fake_backend
    source = make_search(provider=name, worker_type=SynchronousWorker)
    checkpoint = source.save_checkpoint(tmp_path / "custom-worker.json")

    with pytest.raises(ValueError, match=r"cannot be reconstructed.*worker_type="):
        Search.from_checkpoint(
            checkpoint,
            objective_function=lambda config: config["x"],
            search_space={"x": ss.range(0, 10)},
            disable_progress_bar=True,
        )


def test_from_checkpoint_explicit_worker_overrides_saved_builtin(fake_backend, tmp_path):
    """Let the caller deliberately replace a saved built-in execution environment."""

    name, _ = fake_backend
    source = make_search(provider=name, worker_type=WorkerTypes.ISOLATED)
    checkpoint = source.save_checkpoint(tmp_path / "worker-override.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
    )

    assert restored.worker_type is SynchronousWorker
    assert isinstance(restored._worker, SynchronousWorker)


def test_from_checkpoint_reconstructs_tracker_type_without_cleartext_config(
    fake_backend,
    tmp_path,
):
    """Restore only a tracker identity while keeping its runtime settings out of the envelope."""

    name, _ = fake_backend
    source = make_search(
        provider=name,
        worker_type=WorkerTypes.ISOLATED,
        tracker_config=LoguruTrackerConfig(level="ERROR"),
    )
    checkpoint = source.save_checkpoint(tmp_path / "tracker-type.json")
    envelope = json.loads(checkpoint.read_text())

    assert envelope["worker_type_name"] == WorkerTypes.ISOLATED.value
    assert envelope["tracker_type_name"] == TrackerTypes.LOGURU.value
    assert "tracker_config" not in envelope
    assert "level" not in envelope

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        disable_progress_bar=True,
    )

    assert restored.tracker_config.type == TrackerTypes.LOGURU
    assert restored.tracker_config.level == "INFO"


@pytest.mark.parametrize(
    ("saved_policy", "override"),
    [(True, False), (False, True)],
)
def test_from_checkpoint_explicitly_overrides_exit_on_failure(
    fake_backend,
    tmp_path,
    saved_policy,
    override,
):
    """Distinguish an explicit False value from an omitted restore argument."""

    name, _ = fake_backend
    source = make_search(provider=name, exit_on_failure=saved_policy)
    checkpoint = source.save_checkpoint(tmp_path / f"policy-{saved_policy}.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
        exit_on_failure=override,
    )

    assert restored.exit_on_failure is override


def test_direct_restore_retains_target_exit_on_failure(fake_backend, tmp_path):
    """Keep direct restore bound to the fresh target Search's configured policy."""

    name, _ = fake_backend
    source = make_search(provider=name, exit_on_failure=True)
    checkpoint = source.save_checkpoint(tmp_path / "direct-policy.json")
    target = make_search(provider=name, exit_on_failure=False)

    target.restore_checkpoint(checkpoint)

    assert target.exit_on_failure is False


@pytest.mark.parametrize(("exit_on_failure", "raises"), [(True, True), (False, False)])
def test_restored_generation_zero_failures_follow_execution_policy(
    fake_backend,
    tmp_path,
    exit_on_failure,
    raises,
):
    """Apply the restored execution policy without changing sealed optimizer state."""

    name, instances = fake_backend
    source = make_search(provider=name, exit_on_failure=exit_on_failure)
    checkpoint = source.save_checkpoint(tmp_path / f"fail-policy-{exit_on_failure}.json")
    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda _config: INVALID_SCORE,
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="All objective functions failed"):
            restored.start()
        assert restored.result.get_results().empty
        assert instances[-1].current_generation == 0
    else:
        restored.start()
        assert len(restored.result.get_results()) == restored._search_config.pool_size
        assert instances[-1].current_generation == 1


def test_checkpoint_without_result_ledger_is_rejected(fake_backend, tmp_path):
    """Require the visible-history ledger so one artifact restores one logical Search."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "incomplete-envelope.json")
    envelope = json.loads(checkpoint.read_text())
    del envelope["result_ledger"]
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointValidationError, match="malformed"):
        make_search(provider=name).restore_checkpoint(checkpoint)


def test_protected_normalized_checkpoint_preserves_baseline_sentinel(fake_backend, tmp_path):
    """Permit the public baseline marker while keeping protected candidate payloads opaque."""

    name, _ = fake_backend
    search_space = tmp_path / "protected.bin"
    search_space.write_bytes(b"test protected search space")
    source = make_search(
        provider=name,
        search_space=search_space,
        normalize=True,
        objective=lambda _: 5.0,
        worker_type=BaselineInjectingWorker,
    )

    source.start()
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    envelope = json.loads(checkpoint.read_text())

    assert envelope["result_ledger"]["rows"][0]["params"] == BASELINE_CONFIG
    assert all(isinstance(row["params"], str) for row in envelope["result_ledger"]["rows"][1:])

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda _: 5.0,
        search_space=search_space,
        worker_type=BaselineInjectingWorker,
        disable_progress_bar=True,
    )

    assert restored._result.get_results()["params"].tolist() == [
        BASELINE_CONFIG,
        *[f"opaque-{index}" for index in range(6)],
    ]


def test_from_checkpoint_reconstructs_safe_outer_configuration(fake_backend, tmp_path):
    """Rebuild API-owned settings and pending IDs around opaque provider state.

    Pending identities must remain visible to the public facade so restored work
    can be resubmitted and later scores can be matched without exposing or
    reconstructing the provider's sealed optimizer state.
    """

    name, _ = fake_backend
    source = make_search(provider=name)
    pending = source.ask(num_samples=2)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
    )

    assert restored.provider == name
    assert [candidate.param_id for candidate in restored.pending_candidates] == [
        candidate.param_id for candidate in pending
    ]
    assert [candidate.params for candidate in restored.pending_candidates] == [
        candidate.params for candidate in pending
    ]
    assert restored.pending_candidates == tuple(pending)


def test_from_checkpoint_rejects_provider_state_overrides(fake_backend, tmp_path):
    """Make saved execution-affecting configuration authoritative during continuation."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")

    with pytest.raises(TypeError, match="provider state"):
        Search.from_checkpoint(
            checkpoint,
            objective_function=lambda config: config["x"],
            search_space={"x": ss.range(0, 10)},
            worker_type=SynchronousWorker,
            engine_options={"seed": 17},
        )

    with pytest.raises(TypeError, match="search configuration"):
        Search.from_checkpoint(
            checkpoint,
            objective_function=lambda config: config["x"],
            search_space={"x": ss.range(0, 10)},
            worker_type=SynchronousWorker,
            search_config=SearchConfiguration(generations=2, pool_size=6),
        )


def test_checkpoint_restores_and_appends_user_visible_results(fake_backend, tmp_path):
    """Return one result history spanning accepted work before and after restoration."""

    name, _ = fake_backend
    source = make_search(provider=name, generations=2)
    first = source.ask(num_samples=2)
    source.tell([score(candidate, idx) for idx, candidate in enumerate(first)])
    checkpoint = source.save_checkpoint(tmp_path / "history.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
    )

    assert restored._result.get_results().to_dict(orient="records") == (
        source._result.get_results().to_dict(orient="records")
    )
    resumed = restored.start().get_results()
    assert len(resumed) == 8
    assert list(resumed["score_1"][:2]) == [0, 1]
    assert resumed.iloc[0]["params"] == resumed.iloc[2]["params"]
    assert resumed.iloc[0]["score_1"] == resumed.iloc[2]["score_1"]


def test_require_no_pending_fails_without_artifact_or_state_change(fake_backend, tmp_path):
    """Fail closed before sealing or replacing a destination when work is outstanding."""

    name, instances = fake_backend
    search = make_search(provider=name)
    search.ask(num_samples=2)
    destination = tmp_path / "must-not-exist.json"
    before = search._result.get_results().copy(deep=True)
    checkpoint_calls = instances[-1].checkpoint_calls

    with pytest.raises(CheckpointValidationError, match="requires no pending candidates"):
        search.save_checkpoint(destination, require_no_pending=True)

    assert not destination.exists()
    assert instances[-1].checkpoint_calls == checkpoint_calls
    assert search._result.get_results().equals(before)

    existing = tmp_path / "existing.json"
    existing.write_text("keep me")
    with pytest.raises(CheckpointValidationError, match="requires no pending candidates"):
        search.save_checkpoint(existing, require_no_pending=True)
    assert existing.read_text() == "keep me"


def test_discard_policy_rejects_late_results_and_replaces_one_for_one(fake_backend, tmp_path):
    """Discard all saved pending work, reject its late reports, and preserve batch cardinality."""

    name, _ = fake_backend
    source = make_search(provider=name)
    pending = source.ask(num_samples=4)
    checkpoint = source.save_checkpoint(tmp_path / "discard.json")

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
        pending_policy=PendingCandidatePolicy.DISCARD,
    )

    assert not restored.pending_candidates
    assert restored._result.get_results().empty
    with pytest.raises(ValueError, match="unknown=.*candidate-0-0"):
        restored.tell([score(candidate) for candidate in pending])
    assert restored._result.get_results().empty
    assert len(restored.ask()) == 4


def test_direct_restore_rejects_search_with_existing_activity(fake_backend, tmp_path):
    """Prevent direct restore from implicitly merging unrelated or divergent Search histories."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    destination = make_search(provider=name)
    destination.ask(num_samples=2)

    with pytest.raises(CheckpointValidationError, match="fresh Search"):
        destination.restore_checkpoint(checkpoint)


def test_checkpoint_rejects_malformed_result_ledger(fake_backend, tmp_path):
    """Validate visible result shape before it can become the restored Search history."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    envelope = json.loads(checkpoint.read_text())
    envelope["result_ledger"]["rows"] = [
        {
            "metadata": "",
            "generation": 0,
            "scores": [1.0, 2.0],
            "params": {"x": 1},
            "normalized_scores": None,
        }
    ]
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointValidationError, match="malformed"):
        make_search(provider=name).restore_checkpoint(checkpoint)


def test_checkpoint_rejects_non_finite_visible_result(fake_backend, tmp_path):
    """Reject NaN because the portable checkpoint ledger permits only JSON numbers.

    Failed evaluations use CompileIQ's explicit ``"*"`` score sentinel. Accepting
    Python's nonstandard NaN JSON extension would make envelope validation and
    cross-language restore behavior dependent on the decoder.
    """

    name, _ = fake_backend
    source = make_search(provider=name)
    candidate = source.ask(num_samples=1)[0]
    source.tell([score(candidate)])
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    envelope = json.loads(checkpoint.read_text())
    envelope["result_ledger"]["rows"][0]["scores"] = [float("nan")]
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointValidationError, match="malformed"):
        make_search(provider=name).restore_checkpoint(checkpoint)


def test_checkpoint_rejects_search_space_mismatch(fake_backend, tmp_path):
    """Refuse optimizer state when the caller supplies a different search-space identity."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    destination = make_search(provider=name, search_space={"y": ss.range(0, 10)})

    with pytest.raises(CheckpointValidationError, match="search space identity"):
        destination.restore_checkpoint(checkpoint)


def test_checkpoint_rejects_search_config_mismatch(fake_backend, tmp_path):
    """Refuse direct restore when execution-affecting SearchConfiguration differs."""

    name, _ = fake_backend
    source = make_search(provider=name, generations=1)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    destination = make_search(provider=name, generations=2)

    with pytest.raises(CheckpointValidationError, match="search configuration"):
        destination.restore_checkpoint(checkpoint)


def test_checkpoint_provenance_mismatch_is_strict_by_default(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Keep exact artifact provenance fail-closed unless a developer explicitly opts out."""

    monkeypatch.delenv("CIQ_UNSAFE_ALLOW_CHECKPOINT_PROVENANCE_MISMATCH", raising=False)
    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "strict-provenance.json")
    rewrite_checkpoint(checkpoint, artifact_sha256="b" * 64)

    with pytest.raises(CheckpointValidationError, match="artifact sha256"):
        make_search(provider=name).restore_checkpoint(checkpoint)


def test_checkpoint_unsafe_override_warns_once_and_restores_provenance_drift(
    fake_backend,
    tmp_path,
):
    """Permit only requested build-provenance drift while retaining a visible audit warning."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "unsafe-provenance.json")
    rewrite_checkpoint(
        checkpoint,
        provider_version="debug-build",
        artifact_sha256="b" * 64,
    )
    restored = make_search(provider=name)

    with pytest.warns(CheckpointCompatibilityWarning) as warning_records:
        restored.restore_checkpoint(
            checkpoint,
            allow_unsafe_provenance_mismatch=True,
        )

    assert len(warning_records) == 1
    warning_text = str(warning_records[0].message)
    saved_artifact_sha256 = "b" * 64
    current_artifact_sha256 = "a" * 64
    assert "provider_version: checkpoint='debug-build' current='1.2.3'" in warning_text
    assert (
        f"artifact_sha256: checkpoint='{saved_artifact_sha256}' "
        f"current='{current_artifact_sha256}'"
    ) in warning_text
    assert len(restored.ask(num_samples=1)) == 1


def test_checkpoint_provenance_override_reads_environment_at_call_time(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Support debugging through an explicit process setting without import-time caching."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "environment-provenance.json")
    rewrite_checkpoint(checkpoint, artifact_sha256="b" * 64)
    monkeypatch.delenv("CIQ_UNSAFE_ALLOW_CHECKPOINT_PROVENANCE_MISMATCH", raising=False)

    with pytest.raises(CheckpointValidationError, match="artifact sha256"):
        make_search(provider=name).restore_checkpoint(checkpoint)

    monkeypatch.setenv("CIQ_UNSAFE_ALLOW_CHECKPOINT_PROVENANCE_MISMATCH", "1")

    with pytest.warns(CheckpointCompatibilityWarning, match="artifact_sha256"):
        make_search(provider=name).restore_checkpoint(checkpoint)


def test_from_checkpoint_forwards_unsafe_provenance_override(fake_backend, tmp_path):
    """Expose the same narrow development escape hatch through the canonical restore API."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "from-checkpoint-provenance.json")
    rewrite_checkpoint(checkpoint, provider_version="debug-build")

    with pytest.warns(CheckpointCompatibilityWarning, match="provider_version"):
        restored = Search.from_checkpoint(
            checkpoint,
            objective_function=lambda config: config["x"],
            search_space={"x": ss.range(0, 10)},
            worker_type=SynchronousWorker,
            disable_progress_bar=True,
            allow_unsafe_provenance_mismatch=True,
        )

    assert restored.provider == name
    assert len(restored.ask(num_samples=1)) == 1


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("abi_version", 2, "abi version"),
        ("engine", "other-engine", "engine"),
        ("provider", "other-provider", "provider"),
        ("search_space_identity", "b" * 64, "search space identity"),
    ],
)
def test_checkpoint_provenance_override_keeps_identity_and_abi_checks_hard(
    fake_backend,
    tmp_path,
    field,
    value,
    error,
):
    """Never treat ABI, optimizer identity, or search-space identity as build provenance."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / f"hard-{field}.json")
    rewrite_checkpoint(checkpoint, **{field: value})

    with pytest.raises(CheckpointValidationError, match=error):
        make_search(provider=name).restore_checkpoint(
            checkpoint,
            allow_unsafe_provenance_mismatch=True,
        )


def test_checkpoint_provenance_override_keeps_format_parse_gate_hard(fake_backend, tmp_path):
    """Reject an unknown envelope format before applying any provenance exception."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "hard-format.json")
    rewrite_checkpoint(checkpoint, format="compileiq-checkpoint-v2")

    with pytest.raises(CheckpointValidationError, match="malformed"):
        make_search(provider=name).restore_checkpoint(
            checkpoint,
            allow_unsafe_provenance_mismatch=True,
        )


def test_checkpoint_provenance_override_keeps_search_configuration_hard(
    fake_backend,
    tmp_path,
):
    """Keep execution semantics outside the development-only provenance exception."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "hard-search-config.json")
    envelope = json.loads(checkpoint.read_text())
    envelope["search_config"]["generations"] = 2
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointValidationError, match="search configuration"):
        make_search(provider=name).restore_checkpoint(
            checkpoint,
            allow_unsafe_provenance_mismatch=True,
        )


def test_checkpoint_provenance_override_keeps_integrity_checksum_hard(
    fake_backend,
    tmp_path,
):
    """Never bypass the sealed-state corruption check through a provenance option."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "hard-integrity.json")
    rewrite_checkpoint(checkpoint, sealed_blob_sha256="b" * 64)

    with pytest.raises(CheckpointValidationError, match="checksum"):
        make_search(provider=name).restore_checkpoint(
            checkpoint,
            allow_unsafe_provenance_mismatch=True,
        )


def test_managed_checkpoint_preflight_rejects_oversized_result_projection(
    fake_backend,
    tmp_path,
):
    """Fail before objective work when the configured managed history exceeds the v1 cap."""

    name, _ = fake_backend
    objective = Mock(return_value=1.0)
    generations = (ciq_module._MAX_CHECKPOINT_RESULT_ROWS // 6) + 1
    search = make_search(
        provider=name,
        generations=generations,
        objective=objective,
    )
    destination = tmp_path / "oversized-managed.json"

    with pytest.raises(CheckpointValidationError, match="result-history projection"):
        search.start(checkpoint_path=destination)

    objective.assert_not_called()
    assert search._has_activity is False
    assert not destination.exists()


def test_managed_checkpoint_preflight_rejects_oversized_population(
    fake_backend,
    tmp_path,
):
    """Report the checkpoint-specific population cap before evaluating candidates."""

    name, _ = fake_backend
    objective = Mock(return_value=1.0)
    search = make_search(
        provider=name,
        pool_size=ciq_module._MAX_CHECKPOINT_POPULATION + 1,
        objective=objective,
    )

    with pytest.raises(CheckpointValidationError, match="checkpointable population limit"):
        search.start(checkpoint_path=tmp_path / "oversized-population.json")

    objective.assert_not_called()
    assert search._has_activity is False


def test_managed_checkpoint_preflight_rejects_oversized_candidate_history(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Bound the configured minimum candidate-history projection before work."""

    name, _ = fake_backend
    objective = Mock(return_value=1.0)
    search = make_search(provider=name, generations=1, objective=objective)
    monkeypatch.setattr(ciq_module, "_MAX_CHECKPOINT_CANDIDATE_HISTORY", 5)

    with pytest.raises(CheckpointValidationError, match="candidate-history projection"):
        search.start(checkpoint_path=tmp_path / "oversized-candidate-history.json")

    objective.assert_not_called()
    assert search._has_activity is False


def test_managed_checkpoint_preflight_accepts_ordinary_multi_generation_search(fake_backend):
    """Do not multiply ordinary capacity by a provider's dynamic failure-refill limit."""

    name, _ = fake_backend
    search = make_search(provider=name, pool_size=64, generations=100)

    search._preflight_checkpoint_capacity()


def test_manual_checkpoint_reports_result_history_limit_before_sealing(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Name a true row-cap violation without masking unrelated ledger validation errors."""

    name, instances = fake_backend
    search = make_search(provider=name)
    candidates = search.ask(num_samples=2)
    search.tell([score(candidate) for candidate in candidates])
    monkeypatch.setattr(ciq_module, "_MAX_CHECKPOINT_RESULT_ROWS", 1)
    destination = tmp_path / "oversized-manual.json"

    with pytest.raises(CheckpointValidationError, match=r"limit of 1 rows.*2 result"):
        search.save_checkpoint(destination)

    assert instances[-1].checkpoint_calls == 0
    assert not destination.exists()


def test_start_resumes_restored_pending_batch_without_second_ask(fake_backend, tmp_path):
    """Resubmit identical restored candidates before asking the optimizer for replacements."""

    name, instances = fake_backend
    source = make_search(provider=name)
    pending = source.ask()
    checkpoint = source.save_checkpoint(tmp_path / "pending.json")

    restored = make_search(provider=name)
    restored.restore_checkpoint(checkpoint)
    result = restored.start()

    assert len(result.get_results()) == len(pending)
    assert instances[-1].ask_calls == 0
    assert instances[-1].checkpoint_calls == 0
    assert instances[-1].current_generation == 1


def test_start_saves_checkpoint_after_each_accepted_managed_batch(fake_backend, tmp_path):
    """Checkpoint each accepted batch, whether or not it advances the generation."""

    name, instances = fake_backend
    destination = tmp_path / "managed.checkpoint"
    search = make_search(provider=name, generations=2)

    search.start(checkpoint_path=destination)

    assert destination.exists()
    assert instances[0].checkpoint_calls == 2
    assert not list(tmp_path.glob(".managed.checkpoint.*.tmp"))
    restored = make_search(provider=name, generations=2)
    restored.restore_checkpoint(destination)
    assert instances[-1].current_generation == 2
    assert not restored.pending_candidates
    assert len(restored.result.get_results()) == 12


def test_start_rejects_automatic_checkpoint_for_unsupported_engine(
    uncheckpointed_backend,
    tmp_path,
):
    """Preflight managed checkpoint capability before starting workers or Search activity."""

    destination = tmp_path / "must-not-exist.checkpoint"
    search = make_search(provider=uncheckpointed_backend)
    assert search._backend_registration is not None
    assert not search._backend_registration.capabilities.checkpoint

    with pytest.raises(BackendCapabilityError, match="does not support checkpointing"):
        search.start(checkpoint_path=destination)

    assert not destination.exists()
    assert not search._has_activity


def test_start_propagates_automatic_checkpoint_failure(fake_backend, monkeypatch, tmp_path):
    """Stop managed execution when a requested durability write fails."""

    name, instances = fake_backend
    destination = tmp_path / "unwritable.checkpoint"
    attempts = []

    def fail_checkpoint(self, path, *, require_no_pending=False):
        """Record the managed-save contract before simulating a storage failure."""

        attempts.append((path, require_no_pending))
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(Search, "save_checkpoint", fail_checkpoint)
    search = make_search(provider=name, generations=2)

    with pytest.raises(OSError, match="checkpoint write failed"):
        search.start(checkpoint_path=destination)

    assert attempts == [(destination, True)]
    assert instances[-1].current_generation == 1
    assert not destination.exists()


def test_start_checkpoints_refill_before_enforcing_next_batch_limit(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Stop a managed search when refill tells never complete the generation.

    A provider may legitimately request bounded replacement batches after failed
    evaluations. This guard prevents an implementation that consumes every
    tell without advancing or completing from trapping ``start()`` in an
    unlimited ask, evaluate, tell loop.
    """

    name, instances = fake_backend
    destination = tmp_path / "refill.checkpoint"

    def tell_without_advancing(self, scores):
        """Simulate a provider requesting refill work within the same generation."""

        assert len(scores) == len(self._pending)
        self._pending = []

    monkeypatch.setattr(FakeSealedBackend, "tell", tell_without_advancing)
    monkeypatch.setattr(
        FakeSealedBackend,
        "max_refill_batches_per_generation",
        property(lambda self: 1),
    )
    search = make_search(provider=name)

    with pytest.raises(RuntimeError, match="max_refill_batches_per_generation"):
        search.start(checkpoint_path=destination)

    assert instances[-1].ask_calls == 1
    assert instances[-1].checkpoint_calls == 1
    assert destination.exists()


def test_start_uses_generation_learned_during_ask(fake_backend, monkeypatch):
    """Use the generation attached to the issued batch, not a stale pre-ask value."""

    name, _ = fake_backend
    original_ask = FakeSealedBackend.ask

    def ask_drives_generation(self, num_samples=None):
        """Model providers that learn their authoritative generation while asking."""

        candidates = original_ask(self, num_samples)
        self._generation = 1
        return candidates

    monkeypatch.setattr(FakeSealedBackend, "ask", ask_drives_generation)
    progress = Mock()
    progress.n = 0
    progress.update.side_effect = lambda amount=1: setattr(
        progress,
        "n",
        progress.n + amount,
    )
    monkeypatch.setattr(ciq_module, "tqdm", lambda **_kwargs: progress)
    GenerationRecordingWorker.seen_generations = []
    search = Search(
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        search_config=SearchConfiguration(generations=2, pool_size=6),
        worker_type=GenerationRecordingWorker,
        disable_progress_bar=True,
        engine=EngineConfig(name=name),
    )

    search.start()

    assert GenerationRecordingWorker.seen_generations == [1]
    assert progress.n == 2


def test_start_updates_progress_with_best_single_objective_score(fake_backend, monkeypatch):
    """Preserve the useful best-score progress detail across backend orchestration."""

    name, _ = fake_backend

    class RecordingProgress:
        instances = []

        def __init__(self, **kwargs):
            self.initial = kwargs.get("initial", 0)
            self.n = self.initial
            self.postfixes = []
            self.instances.append(self)

        def set_description(self, _description):
            pass

        def set_postfix(self, values):
            self.postfixes.append(values)

        def update(self, amount=1):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setattr(ciq_module, "tqdm", RecordingProgress)

    make_search(provider=name).start()

    assert RecordingProgress.instances[0].initial == 0
    assert RecordingProgress.instances[0].postfixes == [{"🏆 best_score": "0.0000", "at_gen": 0}]


def test_restored_start_initializes_progress_from_saved_generation(
    fake_backend,
    monkeypatch,
    tmp_path,
):
    """Display resumed progress from saved optimizer state instead of zero."""

    name, _ = fake_backend
    source = make_search(provider=name, generations=2)
    candidates = source.ask()
    source.tell([score(candidate) for candidate in candidates])
    checkpoint = source.save_checkpoint(tmp_path / "generation-one.json")

    class RecordingProgress:
        instances = []

        def __init__(self, **kwargs):
            self.initial = kwargs.get("initial", 0)
            self.n = self.initial
            self.instances.append(self)

        def set_description(self, _description):
            pass

        def set_postfix(self, _values):
            pass

        def update(self, amount=1):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setattr(ciq_module, "tqdm", RecordingProgress)
    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda config: config["x"],
        search_space={"x": ss.range(0, 10)},
        worker_type=SynchronousWorker,
        disable_progress_bar=True,
    )

    restored.start()

    assert RecordingProgress.instances[0].initial == 1
    assert RecordingProgress.instances[0].n == 2


def test_restore_surfaces_provider_rejection_after_outer_checksum_passes(
    fake_backend, monkeypatch, tmp_path
):
    """Keep provider-owned validation hard even when build provenance is bypassed."""

    name, _ = fake_backend
    source = make_search(provider=name)
    checkpoint = source.save_checkpoint(tmp_path / "checkpoint.json")
    envelope = json.loads(checkpoint.read_text())
    rejected_blob = b"provider-rejected-checkpoint"
    envelope["sealed_blob"] = base64.b64encode(rejected_blob).decode()
    envelope["sealed_blob_sha256"] = hashlib.sha256(rejected_blob).hexdigest()
    envelope["provider_version"] = "debug-build"
    checkpoint.write_text(json.dumps(envelope))

    received = []

    def reject_restore(
        self,
        sealed_blob,
        *,
        pending_policy=PendingCandidatePolicy.RESUBMIT,
    ):
        """Record the delegated blob before simulating provider-owned validation failure."""

        del self
        received.append((sealed_blob, pending_policy))
        raise ValueError("provider rejected checkpoint")

    monkeypatch.setattr(FakeSealedBackend, "restore", reject_restore)

    with pytest.warns(CheckpointCompatibilityWarning, match="provider_version"):
        with pytest.raises(CheckpointValidationError, match="Backend rejected"):
            source.restore_checkpoint(
                checkpoint,
                allow_unsafe_provenance_mismatch=True,
            )

    assert received == [(rejected_blob, PendingCandidatePolicy.RESUBMIT)]


def test_direct_restore_without_checkpoint_support_rejects_before_parsing(
    uncheckpointed_backend,
    monkeypatch,
    tmp_path,
):
    """Capability-gate a direct restore before it reads or parses the artifact.

    An engine that cannot restore checkpoints must make ``restore_checkpoint``
    reject with the capability error at the very top: before it opens or parses
    the supplied file, and before it creates any Search activity. A malformed or
    hostile artifact is never read on a search that could never restore it.
    """

    search = make_search(provider=uncheckpointed_backend)
    assert search.provider == uncheckpointed_backend
    assert search._backend_registration is not None
    assert not search._backend_registration.capabilities.checkpoint

    artifact = tmp_path / "checkpoint.json"
    artifact.write_text("this is not a valid checkpoint envelope")

    def fail_if_parsed(*_args, **_kwargs):
        """Fail loudly if capability gating did not run before artifact parsing."""

        raise AssertionError("restore_checkpoint parsed the artifact before capability gating")

    monkeypatch.setattr(ciq_module, "_load_checkpoint_envelope", fail_if_parsed)

    with pytest.raises(BackendCapabilityError, match="does not support checkpointing"):
        search.restore_checkpoint(artifact)

    assert search._has_activity is False
    assert search.current_generation == 0
    assert search.pending_candidates == ()
    assert search.result.get_results().empty


@pytest.fixture
def strict_validator_backend():
    """Register a provider whose validator rejects retired settings, as amber's does."""

    name = "test-strict-validator-backend"
    validated = []

    def validator(search_config):
        """Record what the provider sees and fail like amber on non-default retired settings."""

        validated.append(search_config)
        defaults = SearchConfiguration(
            generations=search_config.generations,
            pool_size=search_config.pool_size,
            num_objectives=search_config.num_objectives,
        )
        explicit = sorted(
            name
            for name in ("cull_size", "mutate_rate", "objective_weights", "enable_large_fail_pool")
            if name in search_config.model_fields_set
            and getattr(search_config, name) != getattr(defaults, name)
        )
        if explicit:
            raise BackendCapabilityError(f"explicit retired settings: {explicit}")

    register_backend(
        name,
        FakeSealedBackend,
        capabilities=BackendCapabilities(arbitrary_batch_size=True),
        artifact=BackendArtifactInfo(
            backend_version="0.0.1",
            abi_version=0,
            artifact_sha256="d" * 64,
        ),
        search_config_validator=validator,
    )
    try:
        yield name, validated
    finally:
        unregister_backend(name)


@pytest.mark.parametrize(
    "settings",
    [
        {"mutate_rate": 0.5},
        {"cull_size": 2},
        {"objective_weights": [1.0]},
        {"enable_large_fail_pool": False},
        {"mutate_rate": 0.5, "cull_size": 2},
    ],
)
def test_retired_settings_are_ignored_with_a_warning(strict_validator_backend, settings):
    """Keep 1.0 configurations running: warn, then hide retired settings from the engine."""

    name, validated = strict_validator_backend
    # With pool_size=10 the derived cull_size is 6, so cull_size=2 is a real override.
    config = SearchConfiguration(generations=1, pool_size=10, **settings)

    retired_warning = "have no effect since CompileIQ 1.1 are ignored"
    with pytest.warns(FutureWarning, match=retired_warning) as record:
        search = Search(
            objective_function=lambda params: params["x"],
            search_space={"x": ss.range(0, 10)},
            search_config=config,
            worker_type=SynchronousWorker,
            engine=name,
            disable_progress_bar=True,
        )

    message = str(record[0].message)
    for setting in settings:
        assert setting in message
    assert not validated[-1].model_fields_set & set(settings)
    # The caller's configuration object is left as given.
    assert search.search_config is config
    assert all(getattr(config, key) == value for key, value in settings.items())


def test_retired_settings_never_reach_a_backend_without_a_validator():
    """The warning and the stripping don't depend on the engine declaring a validator."""

    name = "test-recording-backend-no-validator"
    received = []

    class RecordingBackend(FakeSealedBackend):
        def __init__(self, *, search_config, **kwargs):
            received.append(search_config)
            super().__init__(search_config=search_config, **kwargs)

    register_backend(
        name,
        RecordingBackend,
        capabilities=BackendCapabilities(arbitrary_batch_size=True),
        artifact=BackendArtifactInfo(
            backend_version="0.0.1",
            abi_version=0,
            artifact_sha256="e" * 64,
        ),
    )
    try:
        derived = SearchConfiguration(generations=1, pool_size=10)
        config = SearchConfiguration(generations=1, pool_size=10, mutate_rate=0.5, cull_size=2)
        with pytest.warns(FutureWarning, match="have no effect since CompileIQ 1.1 are ignored"):
            Search(
                objective_function=lambda params: params["x"],
                search_space={"x": ss.range(0, 10)},
                search_config=config,
                worker_type=SynchronousWorker,
                engine=name,
                disable_progress_bar=True,
            )
    finally:
        unregister_backend(name)

    assert received[-1].mutate_rate == derived.mutate_rate
    assert received[-1].cull_size == derived.cull_size


def test_explicit_default_retired_settings_stay_silent(strict_validator_backend, recwarn):
    """Spelling out a value CompileIQ would derive anyway is not a retired setting."""

    name, validated = strict_validator_backend
    derived = SearchConfiguration(generations=1, pool_size=8)
    config = SearchConfiguration(
        generations=1,
        pool_size=8,
        cull_size=derived.cull_size,
        mutate_rate=derived.mutate_rate,
        enable_large_fail_pool=True,
    )

    Search(
        objective_function=lambda params: params["x"],
        search_space={"x": ss.range(0, 10)},
        search_config=config,
        worker_type=SynchronousWorker,
        engine=name,
        disable_progress_bar=True,
    )

    assert not [warning for warning in recwarn if issubclass(warning.category, FutureWarning)]
    assert validated[-1] is config


class ExhaustingBackend(FakeSealedBackend):
    """Run out of untried candidates once a configured generation is reached."""

    exhaust_at_generation = 1
    error_factory = staticmethod(
        lambda: ValueError(
            "Unable to generate 6 unique candidates after 10 attempts. "
            "The search space may be exhausted."
        )
    )

    def ask(self, num_samples=None):
        """Raise the engine's exhaustion error instead of issuing a batch."""

        if self._generation >= self.exhaust_at_generation:
            raise self.error_factory()
        return super().ask(num_samples)


@pytest.fixture
def exhausting_backend():
    """Register ``ExhaustingBackend`` and restore its class settings afterwards."""

    name = "test-exhausting-backend"
    register_backend(
        name,
        ExhaustingBackend,
        capabilities=BackendCapabilities(arbitrary_batch_size=True),
        artifact=BackendArtifactInfo(
            backend_version="0.0.1",
            abi_version=0,
            artifact_sha256="e" * 64,
        ),
    )
    try:
        yield name
    finally:
        unregister_backend(name)
        ExhaustingBackend.exhaust_at_generation = 1
        ExhaustingBackend.error_factory = staticmethod(
            lambda: ValueError(
                "Unable to generate 6 unique candidates after 10 attempts. "
                "The search space may be exhausted."
            )
        )


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: ValueError("Unable to generate 6 unique candidates after 10 attempts."),
        lambda: backends_module.SearchSpaceExhaustedError("no untried candidates left"),
    ],
)
def test_start_ends_early_with_results_when_engine_runs_out(exhausting_backend, error_factory):
    """Return accepted results with a warning instead of failing a long search."""

    ExhaustingBackend.error_factory = staticmethod(error_factory)
    ExhaustingBackend.exhaust_at_generation = 2
    search = make_search(provider=exhausting_backend, generations=5)

    with pytest.warns(ciq_module.SearchSpaceExhaustedWarning, match="after 2 of 5 generations"):
        result = search.start()

    assert len(result.get_results()) == 12


def test_start_reports_exhaustion_before_any_result(exhausting_backend):
    """A space smaller than one batch is a configuration error, not an early end."""

    ExhaustingBackend.exhaust_at_generation = 0
    search = make_search(provider=exhausting_backend, generations=3)

    with pytest.raises(backends_module.SearchSpaceExhaustedError, match="fewer distinct"):
        search.start()


def test_unrelated_value_errors_still_fail_the_search(exhausting_backend):
    """Only exhaustion ends a search early; other engine errors propagate."""

    ExhaustingBackend.error_factory = staticmethod(lambda: ValueError("engine bug"))
    search = make_search(provider=exhausting_backend, generations=3)

    with pytest.raises(ValueError, match="engine bug"):
        search.start()


def test_run_id_is_a_deprecated_none(fake_backend):
    """Scripts that printed or saved run_id keep running; there is no run ID any more."""

    name, _ = fake_backend
    search = make_search(provider=name)

    with pytest.warns(FutureWarning, match="run_id is always None"):
        assert search.run_id is None

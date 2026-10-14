import pytest
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, TrackerTypes, WorkerTypes
from compileiq.tracker import _TRACKER_TYPES_TO_CONFIG
from tests.utils import (
    multi_light_obj_func,
    light_obj_func,
    nested_light_obj_func,
    validate_scores,
    async_always_invalid_score_func,
    TEST_WORKERS,
    ASYNC_FUNC_MAP,
)


# ---------------------------------------------------------------------------
# test_start — deterministic representative configurations
# ---------------------------------------------------------------------------
# Each tuple: (norm, pool_size, gens, mrate, problem_type, func, num_workers)
# Covers: both problem_types, norm on/off, all 3 func types, 1 and 2 workers.
# For thorough combinatorial exploration, see tests/fuzz/test_search_fuzz.py.

START_CONFIGS = [
    pytest.param(False, 8, 2, 0.25, "min", light_obj_func, 1, id="min-single-flat"),
    pytest.param(True, 16, 2, 0.25, "max", light_obj_func, 2, id="max-single-flat-norm"),
    pytest.param(False, 16, 2, 0.25, "min", multi_light_obj_func, 1, id="min-multi-flat"),
    pytest.param(True, 16, 2, 0.59, "max", multi_light_obj_func, 2, id="max-multi-flat-norm"),
    pytest.param(False, 6, 2, 0.25, "min", nested_light_obj_func, 1, id="min-single-nested"),
    pytest.param(True, 24, 1, 0.77, "max", nested_light_obj_func, 2, id="max-single-nested-norm"),
]


@pytest.mark.order(-4)
@pytest.mark.parametrize("worker_type", TEST_WORKERS)
@pytest.mark.parametrize(
    "norm, pool_size, gens, mrate, problem_type, func, num_workers",
    START_CONFIGS,
)
def test_start(
    worker_type,
    norm,
    pool_size,
    gens,
    mrate,
    problem_type,
    func,
    num_workers,
    scripted_engine,
    mock_search_space,
    mock_nested_search_space,
):
    """E2E test with representative param combinations."""

    expected_returns = 5 if func == multi_light_obj_func else 1
    main_config = SearchConfiguration(
        normalize=norm,
        pool_size=pool_size,
        generations=gens,
        mutate_rate=mrate,
        problem_type=problem_type,
        num_objectives=expected_returns,
    )

    search_space_config = (
        mock_nested_search_space if func == nested_light_obj_func else mock_search_space
    )
    rfunc = ASYNC_FUNC_MAP[func] if worker_type == WorkerTypes.ASYNC else func
    with Search(
        engine=scripted_engine,
        objective_function=rfunc,
        search_space=search_space_config,
        search_config=main_config,
        worker_type=worker_type,
    ) as tuner:
        results = tuner.start(num_workers)

    results.get_best_result()
    df = results.get_results()

    # Validating scores
    validate_scores(df, func, expected_returns, normalize=norm)


SAMPLE_CONFIGS = [
    pytest.param(4, light_obj_func, id="4-sample-flat"),
    pytest.param(8, multi_light_obj_func, id="8-samples-multi"),
    pytest.param(16, nested_light_obj_func, id="16-samples-nested"),
    pytest.param(32, light_obj_func, id="32-samples-flat"),
]


@pytest.mark.order(2)
@pytest.mark.parametrize("num_samples, func", SAMPLE_CONFIGS)
def test_sample(
    num_samples,
    func,
    scripted_engine,
    mock_search_space,
    mock_nested_search_space,
):
    """E2E sample test with representative param combinations."""

    main_config = SearchConfiguration(
        generations=1,
        problem_type="min",
        num_objectives=10,
    )

    search_space_config = (
        mock_nested_search_space if func == nested_light_obj_func else mock_search_space
    )
    with Search(
        engine=scripted_engine,
        objective_function=func,
        search_space=search_space_config,
        search_config=main_config,
        worker_type=WorkerTypes.ASYNC,
    ) as tuner:
        results = tuner.sample(num_samples)
        assert len(results) == num_samples


@pytest.mark.order(-1)
@pytest.mark.parametrize(
    "worker_type", [pytest.param(WorkerTypes.NATIVE, marks=pytest.mark.requires_ipc)]
)
def test_chaining_calls(
    worker_type,
    scripted_engine,
    mock_search_space,
    mock_nested_search_space,
    tmp_path,
):
    gens = 2
    pool_size = 10
    tracker_type = TrackerTypes.DISABLED
    func = nested_light_obj_func

    expected_returns = 1
    main_config = SearchConfiguration(
        normalize=True,
        pool_size=pool_size,
        generations=gens,
        problem_type="min",
        num_objectives=expected_returns,
    )

    search_space_config = mock_nested_search_space
    rfunc = ASYNC_FUNC_MAP[func] if worker_type == WorkerTypes.ASYNC else func
    with Search(
        engine=scripted_engine,
        objective_function=rfunc,
        search_space=search_space_config,
        search_config=main_config,
        worker_type=worker_type,
        tracker_config=_TRACKER_TYPES_TO_CONFIG[tracker_type](),
    ) as tuner:
        results = tuner.start()
        results.get_best_result()
        tuner.start()


@pytest.mark.order(1)
def test_early_exit_fail(
    scripted_engine,
    mock_search_space,
):
    """
    Testing that the search exits early when all objective functions fail in the first generation
    and `exit_on_failure` is set to True.
    """

    main_config = SearchConfiguration(
        pool_size=8,
        generations=2,
        num_objectives=1,
    )

    with Search(
        engine=scripted_engine,
        objective_function=async_always_invalid_score_func,
        search_space=mock_search_space,
        search_config=main_config,
        worker_type=WorkerTypes.ASYNC,
        exit_on_failure=True,
    ) as tuner:
        with pytest.raises(RuntimeError) as excinfo:
            tuner.start()

        assert "All objective functions failed in the first gen" in str(excinfo.value)

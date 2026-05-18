"""
Fuzz tests for the Search API using Hypothesis.

These tests explore wide parameter spaces to catch edge-case interactions
that deterministic integration tests might miss. They are EXCLUDED from
default pytest runs (via --ignore in pyproject.toml) because they are
deliberately slow and resource-intensive.

Run explicitly with:
    pytest tests/fuzz/ -vvv                          # default: 20 examples (quick)
    CIQ_FUZZ_EXAMPLES=100 pytest tests/fuzz/ -vvv   # full run (nightly CI)
"""

import os
import pytest
from hypothesis import given, strategies as st, HealthCheck, settings, Phase
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, TrackerTypes, WorkerTypes
from compileiq.tracker import _TRACKER_TYPES_TO_CONFIG
from tests.utils import (
    multi_light_obj_func,
    light_obj_func,
    nested_light_obj_func,
    validate_scores,
    ASYNC_FUNC_MAP,
)


MAX_EXAMPLES = int(os.environ.get("CIQ_FUZZ_EXAMPLES", "20"))

FUZZ_SETTINGS = dict(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=MAX_EXAMPLES,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
)


@pytest.mark.order(2)
@settings(**FUZZ_SETTINGS)
@given(
    norm=st.booleans(),
    pool_size=st.integers(16, 128),
    gens=st.integers(1, 5),
    mrate=st.sampled_from([0.01, 0.1, 0.59, 0.77, 0.99]),
    problem_type=st.sampled_from(["min", "max"]),
    func=st.sampled_from([light_obj_func, multi_light_obj_func, nested_light_obj_func]),
    num_workers=st.integers(1, 3),
    worker_type=st.sampled_from(list(WorkerTypes)),
    tracker_type=st.sampled_from(list(TrackerTypes)),
)
def test_start_fuzz(
    norm,
    pool_size,
    gens,
    mrate,
    problem_type,
    func,
    num_workers,
    worker_type,
    tracker_type,
    mock_core_start,
    mock_send_to_core,
    mock_receive_from_core,
    mock_socket_listen,
    mock_search_space,
    mock_nested_search_space,
):
    """Fuzz test_start across the full original parameter space."""
    pytest.current_gen = 0
    pytest.max_gen = gens
    pytest.pool_size = pool_size
    pytest.nested_test = func == nested_light_obj_func
    pytest.encoded_knobs = True

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
        objective_function=rfunc,
        search_space=search_space_config,
        search_config=main_config,
        worker_type=worker_type,
        tracker_config=_TRACKER_TYPES_TO_CONFIG[tracker_type](),
        debug=True,
    ) as tuner:
        results = tuner.start(num_workers)

    results.get_best_result()
    df = results.get_results()

    validate_scores(df, func, expected_returns, normalize=norm)


@pytest.mark.order(1)
@settings(**FUZZ_SETTINGS)
@given(
    num_samples=st.integers(1, 128),
    func=st.sampled_from([light_obj_func, multi_light_obj_func, nested_light_obj_func]),
)
def test_sample_fuzz(
    num_samples,
    func,
    mock_core_start,
    mock_send_to_core,
    mock_receive_from_core,
    mock_socket_listen,
    mock_search_space,
    mock_nested_search_space,
):
    """Fuzz test_sample across the full original num_samples range."""

    main_config = SearchConfiguration(
        generations=1,
        problem_type="min",
        num_objectives=10,
    )

    pytest.current_gen = 0
    pytest.max_gen = 2
    pytest.pool_size = num_samples
    pytest.nested_test = func == nested_light_obj_func
    pytest.encoded_knobs = True

    search_space_config = (
        mock_nested_search_space if func == nested_light_obj_func else mock_search_space
    )
    with Search(
        objective_function=func,
        search_space=search_space_config,
        search_config=main_config,
    ) as tuner:
        results = tuner.sample(num_samples)
        assert len(results) == num_samples



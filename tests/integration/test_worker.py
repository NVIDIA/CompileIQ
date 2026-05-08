from compileiq.types import BASELINE_DNA, INVALID_SCORE
from compileiq.worker import MultiProcessWorker, AsyncWorker
import multiprocessing
import pytest
import asyncio
from pydantic import ValidationError

try:
    from ray.exceptions import RayTaskError
except ImportError:
    RayTaskError = None

from tests.utils import (
    multi_light_obj_func,
    light_obj_func,
    heavy_obj_func,
    funky_obj_func,
    fail_obj_func,
    generate_params,
    TEST_WORKER_CLASSES,
    ASYNC_FUNC_MAP,
)


def validate_scores(func, compileiq_scores, used_params):
    """
    Validates if workers computed the scores correctly and returned it in the correct order.
    Mostly useful for Multi-worker setups.
    """
    for i, param in enumerate(used_params):
        if "duration" in param:
            param["duration"] = 0.01

        scores = func(param)
        if isinstance(scores, float) or isinstance(scores, int):
            scores = [scores]
        for j, ss in enumerate(compileiq_scores[i]):
            assert ss == scores[j]


def create_worker(worker_class, **kwargs):
    return worker_class(**kwargs)


async def async_objective(config):
    await asyncio.sleep(config["duration"])
    return config["x"] ** 2 + config["y"]


@pytest.mark.requires_ipc
def test_wrong_objective_num(tmp_path):
    check_msg = "Something went wrong when executing your objective function."

    worker = create_worker(MultiProcessWorker, cache_folder=str(tmp_path))

    with pytest.raises(RuntimeError) as excinfo:
        worker.run(
            function=multi_light_obj_func,
            params_pool=generate_params(8),
            params_ids=list(range(64)),
            num_workers=1,
            num_function_returns=10,
        )
        assert check_msg in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        worker.run(
            function=multi_light_obj_func,
            params_pool=generate_params(8),
            params_ids=list(range(64)),
            num_workers=2,
            num_function_returns=2,
        )
        assert check_msg in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        worker.run(
            function=light_obj_func,
            params_pool=generate_params(8),
            params_ids=list(range(64)),
            num_workers=10,
            num_function_returns=10,
        )
        assert check_msg in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        worker.run(
            function=light_obj_func,
            params_pool=generate_params(8),
            params_ids=list(range(64)),
            num_workers=1,
            num_function_returns=500,
        )
        assert check_msg in str(excinfo.value)


@pytest.mark.parametrize("worker_class", TEST_WORKER_CLASSES)
def test_problematic_scores(worker_class, tmp_path):
    worker = create_worker(worker_class, cache_folder=str(tmp_path))

    params = generate_params(8)

    if worker_class == AsyncWorker:
        funky_func = ASYNC_FUNC_MAP[funky_obj_func]
        fail_func = ASYNC_FUNC_MAP[fail_obj_func]
    else:
        funky_func = funky_obj_func
        fail_func = fail_obj_func

    scores = worker.run(
        function=funky_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=2,
        num_function_returns=1,
    )
    assert len(scores) == len(params)

    expected_errors = [ValidationError, RuntimeError, ExceptionGroup]
    if RayTaskError is not None:
        expected_errors.append(RayTaskError)
    with pytest.raises(tuple(expected_errors)):
        scores = worker.run(
            function=fail_func,
            params_pool=params,
            params_ids=list(range(len(params))),
            num_workers=2,
            num_function_returns=1,
        )


@pytest.mark.parametrize("worker_class", TEST_WORKER_CLASSES)
def test_single_worker(worker_class, tmp_path):
    worker = create_worker(worker_class, cache_folder=str(tmp_path))

    params = generate_params(4)
    if worker_class == AsyncWorker:
        light_func = ASYNC_FUNC_MAP[light_obj_func]
        multi_light_func = ASYNC_FUNC_MAP[multi_light_obj_func]
    else:
        light_func = light_obj_func
        multi_light_func = multi_light_obj_func

    scores = worker.run(
        function=light_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_function_returns=1,
    )
    assert len(scores) == len(params)
    scores = worker.run(
        function=multi_light_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_function_returns=5,
    )
    assert len(scores) == len(params)


@pytest.mark.parametrize("worker_class", TEST_WORKER_CLASSES)
def test_multi_worker(worker_class, tmp_path):
    worker = create_worker(worker_class, cache_folder=str(tmp_path))

    params = generate_params(8)

    if worker_class == AsyncWorker:
        light_func = ASYNC_FUNC_MAP[light_obj_func]
        multi_light_func = ASYNC_FUNC_MAP[multi_light_obj_func]
        heavy_func = ASYNC_FUNC_MAP[heavy_obj_func]
    else:
        light_func = light_obj_func
        multi_light_func = multi_light_obj_func
        heavy_func = heavy_obj_func

    scores = worker.run(
        function=light_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=2,
        num_function_returns=1,
    )
    assert len(scores) == len(params)
    param_ordered = [s.params for s in scores]
    scores = [[s.score] if s.num_objectives == 1 else s.score for s in scores]
    validate_scores(light_obj_func, scores, param_ordered)

    scores = worker.run(
        function=multi_light_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=4,
        num_function_returns=5,
    )
    assert len(scores) == len(params)
    param_ordered = [s.params for s in scores]
    scores = [[s.score] if s.num_objectives == 1 else s.score for s in scores]
    validate_scores(multi_light_obj_func, scores, param_ordered)

    # Decreasing number of params so it doesn't take too much time
    params = generate_params(5)
    scores = worker.run(
        function=heavy_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=8,
        num_function_returns=1,
    )
    assert len(scores) == len(params)
    param_ordered = [s.params for s in scores]
    scores = [[s.score] if s.num_objectives == 1 else s.score for s in scores]
    validate_scores(heavy_obj_func, scores, param_ordered)


@pytest.mark.parametrize("worker_class", TEST_WORKER_CLASSES)
def test_baseline(worker_class, tmp_path):
    worker = create_worker(worker_class, normalize=True, cache_folder=str(tmp_path))

    params = generate_params(8)

    light_func = ASYNC_FUNC_MAP[light_obj_func] if worker_class == AsyncWorker else light_obj_func

    scores = worker.run(
        function=light_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=2,
        num_function_returns=1,
    )
    baseline = None
    for score in scores:
        if score.is_baseline:
            baseline = [score.score] if score.num_objectives == 1 else score.score
            break
    assert baseline is not None
    validate_scores(light_obj_func, [baseline], [BASELINE_DNA])


@pytest.mark.parametrize("worker_class", TEST_WORKER_CLASSES)
def test_timeout(worker_class, tmp_path):
    worker = create_worker(worker_class, cache_folder=str(tmp_path))

    if not worker.supports_timeout:
        pytest.skip(f"{worker_class.__name__} does not support timeout")

    params = generate_params(8, duration=2)

    if worker_class == AsyncWorker:
        heavy_func = async_objective
    else:
        heavy_func = heavy_obj_func

    scores = worker.run(
        function=heavy_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=2,
        num_function_returns=1,
        task_timeout=0.1,
    )

    for score in scores:
        assert score.score == INVALID_SCORE


@pytest.mark.parametrize("worker_class", TEST_WORKER_CLASSES)
@pytest.mark.parametrize("mp_mode", multiprocessing.get_all_start_methods())
def test_worker_process_modes(worker_class, mp_mode, monkeypatch, tmp_path):
    monkeypatch.setenv("CIQ_PROCESS_MODE", mp_mode)
    worker = create_worker(worker_class, cache_folder=str(tmp_path))

    light_func = ASYNC_FUNC_MAP[light_obj_func] if worker_class == AsyncWorker else light_obj_func

    params = generate_params(6)
    scores = worker.run(
        function=light_func,
        params_pool=params,
        params_ids=list(range(len(params))),
        num_workers=2,
        num_function_returns=1,
    )

    assert len(scores) == len(params)
    param_ordered = [s.params for s in scores]
    raw_scores = [[s.score] if s.num_objectives == 1 else s.score for s in scores]
    validate_scores(light_obj_func, raw_scores, param_ordered)

from __future__ import annotations

import pickle
from types import SimpleNamespace

from xla_ciq.search.factory import get_search_engine
from xla_ciq.search.mock import MockSearchEngine
from xla_ciq.search.compileiq_runtime import (
    CompileIqRuntimeEngine,
    RuntimeObjective,
    _start_tuner,
)
from xla_ciq.xla.tensor_map import TensorMapSpec


def test_factory_mock():
    assert isinstance(get_search_engine(mock=True), MockSearchEngine)


def test_factory_real():
    assert isinstance(get_search_engine(mock=False), CompileIqRuntimeEngine)


def test_runtime_starts_tuner_directly_with_per_objective_timeout():
    calls = []
    sentinel = object()

    class Tuner:
        def start(self, **kwargs):
            calls.append(kwargs)
            return sentinel

    request = SimpleNamespace(num_workers=3, task_timeout=12.5)

    assert _start_tuner(Tuner(), request) is sentinel
    assert calls == [{"num_workers": 3, "task_timeout": 12.5}]


def test_runtime_objective_is_picklable():
    obj = RuntimeObjective(
        ptx_path="/tmp/x.ptx",
        kernel_name="k",
        arch="sm_90a",
        shape="f32[4]",
        block_size=256,
        num_args=3,
        timing_trials=10,
        grid=(4, 1, 1),
        block=(256, 1, 1),
        shared_memory_bytes=0,
        arg_nbytes=16,
    )
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_runtime_objective_with_tensor_maps_is_picklable_and_hashable():
    spec = TensorMapSpec(dtype="bf16", dims=(2048, 8), box=(32, 8))
    obj = RuntimeObjective(
        ptx_path="/tmp/x.ptx",
        kernel_name="k",
        arch="sm_90a",
        shape="bf16[8,2048]",
        block_size=128,
        num_args=2,
        timing_trials=10,
        grid=(128, 1, 1),
        block=(128, 1, 1),
        shared_memory_bytes=4096,
        arg_nbytes=32768,
        tensor_maps=((1, spec),),
    )
    assert pickle.loads(pickle.dumps(obj)) == obj
    assert hash(obj)
    assert obj.tensor_map_specs == {1: spec}

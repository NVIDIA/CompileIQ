import time
import string
import random
import pytest
from compileiq.types import Worker, WorkerTypes, BASELINE_CONFIG, INVALID_SCORE
from compileiq.worker import MultiProcessWorker, RayWorker, AsyncWorker, IsoMultiProcessWorker
import pandas as pd
from typing import Callable

TEST_WORKERS = [
    pytest.param(WorkerTypes.NATIVE, marks=pytest.mark.requires_ipc),
    pytest.param(WorkerTypes.ISOLATED, marks=pytest.mark.requires_ipc),
    pytest.param(WorkerTypes.RAY, marks=pytest.mark.requires_ray),
    WorkerTypes.ASYNC,
]
TEST_WORKER_CLASSES = [
    pytest.param(MultiProcessWorker, marks=pytest.mark.requires_ipc),
    pytest.param(IsoMultiProcessWorker, marks=pytest.mark.requires_ipc),
    pytest.param(RayWorker, marks=pytest.mark.requires_ray),
    AsyncWorker,
]


def generate_params(num: int = 32, nested: bool = False, duration: float | int = 0.1):
    """
    Generates a list of params with `num` params
    """
    if nested:
        keys = ["x_xx1", "y_yy1", "z_zz1_zzz1", "duration"]
    else:
        keys = ["x", "y", "z", "duration"]

    params = []
    for _ in range(num):
        dict_params = {key: random.random() for key in keys}
        dict_params["duration"] = duration
        params.append(dict_params)

    return params


def light_obj_func(config):
    if config == BASELINE_CONFIG:
        score = 2.0
    else:
        score = config["x"] ** 2 + config["y"]

    return score


def always_invalid_score_func(args):
    return INVALID_SCORE


async def async_always_invalid_score_func(args):
    return INVALID_SCORE


async def async_light_obj_func(config):
    return light_obj_func(config)


def nested_light_obj_func(config):
    if config == BASELINE_CONFIG:
        score = 2.0
    else:
        score = config["x"]["xx1"] ** 2 + config["z"]["zz1"]["zzz1"]

    return score


async def async_nested_light_obj_func(config):
    return nested_light_obj_func(config)


def funky_obj_func(config):
    return random.choice(["*", float("nan"), float("inf"), float("-inf")])


async def async_funky_obj_func(config):
    return funky_obj_func(config)


def fail_obj_func(config):
    return random.choice(string.ascii_letters)


async def async_fail_obj_func(config):
    return fail_obj_func(config)


def multi_light_obj_func(config):
    if config == BASELINE_CONFIG:
        score_1, score_2, score_3, score_4, score_5 = [1, 2, 3, 4, 5]
    else:
        score_1 = config["x"] ** 2 + config["y"]
        score_2 = config["y"] ** 2 + config["x"]
        score_3 = config["z"] ** 2 + config["x"]
        score_4 = config["z"] ** 2 + config["y"]
        score_5 = config["y"] ** 2 + config["z"]
    return score_1, score_2, score_3, score_4, score_5


async def async_multi_light_obj_func(config):
    return multi_light_obj_func(config)


def heavy_obj_func(config):
    score = 0.0
    if config == BASELINE_CONFIG:
        score = 2.0
    else:
        st = time.time()
        while time.time() - st < config["duration"]:
            score = config["y"] ** config["x"] + config["z"]

    return score


async def async_heavy_obj_func(config):
    return heavy_obj_func(config)


ASYNC_FUNC_MAP = {
    light_obj_func: async_light_obj_func,
    multi_light_obj_func: async_multi_light_obj_func,
    nested_light_obj_func: async_nested_light_obj_func,
    funky_obj_func: async_funky_obj_func,
    fail_obj_func: async_fail_obj_func,
    heavy_obj_func: async_heavy_obj_func,
}


def validate_scores(
    df: pd.DataFrame, func: Callable, num_returns: int = 1, normalize: bool = False
):
    if normalize:
        score_cols = df.columns[df.columns.str.contains(pat=r"\bnorm_score_\d+\b")].to_list()
    else:
        score_cols = df.columns[df.columns.str.contains(pat=r"\bscore_\d+\b")].to_list()

    measured_baseline = df[df["params"] == BASELINE_CONFIG]
    df["val_score"] = df["params"].apply(lambda x: [func(x)] if num_returns == 1 else list(func(x)))
    df["combined"] = df[score_cols].values.tolist()
    if normalize:
        assert len(measured_baseline) > 0, "Missing baseline measurement from df"
        df = df.drop(measured_baseline.index, axis="index")
        # Recalc baseline here
        baseline_score = func(BASELINE_CONFIG)
        if num_returns == 1:
            baseline_score = [baseline_score]

        df["val_score"] = df["val_score"].apply(
            lambda score: Worker.normalize_scores(score, baseline_score)
        )

    df["check"] = df["val_score"] == df["combined"]
    print(df[["val_score", "combined", "check"]])
    assert df["check"].all(), df[["val_score", "combined", "check"]][~df["check"]].to_string()

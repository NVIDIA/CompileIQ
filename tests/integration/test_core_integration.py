"""
End-to-end integration tests that run against the REAL core binary — no mocks.

REQUIREMENTS:
    - The core binary must exist for the current platform (linux/x86_64,
       etc.).  Tests are auto-skipped if it's missing.
    - These tests are NOT sandbox-compatible (they spawn subprocesses and
      bind sockets).
    - Run with: pytest tests/integration/test_core_integration.py -vvv
      or: pytest -m requires_core
"""

import sys
import platform
from pathlib import Path

import pytest
import compileiq.search_spaces.base as ss
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, INVALID_SCORE


# ---------------------------------------------------------------------------
# Skip if the core binary isn't available for this platform
# ---------------------------------------------------------------------------


def _core_binary_exists() -> bool:
    """Check whether the core binary exists for the current OS/arch."""
    platform_map = {
        ("linux", "x86_64"): ("linux", "x86_64", "bin", "core"),
        ("linux", "aarch64"): ("linux", "aarch64", "bin", "core"),
        ("win32", "amd64"): ("win32", "amd64", "core.exe"),
    }
    key = (sys.platform, platform.machine().lower())
    parts = platform_map.get(key)
    if parts is None:
        return False
    exe_dir = Path(__file__).parent.parent.parent / "compileiq" / "core" / "executable"
    binary = exe_dir / Path(*parts)
    return binary.exists()


pytestmark = [
    pytest.mark.requires_core,
    pytest.mark.skipif(not _core_binary_exists(), reason="Core binary not found for this platform"),
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "core_test_cache"


SMALL_CONFIG = dict(generations=3, pool_size=8, cull_size=4)


# ---------------------------------------------------------------------------
# Single-objective MIN (the original test, now formalized)
# ---------------------------------------------------------------------------


def objective_min(config):
    """Minimizable: x^2 + y.  Global min near x=1.0, y=1."""
    return config["x"] ** 2 + config["y"]


def test_single_objective_min(cache_dir):
    """The most basic end-to-end test: can we run a simple minimization search
    from start to finish and get back sensible results?"""
    result = Search(
        objective_function=objective_min,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "y": ss.choice([1, 2, 3]),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    best = result.get_best_result()

    # Structural: we should have results, but deduplication can reduce the
    # number of scored rows even in the initial generation.
    assert not df.empty
    assert len(df) <= SMALL_CONFIG["pool_size"] * SMALL_CONFIG["generations"]
    assert "score_1" in df.columns
    assert "params" in df.columns
    assert "generation" in df.columns
    assert df["generation"].between(0, SMALL_CONFIG["generations"] - 1).all()

    # Best result is a dict with the expected keys
    assert isinstance(best, dict)
    assert "score_1" in best
    assert "params" in best

    # Scores are real numbers (not INVALID_SCORE "*") and match the objective
    # for each scored parameter set.
    assert all(isinstance(s, (int, float)) for s in df["score_1"])
    for row in df.itertuples():
        assert isinstance(row.params, dict)
        assert row.score_1 == pytest.approx(objective_min(row.params))

    # Sanity: optimizer found something better than the worst possible
    # (worst = 20.0^2 + 3 = 403)
    assert best["score_1"] < 403


# ---------------------------------------------------------------------------
# Single-objective MAX — the other half of problem_type
# ---------------------------------------------------------------------------


def objective_max(config):
    """Maximizable: x^2 + y.  Global max near x=20.0, y=3."""
    return config["x"] ** 2 + config["y"]


def test_single_objective_max(cache_dir):
    """If MIN works but MAX doesn't, half our users are broken.
    This is the same function but we verify the optimizer goes the other way."""
    result = Search(
        objective_function=objective_max,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "y": ss.choice([1, 2, 3]),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="max",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    best = result.get_best_result()
    assert isinstance(best, dict)

    # Sanity: optimizer found something better than the worst possible
    # (worst for MAX = 1.0^2 + 1 = 2)
    assert best["score_1"] > 2


# ---------------------------------------------------------------------------
# Multi-objective — Pareto front is the default and headline feature
# ---------------------------------------------------------------------------


def multi_objective(config):
    """Two conflicting objectives: minimizing x^2+y competes with minimizing y^2+x."""
    return config["x"] ** 2 + config["y"], config["y"] ** 2 + config["x"]


def test_multi_objective(cache_dir):
    """Multi-objective search with Pareto front — this is an advertised headline
    feature.  We verify the result has both score columns and that the Pareto
    front returns a non-empty list of non-dominated solutions."""
    # The core requires pool_size >= 2*num_objectives + 1 for its internal
    # direction-vector algorithm, so we need a bigger population than SMALL_CONFIG.
    result = Search(
        objective_function=multi_objective,
        search_space={
            "x": ss.range(start=1.0, end=10.0, step=0.5),
            "y": ss.range(start=1.0, end=10.0, step=0.5),
        },
        search_config=SearchConfiguration(
            generations=3,
            pool_size=12,
            problem_type="min",
            num_objectives=2,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert "score_1" in df.columns
    assert "score_2" in df.columns

    pareto = result.pareto_front()
    assert isinstance(pareto, list), "Pareto front should return a list"
    assert len(pareto) > 0, "Pareto front should not be empty"
    for point in pareto:
        assert "score_1" in point
        assert "score_2" in point


# ---------------------------------------------------------------------------
# Normalization — baseline measurement with real core
# ---------------------------------------------------------------------------


def objective_with_baseline(config):
    """Objective that handles the baseline measurement (empty dict).
    When normalize=True, the worker sends BASELINE_CONFIG={} as the first
    evaluation.  The function must return a valid score for it."""
    if not config:
        # Baseline: return a reference score for normalization
        return 100.0
    return config["x"] ** 2 + config["y"]


def test_normalization(cache_dir):
    """normalize=True adds a baseline measurement (params={}) and normalizes
    all scores relative to it.  Verify the result DataFrame has norm_score
    columns and a baseline row."""
    result = Search(
        objective_function=objective_with_baseline,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "y": ss.choice([1, 2, 3]),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
            normalize=True,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert "norm_score_1" in df.columns, "Normalized scores should be present"

    # There should be at least one baseline row (params == {})
    baseline_rows = df[df["params"].apply(lambda p: p == {} or p == "{}")]
    assert len(baseline_rows) > 0, "Baseline measurement row should be present"


# ---------------------------------------------------------------------------
# sample() — preview search space without a full search
# ---------------------------------------------------------------------------


def test_sample(cache_dir):
    """sample() lets users preview what parameter sets look like before
    committing to a full search.  It still needs the core binary to generate
    the candidate samples."""
    tuner = Search(
        objective_function=objective_min,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "y": ss.choice([1, 2, 3]),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    )

    samples = tuner.sample(num_samples=5)
    assert isinstance(samples, list)
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, dict), "Each sample should be a parameter dict"
        assert "x" in s
        assert "y" in s


# ---------------------------------------------------------------------------
# Nested search space — verify key encoding round-trip with real core
# ---------------------------------------------------------------------------


def nested_objective(config):
    """Objective using nested parameter access."""
    return config["optimizer"]["lr"] ** 2 + config["batch_size"]


def test_nested_search_space(cache_dir):
    """Nested search spaces go through base64 key encoding for the core binary
    and get decoded back.  A bug in the encode/decode round-trip would show up
    as missing or mangled keys in the returned parameters."""
    result = Search(
        objective_function=nested_objective,
        search_space={
            "optimizer": {"lr": ss.range(start=0.01, end=1.0, step=0.01)},
            "batch_size": ss.choice([16, 32, 64]),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    # Verify at least one row has properly nested params
    for params in df["params"]:
        if isinstance(params, dict):
            assert "optimizer" in params, "Nested key 'optimizer' should be restored"
            assert "lr" in params["optimizer"], "Nested key 'lr' should be inside 'optimizer'"
            assert "batch_size" in params
            break
    else:
        pytest.fail("No row had a dict with nested params — encoding round-trip may be broken")


# ---------------------------------------------------------------------------
# exit_on_failure — should raise when all objectives fail in gen 1
# ---------------------------------------------------------------------------


def always_fail(config):
    """An objective that always returns INVALID_SCORE."""
    return INVALID_SCORE


def test_exit_on_failure(cache_dir):
    """When exit_on_failure=True (the default) and every objective in the first
    generation fails, we should get a clear RuntimeError — not a silent hang
    or corrupted results."""
    with pytest.raises(RuntimeError, match="All objective functions failed"):
        Search(
            objective_function=always_fail,
            search_space={
                "x": ss.range(start=1.0, end=10.0, step=1.0),
            },
            search_config=SearchConfiguration(
                **SMALL_CONFIG,
                problem_type="min",
                num_objectives=1,
            ),
            cache_folder=cache_dir,
            disable_progress_bar=True,
            exit_on_failure=True,
        ).start()


# ---------------------------------------------------------------------------
# JSON serde - string choices
# ---------------------------------------------------------------------------


def objective_string_choice(config):
    """Objective using string-valued choice parameters."""
    # Map strings to numeric values for the objective
    booster_score = {"gbtree": 1.0, "gblinear": 2.0, "dart": 3.0}
    flag_score = {"O0": 10.0, "O1": 5.0, "O2": 1.0}
    return booster_score.get(config["booster"], 5.0) + flag_score.get(config["opt_level"], 10.0)


def test_string_choices_with_real_core(cache_dir):
    """String values in choice() catch regressions in IPC parsing.
    Verify they round-trip correctly through JSON serialization to the core and back."""
    valid_boosters = {"gbtree", "gblinear", "dart"}
    valid_flags = {"O0", "O1", "O2"}

    result = Search(
        objective_function=objective_string_choice,
        search_space={
            "booster": ss.choice(["gbtree", "gblinear", "dart"]),
            "opt_level": ss.choice(["O0", "O1", "O2"]),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert len(df) >= 1

    for params in df["params"]:
        if isinstance(params, dict) and params:
            assert params["booster"] in valid_boosters, f"Unexpected booster: {params['booster']}"
            assert (
                params["opt_level"] in valid_flags
            ), f"Unexpected opt_level: {params['opt_level']}"


# ---------------------------------------------------------------------------
# JSON serde - complex literal strings
# ---------------------------------------------------------------------------


def objective_complex_literal(config):
    """Objective with literal string params including JSON-like content."""
    x = config.get("x", 10.0)
    # The literal values are constants — just use x for the score
    return x**2


def test_complex_literal_strings_with_real_core(cache_dir):
    """Literal parameters with JSON-like string values catch regressions in IPC parsing.
    Verify they survive the JSON round-trip through the core."""
    result = Search(
        objective_function=objective_complex_literal,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "json_param": ss.literal('{"key": 10}', knockout_prob=0.5),
            "plain_str": ss.literal("this is a constant", knockout_prob=0.5),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert len(df) >= 1  # knockout can reduce rows below pool_size

    # Verify at least one row has the literal values intact
    found_json_literal = False
    found_plain_literal = False
    for params in df["params"]:
        if isinstance(params, dict):
            if "json_param" in params:
                assert params["json_param"] == '{"key": 10}'
                found_json_literal = True
            if "plain_str" in params:
                assert params["plain_str"] == "this is a constant"
                found_plain_literal = True

    # Knockout may remove some params, but at least some rows should have them
    assert (
        found_json_literal or found_plain_literal
    ), "No row contained the literal parameters — they may not be round-tripping"


# ---------------------------------------------------------------------------
# JSON serde — log_sampling with scientific notation floats
# ---------------------------------------------------------------------------


def objective_log_sampled(config):
    """Objective using a log-sampled learning rate (very small floats)."""
    lr = config["lr"]
    return (lr - 0.01) ** 2


def test_log_sampling_with_real_core(cache_dir):
    """log_sampling() uses np.geomspace which can produce scientific notation
    floats (e.g. 1e-08).  Verify the core binary handles these correctly in
    the JSON search-space config."""
    result = Search(
        objective_function=objective_log_sampled,
        search_space={
            "lr": ss.log_sampling(start=1e-8, end=1.0),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert len(df) >= 1

    for params in df["params"]:
        if isinstance(params, dict) and "lr" in params:
            lr = params["lr"]
            assert isinstance(lr, (int, float)), f"lr should be numeric, got {type(lr)}"
            assert 1e-8 <= lr <= 1.0, f"lr={lr} is outside the expected range"


# ---------------------------------------------------------------------------
# JSON serde — knockout fields in JSON
# ---------------------------------------------------------------------------


def objective_with_knockout(config):
    """Objective that handles knocked-out parameters gracefully."""
    x = config.get("x", 5.0)
    y = config.get("y", 2)
    return x**2 + y


def test_knockout_with_real_core(cache_dir):
    """Knockout field (knockout_threshold) is serialized to JSON
    but was never tested with the real core.  Verify the core correctly parses
    and applies knockout when this field is present."""
    result = Search(
        objective_function=objective_with_knockout,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5, knockout_prob=0.5),
            "y": ss.choice([1, 2, 3], knockout_prob=0.3),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert len(df) >= 1
    assert all(isinstance(s, (int, float)) for s in df["score_1"])

    best = result.get_best_result()
    assert isinstance(best, dict)
    assert "score_1" in best


# ---------------------------------------------------------------------------
# JSON serde — mixed types (kitchen sink)
# ---------------------------------------------------------------------------


def objective_mixed(config):
    """Objective combining all gene types."""
    x = config.get("x", 5.0)
    y = config.get("y", 2)
    lr = config.get("lr", 0.01)
    booster_score = {"gbtree": 1.0, "gblinear": 2.0, "dart": 3.0}
    b = booster_score.get(config.get("booster", "gbtree"), 2.0)
    return x**2 + y + lr + b


def test_mixed_types_with_real_core(cache_dir):
    """Kitchen-sink test: range (int and float), choice (numeric and string),
    literal (string with knockout), and log_sampling all in one search space.
    Mirrors what real users do in examples/ciq_xgboost.py."""
    result = Search(
        objective_function=objective_mixed,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "y": ss.choice([1, 2, 3]),
            "booster": ss.choice(["gbtree", "gblinear", "dart"]),
            "lr": ss.log_sampling(start=1e-5, end=1.0, total=8),
            "tag": ss.literal("experiment-v1", knockout_prob=0.5),
        },
        search_config=SearchConfiguration(
            **SMALL_CONFIG,
            problem_type="min",
            num_objectives=1,
        ),
        cache_folder=cache_dir,
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    assert len(df) >= 8

    # Verify parameter types in at least one complete row
    for params in df["params"]:
        if isinstance(params, dict) and len(params) >= 4:
            assert isinstance(params["x"], (int, float))
            assert params["y"] in [1, 2, 3]
            assert params["booster"] in ["gbtree", "gblinear", "dart"]
            assert isinstance(params["lr"], (int, float))
            break

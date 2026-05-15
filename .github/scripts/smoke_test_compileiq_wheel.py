import os
from pathlib import Path

import compileiq.ciq as ciq_module
import compileiq.search_spaces.base as ss
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration


def objective(config):
    return config["x"] ** 2 + config["y"]


def assert_imported_from_wheel():
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if not workspace:
        return

    workspace_path = Path(workspace).resolve()
    ciq_path = Path(ciq_module.__file__).resolve()

    if ciq_path == workspace_path or workspace_path in ciq_path.parents:
        raise AssertionError(f"compileiq.ciq was imported from checkout path: {ciq_path}")


def main():
    assert_imported_from_wheel()

    result = Search(
        objective_function=objective,
        search_space={
            "x": ss.range(start=1.0, end=20.0, step=0.5),
            "y": ss.choice([1, 2, 3]),
        },
        search_config=SearchConfiguration(
            generations=1,
            pool_size=8,
            cull_size=4,
            problem_type="min",
            num_objectives=1,
        ),
        disable_progress_bar=True,
    ).start()

    df = result.get_results()
    best = result.get_best_result()

    assert len(df) > 0
    assert "score_1" in df.columns
    assert "params" in df.columns
    assert isinstance(best, dict)
    assert "score_1" in best
    assert "params" in best


if __name__ == "__main__":
    main()

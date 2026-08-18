import pandas as pd
import random
from compileiq.types import ProblemType, MultiScoreComparison, INVALID_SCORE, Score
from compileiq.results import SearchResult
import pytest
import os


@pytest.fixture
def multi_objective_results():
    return {
        "metadata": [-1, 0, 1, 2, 3, 4, 5],
        "score_1": [2.0, 22.625, 56, 26, 0.95, 55, 330.2],
        "norm_score_1": [2.0, 22.625, 56, 26, 0.95, 55, 330.2],
        "score_2": [3.33, 25.625, 60, 30, 0.98, 60, 330.5],
        "norm_score_2": [3.33, 25.625, 60, 30, 0.98, 60, 330.5],
        "params": [
            str({}),
            '{"x": 6.5, "y": 3}',
            '{"x": 8.5, "y": 3}',
            '{"x": 7.0, "y": 3, "z": "z"}',
            '{"x": 6.5, "y": 1, "z": "z"}',
            '{"x": 9.5, "y": 2}',
            '{"x": 9.5, "y": 3}',
        ],
    }


@pytest.fixture
def single_objective_results():
    return {
        "metadata": [-1, 0, 1, 2, 3, 4, 5],
        "score_1": [2.0, 22.625, 56, 26, 0.95, 55, 330.2],
        "norm_score_1": [2.0, 22.625, 56, 26, 0.95, 55, 330.2],
        "params": [
            str({}),
            '{"x": 6.5, "y": 3}',
            '{"x": 8.5, "y": 3}',
            '{"x": 7.0, "y": 3, "z": "z"}',
            '{"x": 6.5, "y": 1, "z": "z"}',
            '{"x": 9.5, "y": 2}',
            '{"x": 9.5, "y": 3}',
        ],
    }


def _generate_results(
    num_scores: int, num_entries: int = 100, allow_invalid: bool = True
) -> pd.DataFrame:
    score_cols = [f"score_{i+1}" for i in range(num_scores)]
    cols = ["metadata", "generation", "params"] + score_cols
    df = pd.DataFrame(columns=cols)
    for i in range(num_entries):
        gen_num = random.randint(0, int(num_entries / 4))
        scores = [random.random() for _ in range(num_scores)]
        df.loc[i] = [i, gen_num, {"x": random.random()}] + scores

    if allow_invalid:
        # Replacing a few values with errors
        replace_idxs = random.sample(range(1, num_entries), random.randint(1, int(num_entries / 4)))
        for idx in replace_idxs:
            for col in score_cols:
                # if random.random() < 0.8:
                df.loc[idx, col] = INVALID_SCORE

    return df


@pytest.mark.parametrize(
    "num_scores, score_val, norm_val, normalize",
    [
        (1, 0.95, None, False),
        (1, 0.95, 1.23, True),
        (2, (0.95, 0.87), (1.23, 1.10), True),
    ],
)
def test_add_result(num_scores, score_val, norm_val, normalize):
    result = SearchResult._initialize_empty(
        num_scores=num_scores, problem_type="max", norm_scores=normalize
    )
    score = Score(
        score=score_val,
        norm_score=norm_val,
        params={"x": 1.0},
        param_id=1,
        num_objectives=num_scores,
    )

    result.add_result(score, generation_num=2, normalize=normalize)

    assert len(result.df_results) == 1
    row = result.df_results.iloc[0]
    assert row["metadata"] == ""
    assert row["generation"] == 2
    assert row["params"] == {"x": 1.0}

    if num_scores > 1:
        assert isinstance(score.score, (list, tuple))
        score_vals = list(score.score)
    else:
        score_vals = [score.score]
    for i, v in enumerate(score_vals):
        assert row[f"score_{i + 1}"] == v

    if normalize:
        if num_scores > 1:
            assert isinstance(score.norm_score, (list, tuple))
            norm_vals = list(score.norm_score)
        else:
            norm_vals = [score.norm_score]
        for i, v in enumerate(norm_vals):
            assert row[f"norm_score_{i + 1}"] == v


def test_save(tmp_path):
    df = _generate_results(num_scores=3, num_entries=20, allow_invalid=False)
    result = SearchResult(df=df, problem_type=ProblemType.MAX, num_scores=3)

    # Saving to a test file
    test_filepath = tmp_path / "test_evo_results.csv"
    result.save(test_filepath)

    # Loading back the file to check contents
    loaded = SearchResult.from_csv(
        test_filepath, problem_type=ProblemType.MAX, clear_duplicates=True
    )

    # Checking that the loaded DataFrame matches the original
    pd.testing.assert_frame_equal(df, loaded.df_results)
    os.remove(test_filepath)


def test_class_method(single_objective_results, multi_objective_results):
    result_single: SearchResult = SearchResult.from_dataframe(
        df=pd.DataFrame(single_objective_results),
        problem_type=ProblemType.MAX,
    )

    assert isinstance(result_single, SearchResult)
    assert result_single.num_scores == 1

    result_multi = SearchResult.from_dataframe(
        df=pd.DataFrame(multi_objective_results),
        problem_type=ProblemType.MIN,
    )

    assert isinstance(result_multi, SearchResult)
    assert result_multi.num_scores == 2


def test_score_column_matching_ignores_substring_columns():
    df = pd.DataFrame(
        {
            "score_1": [1.0],
            "baseline_score_1": [2.0],
            "params": ['{"a": 1}'],
        }
    )

    result = SearchResult.from_dataframe(df=df, problem_type=ProblemType.MIN)

    assert result.num_scores == 1
    assert result.score_columns == ["score_1"]
    best_result = result.get_best_result()
    assert isinstance(best_result, dict)
    assert best_result["score_1"] == 1.0


def test_singlescore_results(single_objective_results):
    correct_result = {
        ProblemType.MAX: {
            "metadata": 5,
            "score_1": 330.2,
            "norm_score_1": 330.2,
            "params": {"x": 9.5, "y": 3},
        },
        ProblemType.MIN: {
            "metadata": 3,
            "score_1": 0.95,
            "norm_score_1": 0.95,
            "params": {"x": 6.5, "y": 1, "z": "z"},
        },
    }

    for pt in ProblemType:
        result = SearchResult.from_dataframe(
            df=pd.DataFrame(single_objective_results), problem_type=pt
        )
        assert result.get_best_result() == correct_result[pt]


def test_multiscore_results(multi_objective_results):
    correct_result = {
        ProblemType.MAX: {
            MultiScoreComparison.AVERAGE: {
                "metadata": 5,
                "score_1": 330.2,
                "score_2": 330.5,
                "norm_score_1": 330.2,
                "norm_score_2": 330.5,
                "params": {"x": 9.5, "y": 3},
            },
            MultiScoreComparison.STDDEV: {
                "metadata": 4,
                "score_1": 55.0,
                "score_2": 60.0,
                "norm_score_1": 55.0,
                "norm_score_2": 60.0,
                "params": {"x": 9.5, "y": 2},
            },
        },
        ProblemType.MIN: {
            MultiScoreComparison.AVERAGE: {
                "metadata": 3,
                "score_1": 0.95,
                "score_2": 0.98,
                "norm_score_1": 0.95,
                "norm_score_2": 0.98,
                "params": {"x": 6.5, "y": 1, "z": "z"},
            },
            MultiScoreComparison.STDDEV: {
                "metadata": 3,
                "score_1": 0.95,
                "score_2": 0.98,
                "norm_score_1": 0.95,
                "norm_score_2": 0.98,
                "params": {"x": 6.5, "y": 1, "z": "z"},
            },
        },
    }

    for pt in ProblemType:
        # TODO: Add test for PARETO front
        result = SearchResult.from_dataframe(
            df=pd.DataFrame(multi_objective_results), problem_type=pt
        )
        assert (
            result.get_best_result(multiscore_scope=MultiScoreComparison.AVERAGE.value)
            == correct_result[pt][MultiScoreComparison.AVERAGE]
        )


def test_invalid_results():
    dfs = [
        _generate_results(num_scores=4),
        _generate_results(num_scores=2),
        _generate_results(num_scores=1),
    ]
    for df in dfs:
        num_scores = len([col for col in df.columns if "score" in col])
        for pt in ProblemType:
            result = SearchResult(df=df, problem_type=pt, num_scores=num_scores)
            for scope in MultiScoreComparison:
                print("Testing:", pt, scope)
                if scope == MultiScoreComparison.PARETO:
                    if result.num_scores == 1:
                        continue
                    best = result.pareto_front()
                else:
                    best = result.get_best_result(multiscore_scope=scope.value)

                if not isinstance(best, list):
                    best = [best]

                for b in best:
                    for col in result.score_columns:
                        assert INVALID_SCORE not in str(b[col]), f"{best} | {pt} | {scope}"


def test_raise_on_all_failures():
    dfs = [
        _generate_results(num_scores=4),
        _generate_results(num_scores=2),
        _generate_results(num_scores=1),
    ]
    for df in dfs:
        df[df.columns[df.columns.str.contains(pat=r"score_\d+")]] = INVALID_SCORE
        num_scores = len([col for col in df.columns if "score" in col])
        result = SearchResult(df=df, problem_type=ProblemType.MIN, num_scores=num_scores)

        with pytest.raises(
            ValueError,
            match="All resulting scores are marked as invalid. "
            "Make sure your search ran correctly.",
        ):
            result.get_best_result()

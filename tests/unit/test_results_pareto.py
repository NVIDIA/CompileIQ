"""
Tests for Pareto front calculation in compileiq/results.py.
"""

import numpy as np
import pandas as pd
from compileiq.results import SearchResult
from compileiq.types import ProblemType, INVALID_SCORE


def _make_result(scores_data, problem_type=ProblemType.MIN):
    """Helper: build a minimal SearchResult from a list of (score_1, score_2) tuples."""
    num_scores = len(scores_data[0])
    score_cols = [f"score_{i+1}" for i in range(num_scores)]
    cols = ["metadata", "generation"] + score_cols + ["params"]
    rows = []
    for i, scores in enumerate(scores_data):
        rows.append([i, 0] + list(scores) + [{"x": i}])
    df = pd.DataFrame(rows, columns=cols)
    return SearchResult(df=df, problem_type=problem_type, num_scores=num_scores)


# ---------------------------------------------------------------------------
# calculate_pareto_front — the core algorithm
# ---------------------------------------------------------------------------


class TestCalculatePareteFrontMin:
    """For minimization: point A dominates point B if A is <= B on all
    objectives and strictly < on at least one."""

    def test_one_dominated_point_excluded(self):
        """Point (5,5) is dominated by (1,1) — it should NOT be on the front."""
        result = _make_result([(1, 1), (5, 5), (1, 5)])
        scores = np.array([(1, 1), (5, 5), (1, 5)])
        mask = result.calculate_pareto_front(scores)

        assert mask[0] is np.True_  # (1,1) is on the front
        assert mask[1] is np.False_  # (5,5) is dominated by (1,1)
        assert mask[2] is np.False_  # (1,5) ties (1,1) on dim 0 and is worse on dim 1

    def test_point_tied_on_one_objective_and_worse_on_another_excluded(self):
        """A tie on one objective does not protect a row that is worse on another."""
        scores = np.array([(1, 5), (1, 1)])
        result = _make_result([(1, 5), (1, 1)])
        mask = result.calculate_pareto_front(scores)
        assert mask.tolist() == [False, True]

    def test_identical_points_kept_and_dominated_row_excluded(self):
        """Exact ties stay on the front together; a row that one of them dominates does not."""
        scores = np.array([(2, 3), (2, 4), (2, 3)])
        result = _make_result([(2, 3), (2, 4), (2, 3)])
        mask = result.calculate_pareto_front(scores)
        assert mask.tolist() == [True, False, True]

    def test_all_on_front(self):
        """When no point dominates another, all should be on the front."""
        # Trade-off: better on one objective means worse on the other
        scores = np.array([(1, 10), (5, 5), (10, 1)])
        result = _make_result([(1, 10), (5, 5), (10, 1)])
        mask = result.calculate_pareto_front(scores)
        assert all(mask), "All points are non-dominated, all should be on the front"

    def test_single_point(self):
        """A single point is trivially on the Pareto front."""
        scores = np.array([(3, 4)])
        result = _make_result([(3, 4)])
        mask = result.calculate_pareto_front(scores)
        assert mask[0] is np.True_

    def test_identical_points_both_on_front(self):
        """Two identical points: neither dominates the other, so both are
        on the front (they represent different parameter sets with equal scores)."""
        scores = np.array([(2, 3), (2, 3)])
        result = _make_result([(2, 3), (2, 3)])
        mask = result.calculate_pareto_front(scores)
        assert all(mask)


class TestCalculateParetoFrontMax:
    """For maximization: point A dominates point B if A is >= B on all
    objectives and strictly > on at least one."""

    def test_dominated_point_excluded(self):
        """Point (1,1) is dominated by (5,5) in a MAX problem."""
        result = _make_result([(5, 5), (1, 1), (5, 1)], problem_type=ProblemType.MAX)
        scores = np.array([(5, 5), (1, 1), (5, 1)])
        mask = result.calculate_pareto_front(scores)

        assert mask[0] is np.True_  # (5,5) dominates
        assert mask[1] is np.False_  # (1,1) is dominated
        assert mask[2] is np.False_  # (5,1) ties (5,5) on dim 0 and is worse on dim 1

    def test_point_tied_on_one_objective_and_worse_on_another_excluded(self):
        scores = np.array([(5, 1), (5, 5)])
        result = _make_result([(5, 1), (5, 5)], problem_type=ProblemType.MAX)
        mask = result.calculate_pareto_front(scores)
        assert mask.tolist() == [False, True]


# ---------------------------------------------------------------------------
# pareto_front — end-to-end
# ---------------------------------------------------------------------------


class TestGetBestResultPareto:
    """Verify the full Pareto pipeline returns a list of dicts."""

    def test_returns_list_of_dicts(self):
        result = _make_result([(1, 10), (5, 5), (10, 1)])
        best = result.pareto_front()
        assert isinstance(best, list), "Pareto front should return a list, not a single dict"
        assert all(isinstance(b, dict) for b in best)

    def test_get_best_result_pareto_scope_matches_pareto_front(self):
        result = _make_result([(1, 10), (5, 5), (10, 1)])
        assert result.get_best_result(multiscore_scope="pareto_front") == result.pareto_front()

    def test_get_best_result_defaults_to_pareto_front_for_multi_score(self):
        result = _make_result([(1, 10), (5, 5), (10, 1)])
        assert result.get_best_result() == result.pareto_front()

    def test_excludes_invalid_scores(self):
        """Rows with INVALID_SCORE ("*") should be filtered out before
        computing the front — you can't have a Pareto point with missing data."""
        scores_data = [(1, 10), (5, 5)]
        result = _make_result(scores_data)
        # Inject an invalid row
        invalid_row = [99, 0, INVALID_SCORE, INVALID_SCORE, {"x": 99}]
        result.df_results.loc[len(result.df_results)] = invalid_row

        best = result.pareto_front()
        # The invalid row should not appear in results
        for b in best:
            assert b.get("score_1") != INVALID_SCORE
            assert b.get("score_2") != INVALID_SCORE

    def test_matches_brute_force_with_discrete_objectives(self):
        """Compiler objectives such as spill bytes or register counts tie often; the
        front must still hold exactly the rows that no other row dominates."""

        def dominates(a, b):
            return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

        rng = np.random.default_rng(0)
        for _ in range(50):
            rows = [
                (int(rng.integers(0, 6)), int(rng.integers(0, 4)), int(rng.integers(0, 3)))
                for _ in range(30)
            ]
            result = _make_result(rows)
            got = sorted(r["metadata"] for r in result.pareto_front())
            expected = [
                i for i, row in enumerate(rows) if not any(dominates(o, row) for o in rows)
            ]
            assert got == expected

    def test_pareto_front_with_clear_winner(self):
        """When one point dominates all others, only it should be on the front."""
        result = _make_result([(1, 1), (5, 5), (10, 10)])
        best = result.pareto_front()
        assert len(best) == 1
        assert best[0]["score_1"] == 1
        assert best[0]["score_2"] == 1

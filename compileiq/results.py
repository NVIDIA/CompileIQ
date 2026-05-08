import json
import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Literal
from compileiq.types import ProblemType, MultiScoreComparison, Score, SingleScore, MultiScore


class SearchResult:
    """
    Stores results from a CompileIQ search in a pandas DataFrame.

    Attributes:

        df_results (pd.DataFrame):
            DataFrame containing the search results, including metadata, generation number,
            scores, and parameters.
        problem_type (ProblemType):
            Indicates whether the optimization problem is a minimization or maximization task.
        num_scores (int):
            The number of scores recorded for each set of parameters.
        score_columns (List[str]):
            List of column names in the DataFrame that contain the scores.

    """

    def __init__(self, df: pd.DataFrame, problem_type: ProblemType, num_scores: int):
        self.df_results = df
        self.problem_type = problem_type
        self.num_scores = num_scores
        self.original_cols = df.columns

    @property
    def score_columns(self) -> list:
        return self.df_results.columns[
            self.df_results.columns.str.contains(pat=r"score_\d+")
        ].to_list()

    @property
    def is_normalized(self) -> bool:
        return bool(self.df_results.columns.str.contains(pat=r"norm_score_\d+").any())

    @classmethod
    def from_csv(
        cls, csv_path: str, problem_type: str | ProblemType, clear_duplicates: bool = True
    ) -> "SearchResult":
        """
        Loads dataframe into an SearchResult object.

        Args:
            csv_path:
                Path to usual CompileIQ `dump_results` csv file or previous SearchResult dataframe.

            problem_type:
                Problem type, either "min" or "max".

            clear_duplicates:
                If True, will remove duplicate rows based on the `params` column.

        Returns:
            SearchResult object
        """

        df = pd.read_csv(csv_path)

        return cls.from_dataframe(
            df=df,
            problem_type=problem_type,
            clear_duplicates=clear_duplicates,
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, problem_type: str | ProblemType, clear_duplicates: bool = True
    ) -> "SearchResult":
        """
        Loads dataframe into an SearchResult object.

        Args:
            df:
                Dataframe with the results.

            problem_type:
                Problem type, either "min" or "max".

            clear_duplicates:
                If True, will remove duplicate rows based on the `params` column.

        Returns:
            SearchResult object
        """

        score_cols = df.columns[df.columns.str.contains(pat=r"\bscore_\d+\b")]

        if len(score_cols) == 0:
            raise ValueError(
                "Score columns are not formatted correctly. Must be named `score_<num>`"
            )

        if "params" not in df.columns:
            raise ValueError("Missing 'params' column in the dataframe.")
        else:

            def json_or_keep(x):
                try:
                    return json.loads(x)
                except (json.JSONDecodeError, TypeError):
                    return x

            df["params"] = df["params"].apply(json_or_keep)

        result = SearchResult(
            df=df,
            problem_type=ProblemType(problem_type),
            num_scores=len(score_cols),
        )

        if clear_duplicates:
            result.clear_duplicates()

        return result

    @classmethod
    def _initialize_empty(
        cls, num_scores: int, problem_type: str | ProblemType, norm_scores: bool = False
    ) -> "SearchResult":
        """
        Initializes an SearchResult object with an empty dataframe.
        """
        cols = (
            ["metadata", "generation"] + [f"score_{i + 1}" for i in range(num_scores)] + ["params"]
        )
        if norm_scores:
            cols += [f"norm_score_{i + 1}" for i in range(num_scores)]
        df = pd.DataFrame(columns=cols)
        result = SearchResult(
            df=df,
            problem_type=ProblemType(problem_type),
            num_scores=num_scores,
        )

        return result

    def _score_values(self, value: SingleScore | MultiScore | None) -> list:
        if self.num_scores > 1:
            assert isinstance(value, (list, tuple))
            return list(value)
        return [value]

    def add_result(self, score: Score, generation_num: int, normalize: bool = False) -> None:
        """
        Appends a single evaluated score to the results DataFrame.

        Args:
            score: a single `Score` object returned from the Worker.
            generation_num: The generation this score was produced in.
            normalize: Whether normalized scores should also be recorded.
        """
        row = [score.metadata, generation_num]
        row += self._score_values(score.score)
        row += [score.params]
        if normalize:
            row += self._score_values(score.norm_score)
        self.df_results.loc[len(self.df_results)] = row

    def _numeric_scores_df(self) -> pd.DataFrame:
        scores_df = self.df_results[self.score_columns].apply(pd.to_numeric, errors="coerce").copy()
        if self.is_normalized:
            norm_score_columns = self.df_results.columns[
                self.df_results.columns.str.contains(pat=r"norm_score_\d+")
            ]
            scores_df = scores_df[norm_score_columns].copy()

        if scores_df.columns.size != self.num_scores:
            raise ValueError(
                "Number of score columns does not match the initialized number of scores. "
                "If using normalized scores make sure your baseline measurements are present."
            )
        return scores_df

    def get_best_result(
        self,
        multiscore_scope: (
            MultiScoreComparison | Literal["avg", "stddev", "pareto_front"]
        ) = MultiScoreComparison.PARETO,
    ) -> Dict | List[Dict]:
        """
        Returns the best result from your search.

        Args:
            multiscore_scope:
                For multi-score searches, how to aggregate scores before picking
                results. The default preserves existing behavior: single-score
                searches return the best row, while multi-score searches return
                the Pareto front.

        Returns:
            Dictionary with the best search scores, parameters and generation number,
            or a list of dictionaries when returning a Pareto front.
        """
        scores_df = self._numeric_scores_df()

        if scores_df.isna().all().all():
            raise ValueError(
                "All resulting scores are marked as invalid. Make sure your search ran correctly."
            )

        if self.num_scores == 1:
            scores_df["score"] = scores_df
        else:
            scope = MultiScoreComparison(multiscore_scope)
            if scope == MultiScoreComparison.PARETO:
                return self.pareto_front()
            if scope == MultiScoreComparison.AVERAGE:
                scores_df["score"] = scores_df.agg("mean", axis="columns")
            elif scope == MultiScoreComparison.STDDEV:
                scores_df["score"] = scores_df.agg("std", axis="columns")
            else:
                raise ValueError(f"Invalid multi-score comparison scope: {scope}")

        if self.problem_type == ProblemType.MAX:
            best_idx = scores_df["score"].idxmax()
        elif self.problem_type == ProblemType.MIN:
            best_idx = scores_df["score"].idxmin()
        else:
            raise ValueError(f"Problem type needs to be {ProblemType.MAX} or {ProblemType.MIN}")

        best_score = self.df_results.loc[best_idx]
        return best_score.to_dict()

    def pareto_front(self) -> List[Dict]:
        """
        Returns the Pareto-efficient rows from a multi-score search.
        """
        if self.num_scores == 1:
            raise ValueError("pareto_front is only meaningful for multi-score searches.")

        scores_df = self._numeric_scores_df()
        scores_df = scores_df[~scores_df.isnull().values.any(axis=1)]
        mask = self.calculate_pareto_front(scores_df.to_numpy())
        fdf = self.df_results.loc[scores_df.index]
        return fdf[mask].to_dict(orient="records")

    def get_results(self) -> pd.DataFrame:
        """
        Returns:
            Dataframe with all evaluated parameters
        """
        return self.df_results

    def __getitem__(self, idx: int):
        return self.df_results.iloc[idx]

    def calculate_pareto_front(self, scores: np.ndarray) -> npt.NDArray[np.bool_]:
        """
        Find the pareto-efficient points given a list of scores.

        Note:
            Using `pareto_front()` will yield the same Pareto front.

        Args:
            scores:
                An (n_params, n_scores) array with the scores.
                Position [i,j] is the jth score for ith dna

        Returns:
            A mask (boolean array) of size (n_params) indicating which rows from
            `scores` were selected
        """
        is_efficient = np.arange(scores.shape[0])
        n_points = scores.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for

        while next_point_index < len(scores):
            if self.problem_type == ProblemType.MIN:
                nondominated_point_mask = np.any(scores <= scores[next_point_index], axis=1)
            else:
                nondominated_point_mask = np.any(scores >= scores[next_point_index], axis=1)

            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            scores = scores[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True

        return is_efficient_mask

    def clear_duplicates(self) -> pd.DataFrame:
        """
        Core will often send the same params to be calculated.
        If the calculated scores are exactly the same as the previous time,
        we can just drop these rows.
        """
        check_cols = ["params_str"] + self.score_columns

        self.df_results["params_str"] = self.df_results["params"].apply(lambda x: json.dumps(x))
        df_dropped = self.df_results[check_cols].drop_duplicates()

        self.df_results = self.df_results[self.original_cols].loc[df_dropped.index]

        # Making sure all scores are numeric
        self.df_results[self.score_columns] = self.df_results[self.score_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        return self.df_results

    def save(self, filepath: str | os.PathLike = "./search_results.csv"):
        """
        Saves entire results dataframe to a CSV file. It will convert the `params`
        column into a JSON string for better parsing.
        Args:
            filepath:
                Path to save the CSV file. Defaults to `./search_results.csv`.
        """
        self.df_results.apply(lambda x: x.apply(json.dumps) if x.name == "params" else x).to_csv(
            filepath, index=False
        )

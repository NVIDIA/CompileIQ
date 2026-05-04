from compileiq.types import Worker, SearchConfiguration, INVALID_SCORE, BASELINE_DNA
from compileiq.utils.validation import Score
from loguru import logger
from compileiq.ciq import Search
import compileiq.search_spaces.base as ss


class SequentialWorker(Worker):
    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(
        self,
        *,
        function: callable,
        params_pool: list[dict | str],
        params_ids: list[int],
        num_function_returns: int = 1,
        **kwargs,
    ) -> list[Score]:
        # Any additional params passed to .start can be accessed here.
        logger.debug(f"Here is your additional param: `{kwargs.get('dummy_arg', None)}`")
        scores = []

        # It is your responsability to handle baseline score and normalization.
        # In this example, we will calculate the baseline score only once for the search
        if self.normalize and self.baseline_score is None:
            logger.info("Calculating Baseline score for normalization.")
            baseline_score = function(BASELINE_DNA)
            # Here we save the baseline score to reuse on future batches/generations.
            self.baseline_score = Score(
                score=baseline_score,
                param_id="baseline",
                params=BASELINE_DNA,
                norm_score=self.normalize_scores(baseline_score, baseline_score),
                num_objectives=num_function_returns,
                is_baseline=True,
            )
            scores.append(self.baseline_score)

        # Here is the algorithm to execute the user-provided objective.
        # A Simple sequential execution for this worker.
        # `Worker` provides you with utilities to handle normalization and validation.
        for i, param in enumerate(params_pool):
            try:
                func_return = function(param)
            except Exception as e:
                # Make sure to handle any uncatched exceptions or the search will be interrupted.
                logger.warning(
                    f"Unhandled exception {e} on your objective function with params {param}"
                    "We will return a invalid score."
                )
                func_return = (
                    [INVALID_SCORE] * num_function_returns
                    if num_function_returns > 1
                    else INVALID_SCORE
                )

            valid_score = Score(
                score=func_return,
                param_id=params_ids[i],
                params=param,
                num_objectives=num_function_returns,
            )

            # Apply normalization if enabled. If you don't have any plans to use normalization,
            # you can skip this step.
            if self.normalize:
                valid_score.norm_score = self.normalize_scores(
                    valid_score.score, self.baseline_score.score
                )

            scores.append(valid_score)

        # Returning the list of all scores measured this round (including baseline if calculated)
        return scores


def objective(params):
    if params == {}:
        return 2
    else:
        score = params["x"] - params["y"] ** 2
        return score


if __name__ == "__main__":
    dna_config = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
    }

    main_config = SearchConfiguration(
        normalize=True,
        pool_size=12,
        generations=3,
        mutate_rate=0.5,
        problem_type="min",
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=dna_config,
        search_config=main_config,
        worker_type=SequentialWorker,
    )

    results = tuner.start(dummy_arg="You can pass in many custom keyword arguments to the worker.")

    print(results.get_results())

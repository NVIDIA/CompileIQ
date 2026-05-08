"""Multi-objective search + Multi-workers"""

from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
import compileiq.search_spaces.base as ss


def multiobjective(config):
    score_1 = config["x"] ** 2 + config["y"]
    score_2 = config["y"] ** 2 + config["x"]
    return score_1, score_2


def main():
    dna_config = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
    }

    main_config = SearchConfiguration(
        pool_size=12,
        generations=3,
        mutate_rate=0.5,
        problem_type="min",
    )

    tuner = Search.multi_objective(
        objective_function=multiobjective,
        search_space=dna_config,
        search_config=main_config,
        num_objectives=2,
    )

    results = tuner.start(num_workers=2)
    print(results.pareto_front())


if __name__ == "__main__":
    main()

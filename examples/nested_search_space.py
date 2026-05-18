from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
import compileiq.search_spaces.base as ss


def objective(config):
    if "z" in config:
        score = config["x"]["xx1"] ** 2 + config["y"]["yy1"]["yyy1"]["yyyy1"]
    else:
        score = config["y"]["yy2"] ** 2 + config["x"]["xx2"]

    return score


def main():
    search_space_config = {
        "x": {
            "xx1": ss.range(start=1.0, end=20.0, step=0.5),
            "xx2": ss.range(start=5.0, end=11.0, step=0.5),
        },
        "y": {
            "yy1": {
                "yyy1": {"yyyy1": ss.choice([1, 2, 3])},
                "yyy2": ss.choice([1, 2, 3]),
            },
            "yy2": ss.choice([4, 5, 6]),
        },
        "z": ss.literal("z", knockout_prob=0.5),
    }

    main_config = SearchConfiguration(
        pool_size=32,
        generations=3,
        mutate_rate=0.5,
        problem_type="min",
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=search_space_config,
        search_config=main_config,
    )

    results = tuner.start()
    print(f"Entire Results Dataframe:\n {results.get_results()}")
    print(f"Best Result: {results.get_best_result()}")


if __name__ == "__main__":
    main()

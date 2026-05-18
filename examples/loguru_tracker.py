from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, LoguruTrackerConfig
import compileiq.search_spaces.base as ss


def objective(config):
    score = config["x"] ** 2 + config["y"]
    return score


def main():
    search_space_config = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
    }

    # In this example we show how to log to a file, and how to
    # pass additional arguments to the LoguruTracker.
    tracker_config = LoguruTrackerConfig(
        sink=["tracker.log"],
        level="DEBUG",
    )

    main_config = SearchConfiguration(
        pool_size=10,
        generations=2,
        mutate_rate=0.5,
        problem_type="min",
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=search_space_config,
        tracker_config=tracker_config,
        search_config=main_config,
    )

    results = tuner.start()
    print(f"Entire Results Dataframe:\n {results.get_results()}")
    print(f"Best Result: {results.get_best_result()}")


if __name__ == "__main__":
    main()

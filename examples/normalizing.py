"""
This example uses shows the implications of setting normalize=True in the SearchConfiguration.
"""

from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, BASELINE_DNA
from compileiq.worker import MultiProcessWorker
import compileiq.search_spaces.base as ss


def objective(config):
    if config == BASELINE_DNA:
        # Returning a constant score for demonstration purposes.
        score = 0.001
    elif "z" in config:
        score = config["x"] ** 2 + config["y"]
    else:
        score = config["y"] ** 2 + config["x"]

    return score


def main():
    dna_config = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
        "z": ss.literal("this is a constant", knockout_prob=0.5),
    }

    # When `normalize` is set to `True`, Evo will normalize all output scores from your
    # objective using a baseline. Your function needs to be prepared for accept `BASELINE_DNA`
    main_config = SearchConfiguration(
        normalize=True,
        pool_size=12,
        generations=2,
        mutate_rate=0.25,
        problem_type="min",
        num_objectives=1,
    )

    # A baseline is measured once at the beginning.
    # When using RAY, a baseline will be measured for every node in your cluster
    # When using NATIVE/DEFAULT, the baseline will be measured once by a single process
    tuner = Search(
        objective_function=objective,
        search_space=dna_config,
        search_config=main_config,
        worker_type=MultiProcessWorker,
    )

    results = tuner.start(num_workers=5)

    # All shown scores will be normalized and the baseline will be available at idx=-1
    print(results.get_results())


if __name__ == "__main__":
    main()

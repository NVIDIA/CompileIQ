"""
This example uses Ray Workers to enable both local or distributed execution
"""

from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
from compileiq.worker import RayWorker
import compileiq.search_spaces.base as ss
import time


def objective(config):
    # Adding sleep so you can see the ray dashboard
    time.sleep(1)
    if "z" in config:
        score = config["x"] ** 2 + config["y"]
    else:
        score = config["y"] ** 2 + config["x"]

    return score


def main():
    search_space_config = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
        "z": ss.log_sampling(start=1e-4, end=1.0, knockout_prob=0.2),
    }

    main_config = SearchConfiguration(
        pool_size=128,
        generations=10,
        mutate_rate=0.25,
        problem_type="max",
        num_objectives=1,
    )

    # Ray workers can run locally or distributed,
    # A ray dashboard is brought up for resource visualization and task progress
    # For a distributed deployment it is your responsibility to configure your Ray cluster
    #   https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html
    # For Local deployment, nothing needs to be done, this code will run in parallel by itself
    tuner = Search(
        objective_function=objective,
        search_space=search_space_config,
        search_config=main_config,
        worker_type=RayWorker,
    )

    # Ray does not limit cpu usage, this only serves for scheduling purposes
    # For example: If `num_gpus=1`, a node without GPUs would never receive this task
    # For more info read about ray resources:
    #   https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#specifying-task-or-actor-resource-requirements
    # In this example, with num_cpus=1, we will have as many tasks as there are cpus available
    results = tuner.start(num_cpus=1, num_gpus=0, scheduling_strategy="DEFAULT")
    print(results.get_best_result())


if __name__ == "__main__":
    main()

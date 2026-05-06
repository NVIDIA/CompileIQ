from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, MLflowTrackerConfig
from compileiq.worker import RayWorker
import compileiq.search_spaces.base as ss
from mlflow import set_tag, set_tags, log_artifact, trace
import os
import hashlib
import json
import tempfile


def write_mangled_config(config, temp_file):
    """
    Writes the hash of the config to a binary file.
    """
    with open(temp_file, "wb") as f:
        f.write(hashlib.sha256(json.dumps(config).encode()).digest())


# Function tracing is optional. It can be activated by using the trace decorator.
# This will use MLflow's tracing capabilities to trace the function calls.
@trace
def objective(config):
    score = config["x"] ** 2 + config["y"]

    # MLflow tags can be set here
    # They can be used to record important information about the run
    set_tag("metadata", "this is a metadata tag")
    set_tags({"job_id": os.getpid()})

    # Also to keep track of the config
    if "z" in config.keys():
        set_tag("z", "present")
    else:
        set_tag("z", "not present")

    # MLflow supports logging a variety of objects as artifacts, including:
    # files (by path), directories, JSON/YAML-serializable dictionaries,
    # Pandas DataFrames (as CSV), Numpy arrays (as .npy), Matplotlib and Plotly
    # figures (as images), images (as numpy arrays, PIL images, or mlflow.Image),
    # and text.
    #
    # Other objects will be attempted to be pickled with the default protocol.
    # The serialization format for dictionaries is inferred from the file extension,
    # defaulting to JSON if not recognized.
    #
    # See mlflow.log_artifact, mlflow.log_dict, mlflow.log_figure, mlflow.log_image,
    # and mlflow.log_table for details.

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "mangled_config.bin")
        # Function tracing is optional.
        # You can additionally use the trace wrapper to trace any
        # function calls inside of the objective function.
        trace(write_mangled_config)(config, temp_file)
        log_artifact(temp_file)

    return score


def main():
    dna_config = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
        "z": ss.literal("this is a constant", knockout_prob=0.5),
    }

    # The config will be logged as an artifact, by default.
    # This can be disabled by setting log_config=False in the tracker config.
    # You can set a tracking_uri if you have an existing mlflow server. Otherwise,
    # mlflow starts one locally.
    tracker_config = MLflowTrackerConfig(
        experiment_name="test",
        description="Test run in CompileIQ",
    )

    main_config = SearchConfiguration(
        pool_size=32,
        generations=3,
        mutate_rate=0.5,
        problem_type="min",
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=dna_config,
        tracker_config=tracker_config,
        worker_type=RayWorker,
        search_config=main_config,
    )

    results = tuner.start()
    print(f"Entire Results Dataframe:\n {results.get_results()}")
    print(f"Best Result: {results.get_best_result()}")


if __name__ == "__main__":
    main()

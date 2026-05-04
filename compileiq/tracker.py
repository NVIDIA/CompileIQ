import functools
import os
import sys
import warnings
from compileiq.types import (
    BASELINE_DNA,
    INVALID_SCORE,
    BaseTracker,
    TrackerTypes,
    DisabledTrackerConfig,
    LoguruTrackerConfig,
    MLflowTrackerConfig,
)
from loguru import logger

import getpass
from datetime import datetime

try:
    import mlflow
except ImportError:
    mlflow = None


def logging_exception(func):
    """
    A decorator that catches any exceptions raised by the decorated method.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            warnings.warn(
                f"An Exception was caught when trying to log metadata ({func.__name__}). "
                "This will not affect the objective function execution."
                "But you might be losing logging information."
            )

    return wrapper


class DisabledTracker(BaseTracker):
    """
    DisabledTracker is a tracker that does nothing.
    """

    def __init__(self, tracker_config: DisabledTrackerConfig = DisabledTrackerConfig()):
        super().__init__(tracker_config)

    def setup(self, **kwargs):
        pass

    def cleanup(self, **kwargs):
        pass

    def search_starts(self, **kwargs):
        pass

    def search_ends(self, **kwargs):
        pass

    def generation_starts(self, generation_number: int, **kwargs):
        pass

    def generation_ends(self, generation_number: int, **kwargs):
        pass

    def pre_objective(self, config: dict, **kwargs):
        pass

    def post_objective(self, scores, **kwargs):
        pass


class LoguruTracker(BaseTracker):
    def setup(self, **kwargs):
        logger.remove()
        sinks = self.tracker_config.sink if self.tracker_config.sink is not None else [sys.stdout]
        log_args = self.tracker_config.model_dump(serialize_as_any=True)
        log_args.pop("type", None)
        log_args.pop("sink", None)
        for sink in sinks:
            logger.add(sink, **log_args)
            logger.configure(extra={"task_id": "<no-task-id>"})

        logger.debug(f"{len(sinks)} LoguruTrackers initialized with {log_args}")

    def cleanup(self, **kwargs):
        logger.remove()

    def search_starts(self, **kwargs):
        logger.info("Search started")

    # TODO: Add handling for different search outcomes (success, failure, etc.)
    def search_ends(self, **kwargs):
        logger.success("Search ended")

    def generation_starts(self, generation_number: int, **kwargs):
        logger.info("================================================")
        logger.info(f"Generation {generation_number} started")

    @logging_exception
    def generation_ends(self, generation_number: int, **kwargs):
        logger.info(f"Generation {generation_number} ended")
        logger.info("================================================")

    @logging_exception
    def pre_objective(self, config: dict, **kwargs):
        logger.bind(task_id=kwargs.get("task_id", "<missing-task-id>")).info(
            f"Executing objective function with: {config}"
        )

    @logging_exception
    def post_objective(self, scores, **kwargs):
        logger.bind(task_id=kwargs.get("task_id", "<missing-task-id>")).info(f"Scored: {scores}")


class MLflowTracker(BaseTracker):
    @staticmethod
    def get_date_string():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Default values are set on the MLflowTrackerConfig object definition
    def __init__(self, tracker_config: MLflowTrackerConfig):
        super().__init__(tracker_config)
        self.experiment_name = tracker_config.experiment_name
        self.run_name = tracker_config.run_name
        self.tracking_uri = tracker_config.tracking_uri
        self.description = tracker_config.description
        self.run_id = None
        self.run = None
        self.score_names = tracker_config.score_names
        self.log_config = tracker_config.log_config

        # Setup MLflow for experiment tracking and logging
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        self.experiment_name = f"{getpass.getuser()}-ciq-search"
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(self.experiment_name)
        )

        logger.info(f"Using experiment ID: {self.experiment_id}")
        logger.info(f"Using experiment name: {self.experiment_name}")
        logger.info(f"Experiment URI: {self.tracking_uri}/#/experiments/{self.experiment_id}")

        if self.run_name is not None and self.run_id is not None:
            self.run_name = self.get_date_string()

    def setup(self, **kwargs):
        os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "1"

    def cleanup(self, **kwargs):
        pass

    def search_starts(self, **kwargs):
        if self.run_id is None:
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                description=self.description,
            )
            self.run_id = self.run.info.run_id
            mlflow.set_tags({"status": "running", "search_name": self.run_name, "type": "search"})
        else:
            self.run = mlflow.get_run(run_id=self.run_id)

    @logging_exception
    def search_ends(self, **kwargs):
        mlflow.set_tag("status", "finished")
        mlflow.end_run()

    def generation_starts(self, generation_number: int, **kwargs):
        pass

    def generation_ends(self, generation_number: int, **kwargs):
        pass

    @logging_exception
    def pre_objective(self, config: dict, **kwargs):
        if config != BASELINE_DNA:
            tag_type = "evaluation"
        else:
            tag_type = "baseline"
        mlflow.start_run(experiment_id=self.experiment_id, parent_run_id=self.run_id, nested=True)
        child_run = mlflow.active_run()
        child_run_name = child_run.info.run_name
        child_run_id = child_run.info.run_id
        mlflow.log_params(config)
        mlflow.set_tags(
            {
                "status": "running",
                "type": tag_type,
                "search_name": self.run_name,
            }
        )
        # If kwargs are passed, set them as tags, otherwise skip
        if kwargs:
            mlflow.set_tags(kwargs)

        if self.log_config:
            mlflow.log_dict(
                dictionary=config, artifact_file=f"{child_run_name}.json", run_id=child_run_id
            )

    @logging_exception
    def post_objective(self, scores, **kwargs):
        all_good = False
        if isinstance(scores, float) or isinstance(scores, int):
            all_good = True
            scores_list = [scores]
        elif isinstance(scores, list):
            all_good = all(x != INVALID_SCORE for x in scores)
            scores_list = scores
        else:
            all_good = False

        if all_good:
            metrics_dict = {}
            for i, score in enumerate(scores_list):
                metrics_dict[f"score_{i}"] = score
                mlflow.log_metrics(metrics_dict)
            mlflow.set_tag("status", "success")
        else:
            mlflow.set_tag("status", "failed")
        mlflow.end_run()


_TRACKER_TYPES_TO_CLASSES = {
    TrackerTypes.LOGURU: LoguruTracker,
    TrackerTypes.MLFLOW: MLflowTracker,
    TrackerTypes.DISABLED: DisabledTracker,
}

_TRACKER_TYPES_TO_CONFIG = {
    TrackerTypes.LOGURU: LoguruTrackerConfig,
    TrackerTypes.MLFLOW: MLflowTrackerConfig,
    TrackerTypes.DISABLED: DisabledTrackerConfig,
}

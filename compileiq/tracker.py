import functools
import json
import os
import sys
import warnings
from types import ModuleType
from typing import Any, Optional
from compileiq.types import (
    BASELINE_DNA,
    INVALID_SCORE,
    BaseTracker,
    ParamArg,
    TrackerTypes,
    DisabledTrackerConfig,
    LoguruTrackerConfig,
    MLflowTrackerConfig,
)
from loguru import logger

import getpass
from datetime import datetime

_mlflow: Optional[ModuleType]
try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None


def _resolution_metadata_to_mlflow_tags(metadata) -> dict[str, str]:
    if not metadata:
        return {}
    return {
        f"search_space.{idx}.{key}": str(value)
        for idx, item in enumerate(metadata)
        for key, value in item.items()
    }


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

    def __init__(self, tracker_config: Optional[DisabledTrackerConfig] = None):
        if tracker_config is None:
            tracker_config = DisabledTrackerConfig()
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

    def pre_objective(self, config: ParamArg, **kwargs):
        pass

    def post_objective(self, scores, **kwargs):
        pass


class LoguruTracker(BaseTracker):
    tracker_config: LoguruTrackerConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, tracker_config: LoguruTrackerConfig):
        super().__init__(tracker_config)

    def setup(self, **kwargs):
        logger.remove()
        raw = self.tracker_config.sink
        if raw is None:
            sinks: list = [sys.stdout]
        elif isinstance(raw, list):
            sinks = raw
        else:
            sinks = [raw]
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
        metadata = kwargs.get("search_space_resolution_metadata")
        if metadata:
            logger.info(f"Resolved search space: {json.dumps(metadata, sort_keys=True)}")

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
    def pre_objective(self, config: ParamArg, **kwargs):
        logger.bind(task_id=kwargs.get("task_id", "<missing-task-id>")).info(
            f"Executing objective function with: {config}"
        )

    @logging_exception
    def post_objective(self, scores, **kwargs):
        logger.bind(task_id=kwargs.get("task_id", "<missing-task-id>")).info(f"Scored: {scores}")


class MLflowTracker(BaseTracker):
    tracker_config: MLflowTrackerConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @staticmethod
    def get_date_string():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Default values are set on the MLflowTrackerConfig object definition
    def __init__(self, tracker_config: MLflowTrackerConfig):
        if _mlflow is None:
            raise RuntimeError("MLflowTracker requires mlflow. Install with `pip install mlflow`.")

        super().__init__(tracker_config)
        self.experiment_name = tracker_config.experiment_name
        self.run_name = tracker_config.run_name
        self.tracking_uri = tracker_config.tracking_uri
        self.description = tracker_config.description
        self.run_id: str | None = None
        self.run: Any = None
        self.score_names = tracker_config.score_names
        self.log_config = tracker_config.log_config

        # Setup MLflow for experiment tracking and logging
        if self.tracking_uri is not None:
            _mlflow.set_tracking_uri(self.tracking_uri)

        self.experiment_name = f"{getpass.getuser()}-ciq-search"
        experiment = _mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else _mlflow.create_experiment(self.experiment_name)
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
        assert _mlflow is not None, "MLflowTracker methods require mlflow"
        if self.run_id is None:
            self.run = _mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                description=self.description,
            )
            self.run_id = self.run.info.run_id
            _mlflow.set_tags({"status": "running", "search_name": self.run_name, "type": "search"})
        else:
            self.run = _mlflow.get_run(run_id=self.run_id)
        metadata = kwargs.get("search_space_resolution_metadata")
        if metadata:
            _mlflow.set_tags(_resolution_metadata_to_mlflow_tags(metadata))

    @logging_exception
    def search_ends(self, **kwargs):
        assert _mlflow is not None, "MLflowTracker methods require mlflow"
        _mlflow.set_tag("status", "finished")
        _mlflow.end_run()

    def generation_starts(self, generation_number: int, **kwargs):
        pass

    def generation_ends(self, generation_number: int, **kwargs):
        pass

    @logging_exception
    def pre_objective(self, config: ParamArg, **kwargs):
        assert _mlflow is not None, "MLflowTracker methods require mlflow"
        if isinstance(config, dict):
            flat_params = config
            structured = config
        elif isinstance(config, str):
            flat_params = {"dna_path": config}
            structured = flat_params
        else:
            flat_params = {}
            for i, entry in enumerate(config):
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        flat_params[f"config_{i}.{key}"] = value
                else:
                    flat_params[f"config_{i}.dna_path"] = entry
            structured = {f"config_{i}": entry for i, entry in enumerate(config)}

        tag_type = "baseline" if config == BASELINE_DNA else "evaluation"
        _mlflow.start_run(experiment_id=self.experiment_id, parent_run_id=self.run_id, nested=True)
        child_run = _mlflow.active_run()
        assert child_run is not None, "start_run() above guarantees an active run"
        child_run_name = child_run.info.run_name
        child_run_id = child_run.info.run_id
        _mlflow.log_params(flat_params)
        _mlflow.set_tags(
            {
                "status": "running",
                "type": tag_type,
                "search_name": self.run_name,
            }
        )
        # If kwargs are passed, set them as tags, otherwise skip
        if kwargs:
            _mlflow.set_tags(kwargs)

        if self.log_config:
            _mlflow.log_dict(
                dictionary=structured, artifact_file=f"{child_run_name}.json", run_id=child_run_id
            )

    @logging_exception
    def post_objective(self, scores, **kwargs):
        assert _mlflow is not None, "MLflowTracker methods require mlflow"
        all_good = False
        scores_list: list = []
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
                _mlflow.log_metrics(metrics_dict)
            _mlflow.set_tag("status", "success")
        else:
            _mlflow.set_tag("status", "failed")
        _mlflow.end_run()


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

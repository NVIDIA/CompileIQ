import tempfile
import pathlib
from compileiq.types import (
    TrackerTypes,
    LoguruTrackerConfig,
    MLflowTrackerConfig,
    SearchConfiguration,
    WorkerTypes,
)
from compileiq.ciq import Search
import compileiq.search_spaces.base as ss
import pytest
import os


def mock_objective(config):
    return 1.0


async def async_mock_objective(config):
    return 1.0


@pytest.fixture
def workers_to_test():
    """Helper function to get worker types to test."""
    return [WorkerTypes.NATIVE, WorkerTypes.RAY, WorkerTypes.ASYNC]


@pytest.mark.requires_ipc
def test_compileiq_default_tracker_type(
    mock_core_start,
    mock_send_to_core,
    mock_receive_from_core,
    mock_socket_listen,
):
    """Test that Search uses default tracker type when not specified."""
    search_space = {"x": ss.range(start=1.0, end=10.0, step=1.0)}

    with tempfile.TemporaryDirectory() as temp_dir:
        config = SearchConfiguration(
            pool_size=6,
            generations=1,
            mutate_rate=0.5,
        )

        search = Search(
            objective_function=mock_objective,
            search_space=search_space,
            search_config=config,
            cache_folder=pathlib.Path(temp_dir),
        )

        # Should default to DISABLED
        assert search.tracker_config.type == TrackerTypes.DEFAULT
        assert search.tracker_config.type == TrackerTypes.DISABLED

        search.start()


@pytest.mark.requires_ipc
def test_mlflow(
    workers_to_test,
    mock_core_start,
    mock_send_to_core,
    mock_receive_from_core,
    mock_socket_listen,
):
    """Test Search with MLflow tracker type."""
    search_space = {"x": ss.range(start=1.0, end=10.0, step=1.0)}
    config = SearchConfiguration(
        pool_size=6,
        generations=1,
        mutate_rate=0.5,
    )

    for wt in workers_to_test:
        obj_func = async_mock_objective if wt == WorkerTypes.ASYNC else mock_objective
        with tempfile.TemporaryDirectory() as temp_dir:
            # An empty tracking uri makes mlflow use a local server
            # TODO: maybe a better way to test is to use a local server
            # and check over the API that events have been logged there.
            test_tracking_uri = ""
            test_kwarg = {"test_kwarg": "test_value"}
            test_experiment_name = "test_experiment"
            test_score_names = ["test_score"]
            test_description = "test_description"
            search = Search(
                objective_function=obj_func,
                search_space=search_space,
                search_config=config,
                tracker_config=MLflowTrackerConfig(
                    experiment_name=test_experiment_name,
                    tracking_uri=test_tracking_uri,
                    log_config=False,
                    description=test_description,
                    score_names=test_score_names,
                    **test_kwarg,
                ),
                worker_type=wt,
                cache_folder=pathlib.Path(temp_dir),
            )

            search.start()


@pytest.mark.requires_ipc
def test_loguru(
    workers_to_test,
    mock_core_start,
    mock_send_to_core,
    mock_receive_from_core,
    mock_socket_listen,
):
    """Test that Search uses loguru tracker type when specified."""
    search_space = {"x": ss.range(start=1.0, end=10.0, step=1.0)}
    for wt in workers_to_test:
        obj_func = async_mock_objective if wt == WorkerTypes.ASYNC else mock_objective
        wt_name = wt.value
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SearchConfiguration(
                pool_size=12,
                generations=2,
                mutate_rate=0.5,
            )

            log_file = os.path.join(temp_dir, f"{wt_name}_test_search.log")

            tracker_config = LoguruTrackerConfig(level="DEBUG", sink=log_file)

            tuner = Search(
                objective_function=obj_func,
                search_space=search_space,
                search_config=config,
                tracker_config=tracker_config,
                worker_type=wt,
            )

            assert tuner.tracker_config.type == TrackerTypes.LOGURU

            tuner.start(num_cpus=1, num_gpus=0, scheduling_strategy="DEFAULT")

            # assert that the log file exists
            assert pathlib.Path(log_file).exists()

            # assert that the log file contains the expected content
            with open(log_file, "r") as f:
                content = f.read()  # Read once
                assert "Search started" in content
                assert "Search ended" in content

from typing import Callable, List, Dict, Optional
import inspect
import multiprocessing
from multiprocessing.connection import Connection, wait as mp_wait
from multiprocessing.synchronize import Event
import traceback
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from queue import Empty
from uuid import uuid4
import time
import ray
import json
import os
from ray._common.ray_option_utils import task_options
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)
import sys
from compileiq.utils.validation import SingleScore, MultiScore, Score
from compileiq.types import INVALID_SCORE, BASELINE_DNA, Worker, BaseTracker

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # This makes sure when using pyinstaller nothing breaks and the executable works properly
    multiprocessing.freeze_support()
else:
    mp_mode = os.environ.get("CIQ_PROCESS_MODE", "forkserver")
    if sys.platform == "win32":
        if "CIQ_PROCESS_MODE" in os.environ:
            warnings.warn(
                "Fork multiprocessing is not available on Windows. Switching to spawn mode."
                "Set the environment variable `CIQ_PROCESS_MODE` to 'spawn' to avoid this warning.",
                RuntimeWarning,
            )
        mp_mode = "spawn"
    if mp_mode not in multiprocessing.get_all_start_methods():
        raise ValueError(
            f"Invalid multiprocessing start method: {mp_mode}. "
            f"Available methods are: {multiprocessing.get_all_start_methods()}"
        )
    multiprocessing.set_start_method(mp_mode, force=True)


class MultiProcessWorker(Worker):
    """
    Worker that uses only existing python primitives.
    Deployes a fixed amount of processes to run the objective functions.
    Does not have distributed machine support, only local parallelism
    """

    def __init__(self, cache_folder: str, normalize: bool = False, tracker: BaseTracker = None):
        super().__init__(
            cache_folder=cache_folder,
            normalize=normalize,
            tracker=tracker,
            respects_num_workers=True,
            supports_timeout=True,
        )
        try:
            self.manager = multiprocessing.Manager()
        except EOFError as e:
            raise RuntimeError(
                "Something went wrong when instantiating a manager "
                "for python's multiprocessing lib. Did you perhaps "
                "forget to put the code in `if __name__ == '__main__'`?"
            ) from e

    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def __del__(self):
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def run(
        self,
        *,
        function: Callable,
        params_pool: List[Dict | str],
        params_ids: List[int],
        num_function_returns: int = 1,
        num_workers: int = 1,
        task_timeout: Optional[int | float] = None,
        **kwargs,
    ) -> List[Score]:
        """
        Executes the objective function (`function`) for every parameter in `params_pool`.
        It levereges python multiprocess and a job/result queues where
        subprocesses pick up additional work.

        The return is a list with the return of `function` for every param inside `param_pool`.
        If normalization is enabled, we normalize the scores.

        Args:
            function:
                The user-defined objective function to be executed with CompileIQ sampled parameters
            params_pool:
                A CompileIQ generated pool of parameters to be used as input for the objective
                function. The parameters will respect the search space defined by the user.
            num_function_returns:
                The number of returns expected from the objective function.
            num_workers:
                The number of parallel worker processes to use for executing the objective function.
            task_timeout:
                The timeout in seconds for objective function execution in each process.
                If the objective takes longer, it returns `INVALID_SCORE`.

        Returns:
            A list of `Score`. There will be a return for each param in `params_pool`.
        """
        scores_queue = self.manager.Queue()
        job_queue = self.manager.Queue()
        error_queue = self.manager.Queue()

        # Adding baseline measurement (only for the first run) at the
        # start of the queue.
        if self.normalize and self.baseline_score is None:
            job_queue.put(("baseline", BASELINE_DNA))

        for i, param in enumerate(params_pool):
            job_queue.put((params_ids[i], param))

        wrapper_args = (
            job_queue,
            scores_queue,
            num_function_returns,
            function,
            self.tracker,
            self.normalize,
            error_queue,
            task_timeout,
        )
        # Defining processes job
        process_pool = [
            multiprocessing.Process(target=MultiProcessWorker._function_wrapper, args=wrapper_args)
            for _ in range(min(num_workers - 1, len(params_pool)))
        ]

        # Starting Processes
        for p in process_pool:
            p.start()

        # Have the parent process also compute
        failed = False
        try:
            MultiProcessWorker._function_wrapper(*wrapper_args)
        except Exception:
            failed = True

        # Waiting for remaining processes
        for p in process_pool:
            if failed:
                p.kill()
            else:
                p.join()
                if p.exitcode == 1:
                    failed = True

        if failed:
            errors = []
            while not error_queue.empty():
                errors.append(error_queue.get())
            if errors:
                raise RuntimeError(
                    "One or more worker processes failed:\n" + "\n---\n".join(errors)
                )
            else:
                raise RuntimeError(
                    "A worker process failed with no captured traceback. "
                    f"Make sure the declared `num_objectives`({num_function_returns}) "
                    "matches the number of returns from your objective function and "
                    "their values are of type int, float, or '*' (for invalid scores). "
                    "If normalization is enabled, make sure your function supports baseline "
                    "measurements."
                )

        # Retrieving scores from the queue
        resulting_scores: List[Score] = []
        while not scores_queue.empty():
            score: Score = scores_queue.get()
            if score.is_baseline:
                self.baseline_score = score

            resulting_scores.append(score)

        # Normalizing all scores if enabled
        if self.normalize:
            for score in resulting_scores:
                if score.failed:
                    score.norm_score = (
                        INVALID_SCORE
                        if num_function_returns == 1
                        else [INVALID_SCORE] * num_function_returns
                    )
                else:
                    score.norm_score = self.normalize_scores(score.score, self.baseline_score.score)

        return resulting_scores

    @staticmethod
    def execute_objective(
        obj_func_args: Dict | str,
        param_id: int | str,
        num_objectives: int,
        objective_func: Callable,
        tracker: BaseTracker,
        norm_enabled: bool = False,
    ) -> Score:
        """
        Executes the user objective function, but handles exceptions, score validation.
        """
        try:
            task_id = str(uuid4().hex)
            if tracker is not None:
                tracker.pre_objective(obj_func_args, task_id=task_id)
            scores = objective_func(obj_func_args)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            warnings.warn(
                "Your objective function encountered an exception. "
                f"Score will be set to {INVALID_SCORE}",
                RuntimeWarning,
            )
            scores = Worker.invalidate_score(num_objectives)

        score = Score(
            score=scores,
            metadata=json.dumps({"pid": os.getpid()}),
            params=obj_func_args,
            param_id=param_id,
            num_objectives=num_objectives,
            is_baseline=(param_id == "baseline"),
        )

        if tracker is not None:
            tracker.post_objective(score.model_dump_json(), task_id=task_id)

        # Baseline is not allowed to fail
        if norm_enabled and score.is_baseline and score.failed:
            raise RuntimeError(
                "Baseline measurement for normalization failed. Make sure your function "
                "supports baseline measurements for normalization or "
                "set normalize=False to avoid score normalization."
            )

        return score

    @staticmethod
    def _function_wrapper(
        job_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        num_objectives: int,
        objective_func: Callable,
        tracker: BaseTracker,
        norm_enabled: bool = False,
        error_queue: Optional[multiprocessing.Queue] = None,
        task_timeout: Optional[int | float] = None,
    ) -> None:
        """
        Executes the user objective function, but handles exceptions, score validation,
        and queue management to increase process utilization.

        All returns are put in separate queue.
        """
        try:
            while not job_queue.empty():
                try:
                    param_id, obj_func_args = job_queue.get(block=False)
                except Empty:
                    break

                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    MultiProcessWorker.execute_objective,
                    obj_func_args=obj_func_args,
                    param_id=param_id,
                    num_objectives=num_objectives,
                    objective_func=objective_func,
                    tracker=tracker,
                    norm_enabled=norm_enabled,
                )
                try:
                    score = future.result(timeout=task_timeout)
                except FuturesTimeoutError:
                    warnings.warn(
                        f"Objective function timed out after {task_timeout} seconds. "
                        f"Score will be set to {INVALID_SCORE}",
                        RuntimeWarning,
                    )
                    score = Score(
                        score=Worker.invalidate_score(num_objectives),
                        metadata=json.dumps({"pid": os.getpid()}),
                        params=obj_func_args,
                        param_id=param_id,
                        num_objectives=num_objectives,
                        is_baseline=(param_id == "baseline"),
                    )
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                results_queue.put(score)

        except Exception:
            if error_queue is not None:
                error_queue.put(traceback.format_exc())
            raise


class IsoMultiProcessWorker(Worker):
    """
    Similar to MultiProcess worker but does not reuse processes for different tasks.
    This means that each task will be executed in a completely isolated process.
    This can be useful for workloads that often hang or cause leaks,
    needing to be killed (through timeout or other mechanisms).
    """

    def __init__(self, cache_folder: str, normalize: bool = False, tracker: BaseTracker = None):
        super().__init__(
            cache_folder=cache_folder,
            normalize=normalize,
            tracker=tracker,
            respects_num_workers=True,
            supports_timeout=True,
        )

    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(
        self,
        *,
        function: Callable,
        params_pool: List[Dict | str],
        params_ids: List[int],
        num_function_returns: int = 1,
        num_workers: int = 1,
        task_timeout: Optional[int | float] = None,
        **kwargs,
    ) -> List[Score]:
        """
        Executes the objective function (`function`) for every parameter in `params_pool`.
        Dispatches a new process per task and polls active processes to respect `num_workers`,
        without threads or a semaphore — safe to use with the `fork` start method.

        Args:
            function:
                The user-defined objective function to be executed with CompileIQ sampled parameters
            params_pool:
                A CompileIQ generated pool of parameters to be used as input for the objective
                function. The parameters will respect the search space defined by the user.
            num_function_returns:
                The number of returns expected from the objective function.
            num_workers:
                The number of parallel worker processes to use for executing the objective function.
            task_timeout:
                The timeout in seconds for each process. If a process takes longer than this to
                execute, it will be killed and return `INVALID_SCORE`.

        Returns:
            A list of `Score`. There will be a return for each param in `params_pool`.
        """
        tasks = []
        if self.normalize and self.baseline_score is None:
            tasks.append(("baseline", BASELINE_DNA))
        for i, param in enumerate(params_pool):
            tasks.append((params_ids[i], param))

        resulting_scores = self.handle_isolation(
            tasks, num_workers, function, num_function_returns, task_timeout
        )

        for score in resulting_scores:
            if score.is_baseline:
                self.baseline_score = score

        if self.normalize:
            for score in resulting_scores:
                if score.failed:
                    score.norm_score = Worker.invalidate_score(num_function_returns)
                else:
                    score.norm_score = self.normalize_scores(score.score, self.baseline_score.score)

        return resulting_scores

    @staticmethod
    def execute_objective(
        obj_func_args: Dict | str,
        param_id: int | str,
        num_objectives: int,
        objective_func: Callable,
        tracker: BaseTracker,
        norm_enabled: bool,
        result_pipe: Connection,
        done_event: Event,
    ) -> None:
        """
        Executes the user objective function, but handles exceptions, score validation,
        and pipe management to return the result to the parent process.
        """
        if tracker is not None:
            tracker.setup()

        score = MultiProcessWorker.execute_objective(
            obj_func_args, param_id, num_objectives, objective_func, tracker, norm_enabled
        )

        if result_pipe is not None:
            # Signal that the objective is done and we are about to write to the pipe.
            # The parent uses this to distinguish a hung objective from a process that
            # finished and is simply serialising the result.
            if done_event is not None:
                done_event.set()

            result_pipe.send(score)

    def handle_isolation(
        self,
        tasks: list[tuple[int | str, Dict | str]],
        num_workers: int,
        function: callable,
        num_function_returns: int,
        task_timeout: Optional[int | float] = None,
    ) -> List[Score]:
        """
        Handles the creation of isolated processes, their timeouts and score collection
        """

        # (process, recv_conn, param_id, param, start_time, done_event)
        active: list[
            tuple[multiprocessing.Process, Connection, int | str, Dict | str, float, Event]
        ] = []
        resulting_scores: List[Score] = []
        task_index = 0

        # Default to 'fork' mode for this worker otherwise things will be super slow
        # User env var still takes precedence if set.
        mode = os.environ.get("CIQ_PROCESS_MODE", "fork")
        iso_ctx = multiprocessing.get_context(mode if sys.platform != "win32" else "spawn")

        # This loops handles timeouts and keeping the number of workers under the defined limit
        # A simpler implementation with Threading will cause issues with mode `fork` and Pool/map
        # approaches cannot handle process kills the way we need.
        try:
            while task_index < len(tasks) or active:
                # Launch new processes up to num_workers
                while len(active) < min(len(tasks), num_workers) and task_index < len(tasks):
                    param_id, param = tasks[task_index]
                    task_index += 1

                    # Creating pipe to receive the score back (lightweight compared to Queue)
                    recv_conn, send_conn = iso_ctx.Pipe(duplex=False)
                    done_event = iso_ctx.Event()
                    p = iso_ctx.Process(
                        target=IsoMultiProcessWorker.execute_objective,
                        args=(
                            param,
                            param_id,
                            num_function_returns,
                            function,
                            self.tracker,
                            self.normalize,
                            send_conn,
                            done_event,
                        ),
                    )
                    p.start()
                    send_conn.close()
                    active.append((p, recv_conn, param_id, param, time.monotonic(), done_event))

                # Check each running process
                still_active = []
                for p, recv_conn, param_id, param, start_time, done_event in active:
                    failed = False

                    if p.is_alive():
                        # If timed out and not writing to the pipe already
                        # kill the process and return invalid score.
                        if (
                            task_timeout is not None
                            and time.monotonic() - start_time >= task_timeout
                        ) and not done_event.is_set():
                            p.kill()
                            p.join()
                            recv_conn.close()
                            warnings.warn(
                                f"Task {param_id} timed out after {task_timeout}s. "
                                f"Score set to {INVALID_SCORE}",
                                RuntimeWarning,
                            )
                            failed = True
                    else:
                        p.join()
                        if p.exitcode != 0:
                            recv_conn.close()
                            failed = True  # for completeness
                            raise RuntimeError(
                                f"Task {param_id} failed with exit code {p.exitcode}. "
                                "Make sure the declared num_objectives matches the number of "
                                "returns from your objective function. If normalization is "
                                "enabled, make sure your function supports BASELINE_DNA "
                            )
                        elif not done_event.is_set():
                            # Process exited cleanly without signalling — should not happen
                            recv_conn.close()
                            raise RuntimeError(
                                f"Task {param_id} exited with code 0 but never signalled completion"
                            )

                    if failed:
                        resulting_scores.append(
                            Score(
                                score=Worker.invalidate_score(num_function_returns),
                                metadata=json.dumps({"pid": p.pid}),
                                params=param,
                                param_id=param_id,
                                num_objectives=num_function_returns,
                                is_baseline=(param_id == "baseline"),
                            )
                        )
                    elif done_event.is_set():
                        try:
                            score = recv_conn.recv()
                        except EOFError:
                            score = Score(
                                score=Worker.invalidate_score(num_function_returns),
                                metadata=json.dumps({"pid": p.pid}),
                                params=param,
                                param_id=param_id,
                                num_objectives=num_function_returns,
                                is_baseline=(param_id == "baseline"),
                            )
                        finally:
                            recv_conn.close()
                            p.join()

                        if score.is_baseline and score.failed:
                            raise RuntimeError(
                                "Baseline measurement for normalization failed at "
                                f" task {param_id}.Make sure your function supports "
                                "baseline measurements for normalization or set "
                                "normalize=False to avoid score normalization."
                            )

                        resulting_scores.append(score)
                    else:
                        # If the process is not done with objective and is not tagged as failed
                        # it is still active
                        still_active.append((p, recv_conn, param_id, param, start_time, done_event))

                active = still_active
                if active:
                    # mp_wait will unblock immediately when pipe is ready
                    # the timeouts are for cases where the processes are hanging forever
                    mp_wait([recv for _, recv, _, _, _, _ in active], timeout=0.05)

        finally:
            # Cleanup any remaining processes in case of unexpected errors
            for p, recv_conn, _, _, _, _ in active:
                if p.is_alive():
                    p.kill()
                p.join()
                if not recv_conn.closed:
                    recv_conn.close()

        return resulting_scores


# TODO: Add Thread Worker for real threads (3.14+)


class RayWorker(Worker):
    def __init__(self, cache_folder: str, normalize: bool = False, tracker: BaseTracker = None):
        super().__init__(
            cache_folder=cache_folder,
            normalize=normalize,
            tracker=tracker,
            respects_num_workers=False,
            # Ray does not support task-level timeout.
            # It doesn't consider if the task started or is in the queue
            # https://github.com/ray-project/ray/issues/18916
            supports_timeout=False,
        )

    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(
        self,
        *,
        function: Callable,
        params_pool: List[Dict | str],
        params_ids: List[int],
        num_function_returns: int = 1,
        **ray_resources,
    ) -> List[Score]:
        """
        Executes the objective function (`function`) for every parameter in `params_pool`.
        It levereges ray to parallelize/distribute computation.
        Values passed down to ray_resources are from:
            https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#specifying-task-or-actor-resource-requirements

        The return is a list with the return of `function` for every param inside `param_pool`.
        If normalization is enabled, it will return the baseline when measured.

        Args:
            function:
                The user-defined objective function to be executed with CompileIQ sampled parameters
            params_pool:
                A CompileIQ generated pool of parameters to be used as input for the objective
                function. The parameters will respect the search space defined by the user.

            num_function_returns:
                The number of returns expected from the objective function.
            **ray_resources:
                Additional resources to be passed down to ray when scheduling tasks.

        Returns:
            A list of `Score`. There will be a return for each param in `params_pool`.

        """

        valid_options = set(task_options.keys()).intersection(ray_resources.keys())
        ray_resources = {key: ray_resources[key] for key in valid_options}

        # Execute baseline measurements on all ray nodes (if needed)``
        # This measurement is done once per node per GPU
        measured_baselines = None
        if self.normalize:
            measured_baselines = self.handle_baseline(
                ray_resources.copy(), function, num_function_returns
            )

        # Dispatching work for execution on Ray Cluster
        futures = [
            self._function_wrapper.options(**ray_resources).remote(
                function,
                params,
                params_ids[i],
                num_function_returns,
                baseline_scores=self.baseline_score,
                tracker=self.tracker,
            )
            for i, params in enumerate(params_pool)
        ]
        # Wait for scores to be ready (Scores will come already normalized)
        scores = ray.get(futures)

        if measured_baselines is not None:
            scores = measured_baselines + scores

        return scores

    def handle_baseline(
        self, user_ray_resources: Dict, function: Callable, num_function_returns: int
    ) -> List[Score] | None:
        """
        This function will handle baselining when normalization is enabled.
        It will execute the baseline measurement on all ray nodes that do not have a baseline yet.
        Because we are often baselining gpu environments, we make sure to have one baseline per each
        gpu available in the node.

        If the tuple (node_id, gpu_id) already has a valid baseline measured, we skip measuring
        it again.

        Args:
            user_ray_resources:
                The ray resources defined by the user to be used when scheduling tasks.
            function:
                The user-defined objective function to be executed with CompileIQ sampled parameters
            num_function_returns:
                The number of returns expected from the objective function.

        Returns:
            The measured baseline scores for this round.
        """
        # We need to use another scheduling strategy then the user's for baseline
        if "scheduling_strategy" in user_ray_resources:
            del user_ray_resources["scheduling_strategy"]

        if self.baseline_score is None:
            self.baseline_score = {}

        # We need to perform init to call ray.nodes() but do not want to overwrite
        if not ray.is_initialized():
            ray.init(_temp_dir=os.environ.get("RAY_TMPDIR"))

        cluster_data = ray.nodes()
        requested_gpus = user_ray_resources.get("num_gpus", 0)
        futures, placement_groups = [], []
        measured_baselines = None

        if requested_gpus > 1:
            raise ValueError(
                "You have requested more than 1 GPU per task with `normalize=True`. "
                "RayWorker is not prepared to perform normalization under this use case."
            )

        if requested_gpus > 0:
            for node in cluster_data:
                gpus_in_node = int(node["Resources"].get("GPU", 0))
                nid, is_alive = node["NodeID"], node["Alive"]
                should_measure_baseline = nid not in self.baseline_score or any(
                    [bs.failed for bs in self.baseline_score[nid].values()]
                )
                # Deciding if we need to measure baseline for this node, i.e.
                # It has available gpu resources, is alive and does not have a valid baseline yet
                if gpus_in_node > 0 and is_alive and should_measure_baseline:
                    self.baseline_score[nid] = {}
                    bundles = [
                        {"CPU": user_ray_resources.get("num_cpus", 1), "GPU": 1}
                    ] * gpus_in_node

                    # Making sure we have one placement group for each gpu available inside the node
                    pg = ray.util.placement_group(
                        bundles, strategy="STRICT_PACK", _soft_target_node_id=nid
                    )
                    placement_groups.append(pg)

            # Dispatching baseline measurements into placement groups
            if len(placement_groups) > 0:
                try:
                    # Waiting for placement groups
                    for pg in placement_groups:
                        ray.get(pg.ready(), timeout=5)
                except Exception:
                    print(
                        "Cannot create a placement group because "
                        "{'GPU': 1} bundle cannot be created."
                        "Try disabling normalization with `normalize=False`.",
                        file=sys.stderr,
                    )
                    for pg in placement_groups:
                        ray.util.remove_placement_group(pg)

                for pg in placement_groups:
                    for bundle_index in range(pg.bundle_count):
                        future = self._function_wrapper.options(
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg, placement_group_bundle_index=bundle_index
                            ),
                            **user_ray_resources,
                        ).remote(
                            function,
                            BASELINE_DNA,
                            "baseline",
                            num_function_returns,
                            tracker=self.tracker,
                        )
                        futures.append(future)

        else:
            # We are not requesting gpus, so we baseline once per node.
            for node in cluster_data:
                nid, is_alive, has_cpu = (
                    node["NodeID"],
                    node["Alive"],
                    node["Resources"].get("CPU", 0) > 0,
                )
                # Baseline only in nodes that do not have a baseline yet, are alive and
                # have cpu resources
                if nid not in self.baseline_score and is_alive and has_cpu:
                    future = self._function_wrapper.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
                        **user_ray_resources,
                    ).remote(
                        function,
                        BASELINE_DNA,
                        "baseline",
                        num_function_returns,
                        tracker=self.tracker,
                    )
                    futures.append(future)

        if len(futures) > 0:
            # Retrieving baseline scores
            measured_baselines: List[Score] = ray.get(futures)
            for baseline_score in measured_baselines:
                metadata = json.loads(baseline_score.metadata)
                nid, gpu_id = metadata["node_id"], metadata["gpu_id"]

                # Using Dict to store baselines per gpu (if requested) and per node
                if requested_gpus > 0:
                    self.baseline_score[nid][gpu_id] = baseline_score
                else:
                    self.baseline_score[nid] = baseline_score

        # Clean placement groups created to unconstrain non-baseline runs
        for pg in placement_groups:
            ray.util.remove_placement_group(pg)

        # We are only returning the baselines measured this round
        return measured_baselines

    @staticmethod
    @ray.remote
    def _function_wrapper(
        objective_func: Callable,
        obj_func_args: Dict | str,
        param_id: int | str,
        num_objectives: int,
        baseline_scores: Optional[Dict[str, Dict[str, Score] | Score]] = None,
        tracker: BaseTracker = None,
    ) -> Score:
        """
        Executes the user objective function, but handles exceptions and score validation.
        This function returns its score normalized if baseline_scores are not None
        """

        ray_context = ray.get_runtime_context()
        task_id, node_id = ray_context.get_task_id(), ray_context.get_node_id()
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        metadata = {"gpu_id": gpu_ids, "task_id": task_id, "node_id": node_id}
        # Reinitialize tracker in because we are now in a different process/machine
        if tracker is not None:
            tracker.setup()

        try:
            if tracker is not None:
                tracker.pre_objective(obj_func_args, **metadata)

            scores = objective_func(obj_func_args)

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            warnings.warn(
                "An Exception was caught during the execution of your objective function. "
                f"Score will be set to {INVALID_SCORE}",
                RuntimeWarning,
            )
            scores = Worker.invalidate_score(num_objectives)

        scores = Score(
            score=scores,
            metadata=json.dumps(metadata),
            params=obj_func_args,
            param_id=param_id,
            num_objectives=num_objectives,
            is_baseline=(param_id == "baseline"),
        )

        if scores.is_baseline:
            norm_score = Worker.normalize_scores(scores.score, scores.score)
            scores.norm_score = norm_score  # Baseline normalizes to 1.0

        # Normalizing the scores if enabled
        try:
            should_normalize = baseline_scores is not None
            if should_normalize and not scores.failed:
                # This is the node executing this function
                baseline_score: Score | Dict = baseline_scores.get(node_id, None)
                if isinstance(baseline_score, dict):
                    baseline_score: Score = baseline_score.get(gpu_ids, None)
                if baseline_score is None or baseline_score.failed:
                    # Because ray is dynamic we can have nodes come in and out at any time
                    # this works as a failsafe to still normalize in case we are at one of
                    # the nodes that joined after we did the initial baseline measurements
                    current_baseline = objective_func(BASELINE_DNA)
                    current_baseline = Score(
                        score=current_baseline,
                        params=BASELINE_DNA,
                        metadata="",
                        param_id="baseline",
                        norm_score=None,
                        num_objectives=num_objectives,
                    )
                    if current_baseline.failed:
                        # Baseline measurements cannot fail or will invalidate this score
                        raise RuntimeError(
                            f"Baseline measurement for normalization failed at node {node_id} "
                            "make sure your function supports baseline measurements for "
                            "normalization or set normalize=False to avoid score normalization."
                        )
                    else:
                        scores.norm_score = Worker.normalize_scores(
                            scores.score, current_baseline.score
                        )
                else:
                    # This should be the most common case, we pre-calculated the baseline
                    # and are just querying it here
                    scores.norm_score = Worker.normalize_scores(scores.score, baseline_score.score)
            elif should_normalize and scores.failed:
                scores.norm_score = scores.score
        except Exception as e:
            warnings.warn(
                "An Exception occured while trying to normalize your scores. "
                f"Score will be set to {INVALID_SCORE}. "
                f"Exception: {e}",
                RuntimeWarning,
            )
            scores.norm_score = Worker.invalidate_score(num_objectives)

        if tracker is not None:
            tracker.post_objective(scores.model_dump_json())

        return scores


class AsyncWorker(Worker):
    def __init__(self, cache_folder: str, normalize: bool = False, tracker: BaseTracker = None):
        super().__init__(
            cache_folder=cache_folder,
            normalize=normalize,
            tracker=tracker,
            respects_num_workers=False,
            supports_timeout=True,
        )

    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(
        self,
        *,
        function: Callable,
        params_pool: List[Dict | str],
        params_ids: List[int],
        num_function_returns: int = 1,
        task_timeout: Optional[int | float] = None,
        **kwargs,
    ) -> List[SingleScore | MultiScore]:
        """
        Leverages Python Async execution to improve concurrency from objective function.
        Args:
            function:
                The user-defined objective function to be executed with CompileIQ sampled parameters
                It must be `async` and leverage existing async libraries to improve concurrency.
            params_pool:
                A CompileIQ generated pool of parameters to be used as input for the objective
                function. The parameters will respect the search space defined by the user.
            num_function_returns:
                The number of returns expected from the objective function.

        Returns:
            A list of `Score`. There will be a return for each param in `params_pool`.
        """

        if not inspect.iscoroutinefunction(function):
            raise TypeError(
                "AsyncWorker requires an async objective function (declared with `async def`)."
            )

        return asyncio.run(
            self.arun(
                function=function,
                params_pool=params_pool,
                params_ids=params_ids,
                num_function_returns=num_function_returns,
                task_timeout=task_timeout,
                **kwargs,
            )
        )

    async def arun(
        self,
        function: Callable,
        params_pool: List[Dict | str],
        params_ids: List[int],
        num_function_returns: int,
        task_timeout: Optional[int | float] = None,
        **kwargs,
    ) -> List[SingleScore | MultiScore]:
        # Adding baseline measurement at the beggining
        if self.normalize and self.baseline_score is None:
            params_pool = [BASELINE_DNA] + params_pool

        futures = []
        async with asyncio.TaskGroup() as tg:
            for params in params_pool:
                task = tg.create_task(
                    self._function_wrapper(
                        obj_func_args=params,
                        objective_func=function,
                        num_objectives=num_function_returns,
                        tracker=self.tracker,
                        norm_enabled=self.normalize,
                        task_timeout=task_timeout,
                    )
                )
                futures.append(task)

        results = []
        if self.baseline_score is None and self.normalize:
            # Removing baseline future from the list
            baseline_score: Score = futures.pop(0).result()
            baseline_score.param_id = "baseline"
            baseline_score.is_baseline = True
            baseline_score.norm_score = self.normalize_scores(
                baseline_score.score, baseline_score.score
            )
            self.baseline_score = baseline_score
            results.append(self.baseline_score)

        for i, f in enumerate(futures):
            score: Score = f.result()
            score.param_id = params_ids[i]

            if self.normalize:
                if score.failed:
                    score.norm_score = (
                        INVALID_SCORE
                        if num_function_returns == 1
                        else [INVALID_SCORE] * num_function_returns
                    )
                else:
                    # This works because the first score will always be the baseline
                    score.norm_score = self.normalize_scores(score.score, self.baseline_score.score)

            results.append(score)

        return results

    @staticmethod
    async def _function_wrapper(
        obj_func_args: Dict[str, any] | str,
        objective_func: Callable,
        num_objectives: int,
        tracker: BaseTracker,
        norm_enabled: bool = False,
        task_timeout: Optional[int | float] = None,
    ) -> Score:
        """
        Executes the user objective function, but handles exceptions
        and score validation.

        `norm_enabled` safeguards a corner case where normalization is
        disabled, but knockout causes CompileIQ to produce BASELINE_DNA ({})
        """

        task_id = str(uuid4().hex)
        if tracker is not None:
            tracker.pre_objective(obj_func_args, task_id=task_id)

        invalid_score = Worker.invalidate_score(num_objectives)
        try:
            if task_timeout is not None:
                scores = await asyncio.wait_for(objective_func(obj_func_args), timeout=task_timeout)
            else:
                scores = await objective_func(obj_func_args)
        except asyncio.TimeoutError:
            warnings.warn(
                f"Objective function timed out after {task_timeout} seconds. "
                f"Score will be set to {INVALID_SCORE}",
                RuntimeWarning,
            )
            scores = invalid_score
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            warnings.warn(
                "An Exception was caught during the execution of your objective function. "
                f"Score will be set to {INVALID_SCORE}",
                RuntimeWarning,
            )
            scores = invalid_score

        scores = Score(
            score=scores,
            metadata=json.dumps({"task_id": task_id}),
            params=obj_func_args,
            param_id="",  # this will be filled out externally
            num_objectives=num_objectives,
        )
        if tracker is not None:
            tracker.post_objective(scores.model_dump_json(), task_id=task_id)

        # Baseline is not allowed to fail
        if norm_enabled and scores.is_baseline and scores.failed:
            raise RuntimeError(
                "Baseline measurement for normalization failed make sure your function"
                "supports baseline measurements for normalization or "
                "set normalize=False to avoid score normalization."
            )

        return scores

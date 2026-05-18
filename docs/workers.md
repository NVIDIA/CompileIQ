# Choosing how your application executes

CompileIQ supports parallel and distributed workload execution through different worker classes. You pass the worker class directly when creating a `Search`:

```python
from compileiq.worker import MultiProcessWorker

tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=MultiProcessWorker,
)
```

Built-in worker classes are available in `compileiq.worker`: `MultiProcessWorker` (default), `IsoMultiProcessWorker`, `RayWorker`, and `AsyncWorker`.

## Native Worker (MultiProcessWorker)

The native worker is the default. It uses Python's native `multiprocessing` library to spin up processes that execute your objective function.

> Warning: It does not support distributed machines; it only parallelizes task execution locally.

You specify the number of processes using the keyword `num_workers` when starting the search. Example:

> Note: CompileIQ defaults to the `forkserver` start method for multiprocessing (configurable via `CIQ_PROCESS_MODE`). When running from a script, keep your entry point under `if __name__ == "__main__":` to avoid issues on some platforms.

```python
from compileiq.worker import MultiProcessWorker

tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=MultiProcessWorker,
)
tuner.start(num_workers=15)
```

## Isolated Worker (IsoMultiProcessWorker)

Like `MultiProcessWorker`, but spawns one fresh process per task instead of reusing a pool. Useful when your objective function tends to hang, leak resources, or otherwise needs to be killed cleanly on timeout — each task is fully isolated, and the parent enforces `task_timeout` by killing the process.

```python
from compileiq.worker import IsoMultiProcessWorker

tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=IsoMultiProcessWorker,
)
tuner.start(num_workers=15, task_timeout=30)
```

> Note: `IsoMultiProcessWorker` defaults to the `fork` start method (overridable via `CIQ_PROCESS_MODE`).

## Async Worker

The async worker leverages Python concurrency through async calls. It is your responsibility to [enable asynchronous capabilities](https://docs.python.org/3/library/asyncio.html) inside the asynchronous function; otherwise, the worker will not take advantage of any concurrency and will execute sequentially.

```python
from compileiq.worker import AsyncWorker

async def objective(config):
    ...

tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=AsyncWorker,
)
tuner.start()
```

> Note: The objective function must be declared as an `async def`.

## Ray Worker

A Ray worker type leverages a Ray cluster to parallelize your objective function locally or in a distributed environment. It is the user's responsibility to configure the cluster and set the correct resources for the task.

We recommend reading Ray's [official documentation](https://docs.ray.io/en/latest/cluster/getting-started.html) if you are interested in performing a distributed search.

> [On-premise deployment](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html) is specifically useful when you have full SSH access to all machines.

All Ray workers must have CompileIQ installed, as well as any other required libraries for your run.
All Ray workers must have CompileIQ installed, as well as any other required libraries for your run.

You also need to run the script that starts the search on a machine that is attached to the cluster (a head or worker).

You can leverage the Ray worker as follows:

```python
from compileiq.worker import RayWorker

tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=RayWorker,
)
tuner.start()
```

> If your objective is to use Ray but run locally, you don't need to set up a cluster; the code runs as shown above.

`tuner.start()` accepts Ray task options (for example, `num_cpus`, `num_gpus`, `resources`, `scheduling_strategy`). These should reflect how many resources a single run of your objective function consumes; this impacts how Ray schedules workers for your execution. Unrecognized options are ignored.

Take a look at all [resource options available for Ray](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html#ray.remote_function.RemoteFunction.options).

> Tip: For a distributed cluster, initialize Ray before calling `tuner.start()` (for example, `ray.init(address="auto")`).

### Tips when using Ray

* Ensure all cluster machines have the necessary environment to run your function (CompileIQ installed plus any other required libraries).
* Ensure all cluster machines have the necessary environment to run your function (CompileIQ installed plus any other required libraries).
* A Ray cluster requires a head node, which also executes tasks and may negatively impact sensitive measurements. You can prevent the head node from picking up tasks by starting it with `ray start --head --num-cpus=0`.
* Specifying `num_cpus` in `tuner.start()` does not pin or limit your function to the specified core. Please refer to [Ray Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.htm) for more details.
* You can find a CompileIQ with Ray [example in our repository](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/distributed.py).
* You can deploy your entire cluster with a `.yml` file using the [Ray Cluster Launcher](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#start-ray-with-the-ray-cluster-launcher).

## Handling a Multi-GPU Environment

Ray is the only supported worker that automatically assigns/schedules GPU IDs to your objective function. Therefore, if you have a multi-machine and/or multi-GPU environment, you must use a `RayWorker` or create your own custom worker.

Ray will automatically assign nodes and GPU resources based on the parameters passed through `.start()`. To know which GPU was assigned to the current task/objective, you can check the `CUDA_VISIBLE_DEVICES` environment variable or use `ray.get_gpu_ids()`.

> Note that it is the user's responsibility to retrieve the GPU IDs and configure execution so this is the only GPU being used. Some existing applications will already leverage `CUDA_VISIBLE_DEVICES` and set things behind the scenes, while others will require you to explicitly specify the ID through a different API.

Below is an example in a Ray Cluster with a head node without GPUs and a single worker with 2 GPUs:

```python
def objective(config):
    print(f"My Visible devices are: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    time.sleep(5)

    return 0

...

tuner.start(num_gpus=1)
```

Output:

```bash
2025-01-27 20:01:18,507 INFO worker.py:1636 -- Connecting to existing Ray cluster at address: xxxxxx:6379...
2025-01-27 20:01:18,518 INFO worker.py:1812 -- Connected to Ray cluster. View the dashboard at http://127.0.0.1:8265
(objective pid=32341, ip=xxxxx) My Visible devices are: 0
(objective pid=32774, ip=xxxxx) My Visible devices are: 1
```

Notice how each objective function was assigned a different GPU ID.

## Create your own Custom Worker

Although the existing workers cover the most common cases, CompileIQ also offers a way for the user to extend the existing worker infrastructure and create something custom.

A custom worker can help you parallelize your workload, save states between generations, and have more fine-grained control over logging and debugging information.

In this section, we create a sequential worker. The sequential worker takes each parameter CompileIQ provides and runs the user's objective function sequentially until all scores are collected.

To create a custom worker, we can inherit from the most basic worker class called `BaseWorker` defined in `compileiq.worker`.

### Base Worker

The `Worker` class is simple and only requires you to implement the `run()` method. It also maintains a few pieces of state and provides a few utilities, like a very basic normalization function.

> You can find detailed information about `Worker` in our [api documentation](api.rst#compileiq.types.Worker)

The `run` method will receive:

* `function` is the user-defined objective function
* `params_pool` is a list (usually of size `pool_size`) with all parameters CompileIQ sampled for this generation.
* `params_ids` is a list of IDs (one per element in `params_pool`) used to uniquely identify each evaluation.
* `num_function_returns` is the number of expected returns, i.e., the number of objectives defined by the user
* `**kwargs` these are any additional keyword parameters you pass to `.start()`

It expects a list of `Score` objects as a return. `Score` contains metadata so we can associate the parameters with their scores.

> You can find detailed information about `Score` in our [api documentation](api.rst#compileiq.utils.validation.Score)

### Sequential Worker

Now that we know what we can receive and what we should return, let's create a worker that sequentially calls the user's objective function with each parameter in `params_pool`.

```python
class SequentialWorker(BaseWorker):
    def run(
        self,
        *,
        function: callable,
        params_pool: list[dict | str],
        params_ids: list[int],
        num_function_returns: int = 1,
        **kwargs, # These are additional parameters passed in to `.start()`
    ) -> list[Score]:
        scores = []

        # Simple sequential execution for this worker.
        # `Score` class provides you with utilities to handle validation.
        for i, param in enumerate(params_pool):
            try:
                func_return = function(param)
            except Exception as e:
                # Make sure to handle any uncaught exceptions, or the search will be interrupted.
                logger.warning(
                    f"Unhandled exception {e} on your objective function with params {param}. "
                    "We will return an invalid score."
                )
                func_return = (
                    [INVALID_SCORE] * num_function_returns
                    if num_function_returns > 1
                    else INVALID_SCORE
                )

            valid_score = Score(
                score=func_return,
                param_id=params_ids[i],
                params=param,
                num_objectives=num_function_returns,
            )
            scores.append(valid_score)

        # Return the list of all scores measured this round (including baseline if calculated)
        return scores
```

Besides looping over the sampled parameters, notice how we also encapsulate execution to handle any errors within the objective. A warning is issued back to the user so they are aware that something went wrong, but in many use cases, failures and bad combinations are expected so we prefer not to exit early and cancel a potentially already long-running search midway through.

Another thing happening behind the scenes here is that, by instantiating the `Score` class, it performs a few validations on the score to make sure the value returned by the function adheres to the number of expected objectives and expected types.

> Note: The `Score` class also accepts a metadata field that can store additional information forwarded to the final `SearchResult` dataframe. `RayWorker`, for example, will store `node_id` and `gpu_id` information so the user can track where things executed.

#### Supporting Built-in Normalization

Because we are trying to create something generic, we also want to be able to handle the built-in [normalization feature](normalization.md).

For that, we need to calculate a baseline and add the normalized scores to the returned `Score` list.

To simplify our baselining, let's calculate it once at the first generation and use this value as the reference value to normalize all subsequent runs.

```python
class SequentialWorker(BaseWorker):
    def run(
        self,
        *,
        function: callable,
        params_pool: list[dict | str],
        params_ids: list[int],
        num_function_returns: int = 1,
        **kwargs,
    ) -> list[Score]:
        scores = []

        # It is your responsibility to handle baseline score and normalization.
        # In this example, we will calculate the baseline score only once for the search
        if self.normalize and self.baseline_score is None:
            logger.info("Calculating Baseline score for normalization.")
            baseline_score = function(BASELINE_CONFIG)
            # Save the baseline score to reuse in future batches/generations.
            self.baseline_score = Score(
                score=baseline_score,
                param_id="baseline",
                params=BASELINE_CONFIG,
                norm_score=self.normalize_scores(baseline_score, baseline_score),
                num_objectives=num_function_returns,
                is_baseline=True,
            )
            scores.append(self.baseline_score)

        # Simple sequential execution for this worker.
        # `BaseWorker` provides you with utilities to handle normalization and validation.
        for i, param in enumerate(params_pool):
            try:
                func_return = function(param)
            except Exception as e:
                # Make sure to handle any uncaught exceptions, or the search will be interrupted.
                logger.warning(
                    f"Unhandled exception {e} on your objective function with params {param}. "
                    "We will return an invalid score."
                )
                func_return = (
                    [INVALID_SCORE] * num_function_returns
                    if num_function_returns > 1
                    else INVALID_SCORE
                )

            valid_score = Score(
                score=func_return,
                param_id=params_ids[i],
                params=param,
                num_objectives=num_function_returns,
            )

            # Apply normalization if enabled. If you don't plan to use normalization,
            # you can skip this step.
            if self.normalize:
                valid_score.norm_score = self.normalize_scores(
                    valid_score.score, self.baseline_score.score
                )

            scores.append(valid_score)

        # Return the list of all scores measured this round (including baseline if calculated)
        return scores
```

Notice how we check if there is already a baseline available in `self.baseline_score` before calculating it. Afterwards, we leverage the `BaseWorker.normalize_scores` method to find the normalized value and update the `Score.norm_score` attribute. The normalized scores will be the ones used by CompileIQ to traverse the search space smartly, while the non-normalized scores are only used for tracking purposes and dumped in the resulting dataframe on `SearchResult`.

#### Bringing it all together

With our worker defined, we can now see how we enable it before starting the search:

```python
tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=SequentialWorker(
        cache_folder=tuner.cache_folder,
        normalize=main_config.normalize,
    )
)

results = tuner.start()
```

> Any arguments passed to `.start()` are accessible through `**kwargs` for the `.run` method

> You can find an updated running example in [our repository](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/ptx_spill.py/custom_worker.py).

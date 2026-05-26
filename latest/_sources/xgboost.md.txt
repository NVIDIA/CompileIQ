# Fine-tune an XGBoost ML Model

Make sure you followed the [Installation Guide](install.md). You can find the source code for [this example here](https://github.com/NVIDIA/CompileIQ/blob/main/examples/ciq_xgboost.py).

## Overview

For this example, we will try to improve the accuracy of an XGBoost model by testing out different model configurations.

The example will go through how we set the parameter search space, how we set CompileIQ configurations, and how we write the script that enables CompileIQ workers to test the different configurations.

The main program follows a standard machine learning flow: it downloads a dataset, splits it into training and validation, trains a model with a set of parameters, predicts the validation set using the trained model, and reports the accuracy.

## Prepare global variables

Your objective function has access to external values from your code. Here we download the dataset and convert it into the XGBoost format.

```python
# We are going to download the dataset and split into a train and validation subset
(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

# Converting into XGBoost's expected format
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)
```

> Warning: CompileIQ uses Python multiprocessing, so each process receives a copy of the global variables.

## The Search Space

```python
search_space_config = {
    "booster": ss.choice(["gbtree", "gblinear", "dart"]),
    "lambda": ss.log_sampling(start=1e-8, end=1.0),
    "alpha": ss.log_sampling(start=1e-8, end=1.0),
    "subsample": ss.range(0.2, 1.0, step=0.05),
    "colsample_bytree": ss.range(0.2, 1.0, step=0.05),
    "csample_type": ss.range(0.2, 1.0, step=0.05),
    "max_depth": ss.range(3, 9, step=2),
    "min_child_weight": ss.range(2, 10, step=1),
    "eta": ss.log_sampling(start=1e-8, end=1.0),
    "gamma": ss.log_sampling(start=1e-8, end=1.0),
    "grow_policy": ss.choice(["depthwise", "lossguide"]),
    "sample_type": ss.choice(["uniform", "weighted"]),
    "normalize_type": ss.choice(["tree", "forest"]),
    "rate_drop": ss.log_sampling(start=1e-8, end=1.0),
    "skip_drop": ss.log_sampling(start=1e-8, end=1.0),
}
```

These are the parameters we want to tune for this XGBoost example. Each sampled configuration is passed to your objective function as a dictionary, for example `{"booster": "gbtree", "subsample": 0.35, ...}`.

## The Search Configuration

```python
main_config = SearchConfiguration(
    pool_size=64,
    generations=5,
    mutate_rate=0.25,
    problem_type="max",
    num_objectives=1,
)
```

These are the CompileIQ-specific parameters. A small search should yield good results for this simple dataset. Larger searches should have a bigger pool size (around 128+) and larger generation count (30+).

## The Objective

```python
def objective(config: dict):
    warnings.filterwarnings("ignore")

    bst = xgb.train(config, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

    return accuracy
```

CompileIQ expects a Python callable object that receives a dictionary and returns a score of type `int | float | "*"`, where `*` represents an invalid score, such as an exception or an error.

In this example, we train and predict with the sampled parameters. We then use scikit-learn to measure prediction accuracy and return that as the score.

## Starting the search and the Results

We are now ready to pass the objective function, search space, and search configuration to `Search` and call `start`.

```python
tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
)
results = tuner.start()
```

> You can increase the parallelism with `start(num_workers=10)`. This will start 10 local processes on your machine. Look at the [Advanced Usage page](workers.md) to learn more.

The output will be a `SearchResult` object which you can then either retrieve the best result with `results.get_best_result()` or use `results.get_results()` to receive a dataframe with all tested parameters and their associated scores.

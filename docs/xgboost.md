# Fine-tune a XGBoost ML Model

Make sure you followed the [Installation Guide](install.md). You can find the source code for [this example here](https://github.com/NVIDIA/CompileIQ/blob/main/examples/ciq_xgboost.py).

## Overview

For this example, we will try to improve the accuracy of an XGBoost model by testing out different model configurations.

The example will go through how we set the parameter search space, how we set CompileIQ configurations, and how we write the script that enables CompileIQ workers to test the different configurations.

The main program follows a standard machine learning flow: it downloads a dataset, splits it into training and validation, trains a model with a set of parameters, predicts the validation set using the trained model, and reports the accuracy.

## Prepare global variables

You objective function has access to external values from your code. So we can download the dataset and convert their data into the xgboost format.

```python
# We are going to download the dataset and split into a train and validation subset
(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

# Converting into Xgboost expected format
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)
```

> Warning: CompileIQ uses python multiprocess so each process will receive a copy of the global variables.

## The Search Space

```python
dna_config = {
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

These are all parameters we want to tune for this xgboost example. Each one of them will come in as a dictionary to your objective function with the defined key and a single value such as `{config['booster']: 'gbtree', config['subsample]: 0.35, ...}`.

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

These are the CompileIQ specific parameters. A small search should yield good results for this simple dataset. Larger searches should have a bigger pool size (around 128+) and larger generation number (30+).

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

CompileIQ expects a python callable object that will receive a dictionary and return a score of type `int | float | '*'`, where `*` represents an invalid score, like an exception or an error.

In this example,  we are simply training and prediction with the given parameters in dict. We than use scikit-learn to measure the accuracy prediction and return that as the score.

## Starting the search and the Results

We are now ready to start the search, pass in the configuration, search space and the objective function for Search and hit `start`.

```python
tuner = Search(objective_function=objective, search_space=dna_config, search_config=main_config)
results = tuner.start()
```

> You can increase the parallelism with `start(num_workers=10)`. This will start 10 local processes on your machine. Look at the [Advanced Usage page](workers.md) to learn more.

The output will be a `SearchResult` object which you can then either retrieve the best result with `results.get_best_result()` or use `results.get_results()` to receive a dataframe with all tested parameters and their associated scores.

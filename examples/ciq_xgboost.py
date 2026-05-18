import warnings
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.datasets
import sklearn.metrics
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
import compileiq.search_spaces.base as ss

# We are going to download the dataset and split into a train and validation subset
(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

# Converting into XGBoost's expected format
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)


def objective(config: dict):
    warnings.filterwarnings("ignore")

    bst = xgb.train(config, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

    return accuracy


def main():
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

    main_config = SearchConfiguration(
        pool_size=64,
        generations=5,
        mutate_rate=0.25,
        problem_type="max",
        num_objectives=1,
    )

    tuner = Search(
        objective_function=objective,
        search_space=search_space_config,
        search_config=main_config,
    )
    results = tuner.start()
    print(results.get_best_result())


if __name__ == "__main__":
    main()

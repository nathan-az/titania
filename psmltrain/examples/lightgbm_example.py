import pandas as pd
from sklearn.datasets import make_classification

from psmltrain.trainers.algorithm_managers import LGBMClassifierManager

# generate training data
dataset = make_classification(n_samples=100_000, n_features=100, n_informative=40)
data = pd.DataFrame(dataset[0])
data.columns = [f"feature_{c}" for c in data.columns]
features = list(data.columns)
data["label"] = dataset[1]
data["dataset_type"] = "training"
data.loc[80_000:, "dataset_type"] = "validation"


# define parameters for training lightgbm model
model_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "max_depth": 10,
    "num_leaves": 128,
    "num_boost_round": 150,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "min_data_in_leaf": 10,
    "num_threads": 4,
    "first_metric_only": True,
    # "force_col_wise": True,
}

label_col = "label"

dataset_params = {
    "feature_name": features,
    "early_stopping_rounds": 2,
    "eval_metric": "auc",
}
dataset_type_col = "dataset_type"
use_early_stopping = True

model = LGBMClassifierManager(**model_params)
model.train(
    data,
    label_col="label",
    fit_params=dataset_params,
    dataset_type_col="dataset_type",
)

valid = data.loc[data["dataset_type"] == "validation"]
preds = model.binary_predict_proba(valid)
print(preds.shape)

label_to_pred = zip(valid["label"], preds)
for i in range(10):
    print(next(label_to_pred))

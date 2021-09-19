import pandas as pd
from sklearn.datasets import make_classification

from psmltrain.trainers.training_functions import lightgbm_training_function
from psmltrain.trainers.algorithm_managers import LightGBMTrainingManager


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
    "metric": {"auc"},
    "max_depth": 10,
    "num_leaves": 1024,
    "num_boost_round": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "min_data_in_leaf": 10,
    "num_threads": 4,
    # "force_col_wise": True,
}

label_col = "label"

dataset_params = {
    "feature_name": features,
}
dataset_type_col = "dataset_type"
use_early_stopping = True


trained = lightgbm_training_function(
    df=data,
    model_params=model_params,
    label_col=label_col,
    dataset_params=dataset_params,
    dataset_type_col=dataset_type_col,
    use_early_stopping=use_early_stopping,
)

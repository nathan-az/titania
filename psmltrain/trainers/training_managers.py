from abc import ABC, abstractmethod
from typing import Dict, Union, List, Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster
from pandas import DataFrame


class BaseTrainingManager(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str):
        joblib.dump(self, path)


class LightGBMTrainingManager(BaseTrainingManager):
    """
    Training manager for a LightGBM booster. Particularly useful for maintaining a single pickle-able class
    registering results as well as parameters used during training.
    """

    model_params: Dict[str, Union[str, int, float]]
    eval_results: Dict[str, Dict]
    model: Booster
    dataset_params: Dict[str, Union[str, int, float]]
    eval_results: Dict

    def __init__(self, model_params: Dict[str, Union[str, int, float]]):
        # copy model params on instantiation since the params are mutable and stored
        self.model_params = model_params.copy()
        self.eval_results = {}

    def train(
        self,
        df: pd.DataFrame,
        label_col: str,
        dataset_params: Dict[str, Union[str, int, float]],
        dataset_type_col: str,
        use_early_stopping: bool,
    ):
        train = df.loc[df[dataset_type_col] == "training", :]
        val = df.loc[~df[dataset_type_col] == "validation", :]

        feature_name = dataset_params.get("feature_name", None)
        self._confirm_valid_feature_name(feature_name)

        X_train = train[feature_name]
        y_train = train[label_col]

        X_val = val[feature_name]
        y_val = val[label_col]

        if "feature_name" not in dataset_params:
            raise KeyError(
                "`feature_name` should be passed in via the `features` argument, not in the `dataset_params` dictionary",
            )

        train_set = lgb.Dataset(X_train, label=y_train, **dataset_params)
        if use_early_stopping:
            val_set = lgb.Dataset(X_val, label=y_val, **dataset_params)
            self.model = lgb.train(
                params=self.model_params,
                train_set=train_set,
                valid_sets=[val_set],
                evals_result=self.eval_results,
            )
            return self

        else:
            self.model = lgb.train(params=self.model_params, train_set=train_set)
            return self

    def predict(self, df: DataFrame) -> np.ndarray:
        return self.model.predict(df)

    def _confirm_valid_feature_name(self, feature_name: Any):
        if feature_name is None:
            raise KeyError(
                f"`feature_name` must be a list of strings containing feature names to use during training. \
                Got `{type(feature_name)}`. Note: `auto` is not supported"
            )
        elif not isinstance(feature_name, list):
            raise ValueError(
                f"`feature_name` must be a list of strings containing feature names to use during training. \
                Got `{type(feature_name)}`. Note: `auto` is not supported"
            )

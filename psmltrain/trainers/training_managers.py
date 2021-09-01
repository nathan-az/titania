from abc import ABC, abstractmethod
from typing import Dict, Union, List, Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster
from pandas import DataFrame, Series


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


class SubsampledClassifier(BaseTrainingManager):
    def __init__(
        self, base_classifier_class, original_alpha: float, **base_classifier_kwargs
    ):
        self.base_classifier = base_classifier_class(**base_classifier_kwargs)
        self.original_alpha = original_alpha

    def train(self, df: DataFrame, dataset_type_col: str, label_col: str, **kwargs):
        self.sampled_alpha = self._get_sampled_alpha(
            df=df, dataset_type_col=dataset_type_col, label_col=label_col
        )
        self.base_classifier.train(df, label_col, dataset_type_col, **kwargs)
        return self

    def predict(self, df: DataFrame) -> np.ndarray:
        base_preds = self.base_classifier.predict(df)
        beta = self._calculate_beta(
            alpha_old=self.original_alpha, alpha_new=self.sampled_alpha
        )
        return self._calibrate_probabilities(probs=base_preds, beta=beta)

    def _get_sampled_alpha(
        self, df: DataFrame, dataset_type_col: str, label_col: str
    ) -> float:
        training_labels = df[df[dataset_type_col] == "training"][label_col]
        sampled_alpha = len(training_labels[training_labels == 1]) / len(
            training_labels
        )
        return sampled_alpha

    @staticmethod
    def _calculate_beta(alpha_old, alpha_new):
        return (alpha_old / (1 - alpha_old)) / (alpha_new / (1 - alpha_new))

    @staticmethod
    def _calibrate_probabilities(probs, beta):
        # See the below link for the calibration method and derivation
        # https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
        return probs * beta / (probs * beta + 1 - probs)

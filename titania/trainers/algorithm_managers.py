from typing import Dict, Union, Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier
from pandas import DataFrame

from titania.trainers.generic_managers import BaseModelManager


class LightGBMTrainingManager(BaseModelManager):
    """
    Training manager for any LightGBM booster. This is a more flexible wrapper for either
    classification/regression tasks, but does not conform to the sklearn API.
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
        val = df.loc[~(df[dataset_type_col] == "validation"), :]

        feature_name = dataset_params.get("feature_name", None)
        self._confirm_valid_feature_name(feature_name)

        X_train = train[feature_name]
        y_train = train[label_col]

        X_val = val[feature_name]
        y_val = val[label_col]

        if "feature_name" not in dataset_params:
            raise KeyError(
                "`feature_name` should be passed in via the `features` argument, not in the \
                `dataset_params` dictionary",
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

    @staticmethod
    def _confirm_valid_feature_name(feature_name: Any):
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


class LGBMClassifierManager(LGBMClassifier, BaseModelManager):
    """
    Training manager for an LGBMClassifier. Useful because it extends the class, so acts like
    a standard sklearn model.
    """

    def train(
        self,
        df: pd.DataFrame,
        label_col: str,
        fit_params: Dict[str, Any],
        dataset_type_col: Optional[str] = None,
    ):
        """
        An API for the fit method of LightGBMs sklearn API. Handles the splitting of training/validation and
        isolation of the label column

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with training and optionally validation set. Requires column named in `dataset_type_col`
            identifying at least "training" rows and optionally "validation" rows
        label_col: str
            Name of column containing binary label
        fit_params: Dict[str, Any]
            Parameters to pass into `fit`, e.g. early_stopping_rounds
        dataset_type_col: str
            Column name identifying rows as training/validation/test

        Returns
        -------
        Trained LGBMClassifierManager
        """
        feature_name = fit_params.get("feature_name", None)
        if "feature_name" not in fit_params:
            raise KeyError(
                "`feature_name` should be passed in via the `features` argument, not in \
                the `dataset_params` dictionary",
            )

        early_stopping_rounds = fit_params.get("early_stopping_rounds", 0)
        if early_stopping_rounds > 0:

            if not dataset_type_col:
                raise ValueError(
                    f"`early_stopping_rounds` set to {early_stopping_rounds} but \
                `dataset_type_col` is {dataset_type_col}. If {early_stopping_rounds} is set, a \
                 valid {dataset_type_col} must be set, with values containing at least 'training' \
                 and 'validation'."
                )

            train = df.loc[df[dataset_type_col] == "training", :]
            valid = df.loc[df[dataset_type_col] == "validation", :]
            self.fit(
                X=train[feature_name],
                y=train[label_col],
                eval_set=[(valid[feature_name], valid[label_col])],
                **fit_params,
            )
        else:
            self.fit(X=df[feature_name], y=df[label_col], **fit_params)

        return self

    def binary_predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs,
    ):
        if self.n_classes_ > 2:
            raise ValueError(
                f"`binary_predict_proba` not compatible with n_classes {self.n_classes_}"
            )
        # Supports the case that a DataFrame with excess columns is passed in,
        # otherwise assumes an np.ndarray with same column ordering as training time
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_name_]

        preds = self.predict_proba(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        if preds.shape[1] == 2:
            return preds[:, 1]
        else:
            return preds

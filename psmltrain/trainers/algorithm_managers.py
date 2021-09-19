from typing import Dict, Union, Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier
from pandas import DataFrame

from psmltrain.trainers.generic_managers import BaseModelManager


class LGBMClassifierManager(LGBMClassifier, BaseModelManager):
    """
    Training manager for a LightGBM booster. Particularly useful for maintaining a single pickle-able class
    registering results as well as parameters used during training.
    """

    model_params: Dict[str, Union[str, int, float]]
    eval_results: Dict[str, Dict]
    model: Booster
    dataset_params: Dict[str, Union[str, int, float]]
    eval_results: Dict

    def train(
        self,
        df: pd.DataFrame,
        label_col: str,
        dataset_type_col: str,
        fit_params: Dict[str, Any],
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
                "`feature_name` should be passed in via the `features` argument, not in the `dataset_params` dictionary",
            )

        if fit_params.get("early_stopping_rounds", 0) > 0:
            train = df.loc[df[dataset_type_col] == "training", :]
            valid = df.loc[df[dataset_type_col] == "validation", :]
            super(LGBMClassifier).fit(
                X=train[feature_name],
                y=train[label_col],
                eval_set=[(valid[feature_name], valid[label_col])],
                **fit_params,
            )
            return self

        else:
            super(LGBMClassifier).fit(X=df[feature_name], y=df[label_col], **fit_params)
            return self

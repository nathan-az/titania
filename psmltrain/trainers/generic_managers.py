from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
import pyspark
from pandas import DataFrame


class BaseModelManager(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str):
        joblib.dump(self, path)


class UndersampledClassifier(BaseModelManager):
    base_classifier: BaseModelManager
    original_alpha: float

    def __init__(
        self, base_classifier_class, original_alpha: float, **model_init_kwargs
    ):
        self.base_classifier = base_classifier_class(**model_init_kwargs)
        self.original_alpha = original_alpha

    def train(self, df: DataFrame, label_col: str, **training_kwargs):
        self._set_sampled_alpha(
            df=df,
            label_col=label_col,
            dataset_type_col=training_kwargs.get("dataset_type_col"),
        )
        self.base_classifier.train(df, label_col, **training_kwargs)
        return self

    def predict(self, df: DataFrame) -> np.ndarray:
        base_preds = self.base_classifier.predict(df)
        beta = self._calculate_beta(
            alpha_old=self.original_alpha, alpha_new=self.sampled_alpha
        )
        return self._calibrate_probabilities(probs=base_preds, beta=beta)

    def _set_sampled_alpha(self, df, label_col, dataset_type_col):
        if dataset_type_col:
            df = df[df[dataset_type_col] == "training"]

        self.sampled_alpha = len(df[df[label_col] == 1]) / len(df)

    @staticmethod
    def _calculate_beta(alpha_old, alpha_new):
        # This follows from the fact that the beta (probability of selecting a negative instance while
        # undersampling) is calculated through (num_pos / (desired pos/neg ratio)) / num_neg
        return (alpha_old / (1 - alpha_old)) / (alpha_new / (1 - alpha_new))

    @staticmethod
    def _calibrate_probabilities(probs, beta):
        # See the below link for the calibration method and derivation
        # https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
        return probs * beta / (probs * beta + 1 - probs)


class FlexibleModelManager(ABC):
    @abstractmethod
    def train_spark(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_local(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_spark(
        self, df: pyspark.sql.DataFrame, row_id_col: str
    ) -> pyspark.sql.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def predict_local(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def train(self, df, **kwargs):
        if isinstance(df, pyspark.sql.DataFrame):
            self.train_spark(df, **kwargs)
        else:
            self.train_local(df, **kwargs)

    def predict(self, df, **kwargs):
        if isinstance(df, pyspark.sql.DataFrame):
            self.predict_spark(df, **kwargs)
        else:
            self.predict_local(df, **kwargs)

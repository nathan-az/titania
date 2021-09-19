from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
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


class UndersampledClassifier(BaseTrainingManager):
    base_classifier: BaseTrainingManager
    original_alpha: float

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
        # This follows from the fact that the beta (probability of selecting a negative instance while
        # undersampling) is calculated through (num_pos / (desired pos/neg ratio)) / num_neg
        return (alpha_old / (1 - alpha_old)) / (alpha_new / (1 - alpha_new))

    @staticmethod
    def _calibrate_probabilities(probs, beta):
        # See the below link for the calibration method and derivation
        # https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
        return probs * beta / (probs * beta + 1 - probs)

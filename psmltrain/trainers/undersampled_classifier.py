import numpy as np
import pandas as pd


def _calculate_beta(alpha_old, alpha_new):
    # This follows from the fact that the beta (probability of selecting a negative instance while
    # undersampling) is calculated through (num_pos / (desired pos/neg ratio)) / num_neg
    return (alpha_old / (1 - alpha_old)) / (alpha_new / (1 - alpha_new))


def _calibrate_probabilities(probs, beta):
    # See the below link for the calibration method and derivation
    # https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
    return probs * beta / (probs * beta + 1 - probs)


def _undersampled_class_factory(base_class):
    """Allows bolting on of undersampling calibration if needed, in addition to any Super class, so that
    the underlying model's API is immediately accessible. See:
    https://stackoverflow.com/questions/15247075/how-can-i-dynamically-create-derived-classes-from-a-base-class
    """

    def __init__(self, original_alpha, **model_init_kwargs):
        self.original_alpha = original_alpha
        base_class.__init__(self, **model_init_kwargs)

    def set_sampled_alpha(self, df, label_col, dataset_type_col):
        if dataset_type_col:
            df = df[df[dataset_type_col] == "training"]

        self.sampled_alpha = len(df[df[label_col] == 1]) / len(df)

    def train(self, df: pd.DataFrame, label_col: str, **training_kwargs):
        set_sampled_alpha(
            self,
            df=df,
            label_col=label_col,
            dataset_type_col=training_kwargs.get("dataset_type_col"),
        )
        base_class.train(self, df=df, label_col=label_col, **training_kwargs)
        return self

    def binary_predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        base_preds = base_class.binary_predict_proba(self, df)
        beta = _calculate_beta(self.original_alpha, self.sampled_alpha)
        return _calibrate_probabilities(probs=base_preds, beta=beta)

    newclass = type(
        f"Undersampled{base_class.__name__}",
        (base_class,),
        dict(
            __init__=__init__,
            set_sampled_alpha=set_sampled_alpha,
            train=train,
            binary_predict_proba=binary_predict_proba,
        ),
    )
    return newclass

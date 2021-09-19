# pylint: disable=arguments-differ

import os
from datetime import datetime
from typing import Dict, Optional, Any, Type

import joblib
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, DoubleType
import pyspark

from psmltrain.trainers.definitions import (
    TRAIN_DATETIME_COLUMN_NAME,
    TRAINING_ROW_COUNT_COLUMN_NAME,
    SAVE_PATH_COLUMN_NAME,
    TRAINING_GROUPED_MAP_OUTPUT_SCHEMA,
)
from psmltrain.trainers.generic_managers import (
    FlexibleModelManager,
    BaseModelManager,
    UndersampledClassifier,
)


class EnsembleTrainer(FlexibleModelManager):
    model_init_kwargs: Dict
    original_alpha: float
    metadata_table: Optional[pd.DataFrame]
    models: Any

    def __init__(
        self,
        base_classifier_class: Type[BaseModelManager],
        model_init_kwargs,
        model_name: str,
        use_undersampled_classifier: bool,
        original_alpha: Optional[float] = None,
    ):
        self.base_classifier_class = base_classifier_class
        self.model_init_kwargs = model_init_kwargs
        self.original_alpha = original_alpha
        self.model_name = model_name
        if use_undersampled_classifier:
            self._valid_undersampling_alphas(original_alpha=original_alpha)
            self.original_alpha = original_alpha
            self.use_undersampled_classifier = use_undersampled_classifier

    def train_spark(
        self,
        df: pyspark.sql.DataFrame,
        label_col: str,
        model_save_dir: str,
        group_id_col: str,
        training_kwargs: Dict[str, Any],
    ):
        training_dt = datetime.now().timestamp()

        def train_udf(pdf):
            clf = self._train_single_classifier(
                df=pdf,
                label_col=label_col,
                base_classifier_class=self.base_classifier_class,
                model_init_kwargs=self.model_init_kwargs,
                training_kwargs=training_kwargs,
                use_undersampled_classifier=self.use_undersampled_classifier,
                original_alpha=self.original_alpha,
            )

            dataset_id = pdf.loc[0, group_id_col]
            suffixed_model_name = f"{self.model_name}_{dataset_id}"
            save_path = os.path.join(model_save_dir, suffixed_model_name)

            clf.save(path=save_path)

            return pd.DataFrame(
                {
                    TRAIN_DATETIME_COLUMN_NAME: [training_dt],
                    TRAINING_ROW_COUNT_COLUMN_NAME: [len(pdf)],
                    SAVE_PATH_COLUMN_NAME: [save_path],
                },
                columns=[
                    TRAIN_DATETIME_COLUMN_NAME,
                    TRAINING_ROW_COUNT_COLUMN_NAME,
                    SAVE_PATH_COLUMN_NAME,
                ],
            )

        metadata_df = (
            df.groupBy(group_id_col)
            .applyInPandas(train_udf, schema=TRAINING_GROUPED_MAP_OUTPUT_SCHEMA)
            .toPandas()
        )
        self.metadata_table = metadata_df

        file_locations = self.metadata_table[SAVE_PATH_COLUMN_NAME].tolist()
        self.models = [joblib.load(file) for file in file_locations]

        return self

    def predict_spark(self, df: pd.DataFrame, row_id_col: str) -> pd.Series:
        probability_mean_col = f"{self.model_name}_probability_mean"

        def predict_func(iterator):
            for pdf in iterator:
                predictions = pd.DataFrame(
                    {
                        row_id_col: pdf[row_id_col],
                        **{
                            f"{self.model_name}_model_{i + 1}": model.predict(pdf)
                            for i, model in enumerate(self.models)
                        },
                    }
                )
                prediction_cols = [
                    f"{self.model_name}_{i + 1}" for i in range(len(self.models))
                ]

                predictions[probability_mean_col] = predictions[prediction_cols].mean(
                    axis=1
                )

                yield predictions

        output_col_names = [
            f"{self.model_name}_{i + 1}" for i in range(len(self.models))
        ] + [
            probability_mean_col,
        ]

        output_schema = StructType(
            [df.schema[row_id_col]]
            + [StructField(column, DoubleType(), False) for column in output_col_names]
        )
        df = df.mapInPandas(predict_func, output_schema)
        return df

    def train_local(self, *args, **kwargs):
        raise NotImplementedError

    def predict_local(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _train_single_classifier(
        df: pd.DataFrame,
        label_col: str,
        base_classifier_class: Type[BaseModelManager],
        model_init_kwargs: Dict[str, Any],
        training_kwargs: Dict[str, Any],
        use_undersampled_classifier: bool,
        original_alpha: Optional[float] = None,
    ):
        if use_undersampled_classifier:
            EnsembleTrainer._valid_undersampling_alphas(original_alpha=original_alpha)
            clf = UndersampledClassifier(
                base_classifier_class=base_classifier_class,
                original_alpha=original_alpha,
                **model_init_kwargs,
            )
        else:
            clf = base_classifier_class(**model_init_kwargs)

        clf = clf.train(df=df, label_col=label_col, **training_kwargs)
        return clf

    @staticmethod
    def _valid_undersampling_alphas(original_alpha: float):
        if not original_alpha:
            raise ValueError(f"`original_alpha` must be set, got {original_alpha=}.")

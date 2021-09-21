# pylint: disable=arguments-differ

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, Type, Union, List

import joblib
import pandas as pd
from pyspark.sql.types import StructType, StructField, DoubleType
import pyspark

from titania.trainers.definitions import (
    TRAIN_DATETIME_COLUMN_NAME,
    TRAINING_ROW_COUNT_COLUMN_NAME,
    SAVE_PATH_COLUMN_NAME,
    TRAINING_GROUPED_MAP_OUTPUT_SCHEMA,
)
from titania.trainers.generic_managers import FlexibleModelManager
from titania.trainers.undersampled_classifier import _undersampled_class_factory


def _kwargs_all_type(type_, **kwargs):
    return all([isinstance(arg, type_) for kw, arg in kwargs.items()])


@dataclass
class ModelSpec:
    base_classifier_class: Type
    model_init_kwargs: Dict[str, Any]
    training_kwargs: Dict[str, Any]
    additional_options: Dict[str, float]


class EnsembleClassifier(FlexibleModelManager):
    model_specs: Union[ModelSpec, List[ModelSpec]]
    model_name: str
    metadata_table: Optional[pd.DataFrame]
    models: Any

    def __init__(
        self,
        model_specs: Union[ModelSpec, List[ModelSpec]],
        model_name: str,
    ):
        if isinstance(model_specs, list):
            self.model_specs = [
                self._update_model_spec_for_undersampling(spec=spec)
                for spec in model_specs
            ]
        else:
            self.model_specs = self._update_model_spec_for_undersampling(
                spec=model_specs
            )

        self.model_name = model_name

    def train_spark(
        self,
        df: pyspark.sql.DataFrame,
        label_col: str,
        model_save_dir: str,
        group_id_col: str,
    ):
        training_dt = datetime.now().timestamp()

        def train_udf(pdf):
            dataset_id = pdf.loc[0, group_id_col]

            if isinstance(self.model_specs, list):
                model_spec = self.model_specs[dataset_id]

            else:
                model_spec = self.model_specs

            clf = self._train_single_classifier(
                df=pdf,
                label_col=label_col,
                model_spec=model_spec,
            )

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

    def predict_spark(self, df: pyspark.sql.DataFrame, row_id_col: str) -> pd.Series:
        probability_mean_col = f"{self.model_name}_probability_mean"

        def predict_func(iterator):
            for pdf in iterator:
                # TODO: make this compatible with multi-class classification,
                #  column for each (class, submodel) pair
                predictions = pd.DataFrame(
                    {
                        row_id_col: pdf[row_id_col],
                        **{
                            f"{self.model_name}_model_{i + 1}": model.binary_predict_proba(
                                pdf
                            )
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
    def _update_model_spec_for_undersampling(spec: ModelSpec):
        if "original_alpha" in spec.additional_options:
            spec.model_init_kwargs["original_alpha"] = spec.additional_options["original_alpha"]
            undersampled_class = _undersampled_class_factory(spec.base_classifier_class)
            spec.base_classifier_class = undersampled_class
        return spec

    @staticmethod
    def _train_single_classifier(
        df: pd.DataFrame,
        label_col: str,
        model_spec: ModelSpec,
    ):
        clf = model_spec.base_classifier_class(**model_spec.model_init_kwargs)
        clf = clf.train(df=df, label_col=label_col, **model_spec.training_kwargs)
        return clf

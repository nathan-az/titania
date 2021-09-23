from abc import ABC, abstractmethod

import cloudpickle
import pandas as pd
import pyspark


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
        with open(path, "wb") as file:
            cloudpickle.dump(self, file)


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

    def save(self, path: str):
        with open(path, "wb") as file:
            cloudpickle.dump(self, file)

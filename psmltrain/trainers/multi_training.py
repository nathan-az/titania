from typing import List
import pandas as pd
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql import functions as f


def train_udf_factory(
    training_function, model_save_path, column_names: List[str], **kwargs
):
    @f.pandas_udf(returnType=ArrayType(DoubleType()))
    def train_udf(*cols):
        df = pd.concat(cols, axis=1)
        df.columns = column_names
        model = training_function(df, **kwargs)
        model.save(path=model_save_path)
        return pd.Series(model.predict(df))

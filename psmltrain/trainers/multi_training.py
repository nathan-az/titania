from typing import Optional, List, Dict, Union

import pandas as pd
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql import functions as f


def train_udf_factory(fn, column_names: List[str], **kwargs):
    @f.pandas_udf(returnType=ArrayType(DoubleType()))
    def train_udf(*cols):
        df = pd.concat(cols, axis=1)
        df.columns = column_names
        model = fn(df, **kwargs)

        return pd.Series(model.predict(df))

from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
)

TRAIN_DATETIME_COLUMN_NAME = "train_datetime"
TRAINING_ROW_COUNT_COLUMN_NAME = "num_training_rows"
SAVE_PATH_COLUMN_NAME = "save_path"

TRAINING_GROUPED_MAP_OUTPUT_SCHEMA = StructType(
    [
        StructField(TRAIN_DATETIME_COLUMN_NAME, LongType(), False),
        StructField(TRAINING_ROW_COUNT_COLUMN_NAME, LongType(), False),
        StructField(SAVE_PATH_COLUMN_NAME, StringType(), False),
    ]
)

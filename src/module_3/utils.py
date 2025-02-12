from datetime import datetime
import pandas as pd
import os
import logging


logger = logging.getLogger(__name__)
logger.level = logging.INFO

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

STORAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/")
)

PREDICTIONS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "predictions/")
)

INFO_COLS = ["variant_id", "order_id", "user_id", "created_at", "order_date"]
LABEL_COL = "outcome"
CATEGORICAL_COLS = ["product_type", "vendor"]
BINARY_COLS = ["ordered_before", "abandoned_before", "active_snoozed", "set_as_regular"]


def load_dataset() -> pd.DataFrame:
    dataset_name = "feature_frame.csv"
    loading_file = os.path.join(STORAGE_PATH, dataset_name)
    logger.info(f"Loading dataset from {loading_file}")
    return pd.read_csv(loading_file)

def get_numerical_cols(feature_frame: pd.DataFrame):
    features_cols = [col for col in feature_frame.columns if col not in INFO_COLS + [LABEL_COL]]
    return [col for col in features_cols if col not in CATEGORICAL_COLS + BINARY_COLS]

def get_feature_cols(numerical_cols, binary_cols, categorical_cols):
    return numerical_cols + binary_cols + categorical_cols

def push_relevant_orders(df: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    order_size = df.groupby("order_id").outcome.sum()
    orders_of_min_size = order_size[order_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_of_min_size)]

def build_feature_frame() -> pd.DataFrame:
    logger.info("Building feature frame")
    return (
        load_dataset()
        .pipe(push_relevant_orders)
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x: pd.to_datetime(x.order_date).dt.date)
    )

def save_predictions(y, y_pred, model_name, df):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_filtered = df[['order_id', 'user_id', 'variant_id']]

    df_predictions = df_filtered.assign(
        model_name=model_name,
        date=date,
        y=y,
        y_pred=y_pred
    )

    dataset_name = f"{model_name}_{date.replace(':', '-')}.csv"

    filename = os.path.join(PREDICTIONS_PATH, dataset_name)

    df_predictions.to_csv(filename, index=False)

    logger.info("Saving predictions into csv")
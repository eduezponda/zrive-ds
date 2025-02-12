from typing import Tuple
import pandas as pd
import os
import logging
import joblib
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator
from module_3.utils import (
    get_numerical_cols, get_feature_cols, build_feature_frame,
    LABEL_COL, BINARY_COLS, CATEGORICAL_COLS
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)


TRAIN_SIZE = 0.8

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models/")
)


def evaluate_model(model_name: str, y_test: pd.Series, y_pred: pd.Series) -> float:
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_, precision_)
    roc_auc = roc_auc_score(y_test, y_pred)
    logger.info(
        f"{model_name} results: {{PR AUC: {pr_auc:.2f}, 'ROC AUC': {roc_auc:.2f}}}"
    )
    return pr_auc

def feature_label_split(df: pd.DataFrame, feature_cols) -> Tuple[pd.DataFrame, pd.Series]:
    return df[feature_cols], df[LABEL_COL]

def train_test_split(
    df: pd.DataFrame, train_size: float, feature_cols
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()

    cutoff = cumsum_daily_orders[cumsum_daily_orders <= train_size].idxmax()

    X_train, y_train = feature_label_split(df[df.order_date <= cutoff], feature_cols)
    X_val, y_val = feature_label_split(df[df.order_date > cutoff], feature_cols)

    logger.info(
        "Splitting data on {}. {} training samples and {} validation samples".format(
            cutoff, X_train.shape[0], X_val.shape[0]
        )
    )

    return X_train, y_train, X_val, y_val

def save_model(model: BaseEstimator, model_name: str) -> None:
    logger.info(f"Saving model {model_name} to {OUTPUT_PATH}")
    
    if not os.path.exists(OUTPUT_PATH):
        logger.info(f"Creating directory {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH)

    model_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    
    joblib.dump(model, os.path.join(OUTPUT_PATH, model_name))

def ridge_model_selection(df: pd.DataFrame, feature_cols, numerical_cols) -> None:
    X_train, y_train, X_val, y_val = train_test_split(df, TRAIN_SIZE, feature_cols)

    categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", "passthrough", numerical_cols),
            ("binary", "passthrough", BINARY_COLS),
            ("categorical", categorical_preprocessor, CATEGORICAL_COLS),
        ]
    )

    lr = make_pipeline(preprocessor, StandardScaler(), LogisticRegression(penalty="l2"))
    lr.fit(X_train, y_train)

    _ = evaluate_model(
        "Ridge training", y_test=y_train, y_pred=lr.predict_proba(X_train)[:, 1]
    )

    pr_auc = evaluate_model(
        "Ridge validation", y_test=y_val, y_pred=lr.predict_proba(X_val)[:, 1]
    )

    logger.info(f"Training model with AUC: {pr_auc}")

    X, y = feature_label_split(df, feature_cols)
    lr.fit(X, y)

    save_model(lr, "ridge")

def main():
    feature_frame = build_feature_frame()

    numerical_cols = get_numerical_cols(feature_frame)

    feature_cols = get_feature_cols(numerical_cols, BINARY_COLS, CATEGORICAL_COLS)

    ridge_model_selection(feature_frame, feature_cols, numerical_cols)


if __name__ == "__main__":
    main() 

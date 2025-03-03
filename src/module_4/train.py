from typing import Tuple
import pandas as pd
import os
import logging
import joblib
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from module_4.utils import build_feature_frame, get_feature_cols, LABEL_COL
import time
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

LEARNING_RATE = 0.05
MAX_DEPTH = 5
N_TREES = 50

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models/")
)


def feature_label_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_cols(df)
    return df[feature_cols], df[LABEL_COL]

def save_model(model: BaseEstimator, model_name: str) -> None:
    logger.info(f"Saving model {model_name} to {OUTPUT_PATH}")
    
    if not os.path.exists(OUTPUT_PATH):
        logger.info(f"Creating directory {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH)

    model_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    
    joblib.dump(model, os.path.join(OUTPUT_PATH, model_name))

def time_training_model(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper

@time_training_model
def gbt_model_selection(df: pd.DataFrame) -> None:
    gbt = GradientBoostingClassifier(
        learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH, n_estimators=N_TREES
    )  

    X, y = feature_label_split(df)

    logger.info(f"Training model")
    gbt.fit(X, y)

    save_model(gbt, f"gbt_grid_search")

def main():
    df = build_feature_frame()
    gbt_model_selection(df)


if __name__ == "__main__":
    main() 

import os
import logging
from joblib import load
from module_3.train import OUTPUT_PATH, evaluate_model, feature_label_split
from module_3.utils import (
    build_feature_frame, save_predictions, get_numerical_cols, get_feature_cols,
    BINARY_COLS, CATEGORICAL_COLS 
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

def main():
    model_name = "20250212-182434_ridge.pkl"
    model = load(os.path.join(OUTPUT_PATH, model_name))
    logger.info(f"Loaded model {model_name}")

    df = build_feature_frame()

    numerical_cols = get_numerical_cols(df)

    feature_cols = get_feature_cols(numerical_cols, BINARY_COLS, CATEGORICAL_COLS)

    X, y = feature_label_split(df, feature_cols)

    y_pred = model.predict_proba(X)[:, 1]

    evaluate_model("Inference test", y, y_pred)

    save_predictions(y, y_pred, model_name, df)


if __name__ == "__main__":
    main()
import os
import logging
from joblib import load
from module_4.train import OUTPUT_PATH, feature_label_split
from module_4.utils import build_feature_frame, save_predictions


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

def main():
    model_name = "20250222-132624_gbt_grid_search.pkl"
    model = load(os.path.join(OUTPUT_PATH, model_name))
    logger.info(f"Loaded model {model_name}")

    df = build_feature_frame()
    X, y = feature_label_split(df)

    y_pred = model.predict_proba(X)[:, 1]

    save_predictions(y, y_pred, model_name, df)


if __name__ == "__main__":
    main()
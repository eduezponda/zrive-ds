import os
import logging
from joblib import load
from module_3.train import OUTPUT_PATH, evaluate_model, feature_label_split
from module_3.utils import build_feature_frame, save_predictions


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

def main():
    model_name = "20250212-012002_ridge_1e-08.pkl"
    model = load(os.path.join(OUTPUT_PATH, model_name))
    logger.info(f"Loaded model {model_name}")

    df = build_feature_frame()
    X, y = feature_label_split(df)

    y_pred = model.predict_proba(X)[:, 1]

    evaluate_model("Inference test", y, y_pred)

    save_predictions(y, y_pred, model_name, df)


if __name__ == "__main__":
    main()
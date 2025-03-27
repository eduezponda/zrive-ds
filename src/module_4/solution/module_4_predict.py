from module_4.solution.module_4_fit import DEFAULT_MODEL_FOLDER_PATH 
from module_4.solution.push_model import PushModel
from module_4.solution.utils import build_feature_frame 
from datetime import datetime
from typing import Any, Dict, Optional
import joblib
import json
import pandas as pd


def load_data(input_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(input_data, orient="index") 
    feature_frame = build_feature_frame(df)
    return feature_frame

def load_model(model_path: Optional[str]) -> PushModel: 
    """Loads classifier from model path using joblib

    Args:
        model_path: path to the joblib model in disk. If none it seeks for DEFAULT_MODEL_FOLDER_PATH/push_YYYY_MM_DD.joblib

    Raises:
        FileExistsError: if model does not exist.

    Returns:
        Instance of PushModel
    """
    
    if not model_path:
        today_str = datetime.today().strftime("%Y-%m-%d")
        model_path = DEFAULT_MODEL_FOLDER_PATH / f"push_{today_str}.joblib"
        
    if not model_path.exists():
        raise FileExistsError(f"File {model_path} does not exist.")

    clf = joblib.load(model_path)

    return clf

def handler_predict(event: Dict, _) -> Dict[str, Any]:
    """
    Entry point for predict function

    Args:
        event: {
            "users": {
                "user_id1": { "feature 1": feature value for id1,
                              "feature 2": feature value for id1, ...},
                "user_id2": { "feature 1": feature value for id2,
                              "feature 2": feature value for id2, ...},
                ...
            },
            "model_path": value
        }
    """

    data_to_predict = load_data(event["users"])
    model_path = event.get("model_path", None)
    clf = load_model(model_path)

    predictions = clf.predict(data_to_predict)

    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": predictions.to_dict()}),
    }


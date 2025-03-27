from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from module_4.solution.push_model import PushModel
from module_4.solution.utils import load_training_feature_frame
import joblib
import json
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_FOLDER_PATH = (
    Path("models").resolve()
)

DEFAULT_PREDICTION_THRESHOLD = 0.05

DEFAULT_CALIBRATION_PARAMETERISATION = {}


def create_output_path(model_folder_path: str, model_name: str) -> Path:
    """
    Creates the final output model path from a folder path and model name.

    Joins model_folder_path and model_name (removing its extension if any) and 
    adds ".joblib" extension. If that path exists, it adds "_%H_%M" to the model_name.

    Args:
        - model_folder_path: path to the folder for the model to be saved into.
        - model_name: name of the model (if extension is provided, it will be ignored).

    Warnings:
        - If the model_folder_path + model_name.joblib already exists.

    Returns:
        - The final model path, after checking that path does not exist already.
    """
    
    model_name_without_extension = Path(model_name).stem

    model_stored_path = Path(model_folder_path) / (model_name_without_extension + ".joblib")
    
    if model_stored_path.exists():
        logger.warning(
            f"Provided path: {model_stored_path} already exists. "
            f"Creating a new model_name"
        )

    while model_stored_path.exists():
        current_time = datetime.now().strftime("%H_%M")
        model_stored_path = Path(model_folder_path) / (
            model_name_without_extension + f"__{current_time}.joblib"
        )

    return model_stored_path

def save_model(clf, model_name: str, model_folder_path: Optional[str] = None) -> Path:
    """Persists model into disk using joblib with name model_path + model_name.

    Args:
        clf: fitted sklearn model.
        model_name: name of the model to be stored.
        model_folder_path: string with the folder path of the model to be saved.
                           If none, DEFAULT_MODEL_PATH is used.

    Raises:
        FileExistsError: if the model_path + model_name already exists.

    Returns:
        model_stored_path: path to the stored model, composed of model_path + model_name
    """
    if not model_folder_path:
        model_folder_path = DEFAULT_MODEL_FOLDER_PATH

    model_stored_path = create_output_path(
        model_folder_path=model_folder_path, model_name=model_name
    )

    if model_stored_path.exists():
        raise FileExistsError(f"{model_stored_path} already exists")

    joblib.dump(clf, model_stored_path)
    return model_stored_path


def _extract_model_parameters(event: Dict) -> Tuple[Path, Dict, Dict, float]:
    """Extracts model parameters from an event dictionary.

    Args:
        event: Dictionary containing model parameters.

    Returns:
        A tuple containing:
        - model_folder_path: Path where the model should be stored.
        - classifier_parametrisation: Parameters for the classifier.
        - calibration_parametrisation: Parameters for calibration.
        - prediction_threshold: Probability threshold for classification.
    """
    model_parametrisation = event["model_parametrisation"]
    classifier_parametrisation = model_parametrisation["classifier_parametrisation"]

    calibration_parametrisation = model_parametrisation.get(
        "calibration_parametrisation", DEFAULT_CALIBRATION_PARAMETERISATION
    )

    prediction_threshold = model_parametrisation.get(
        "prediction_threshold", DEFAULT_PREDICTION_THRESHOLD
    )

    model_folder_path = Path(event.get("model_folder_path", DEFAULT_MODEL_FOLDER_PATH))

    return model_folder_path, classifier_parametrisation, calibration_parametrisation, prediction_threshold

def generate_model_name(event: Dict) -> str:
    today_date_str = datetime.today().strftime("%Y_%M_%D")
    model_name = event.get("model_name", "push_" + today_date_str)
    return model_name


def handler_fit(event: Dict) -> Dict[str, Any]:
    """
    Handles the model fitting process.

    Args:
        event: Dictionary containing model parameters.

    Returns:
        A dictionary with the HTTP status code and the model path or an error message.
    """
    (
        model_folder_path,
        classifier_parametrisation,
        calibration_parametrisation,
        prediction_threshold,
    ) = _extract_model_parameters(event)

    df = load_training_feature_frame()

    model_name = generate_model_name(event)

    clf = PushModel(
        classifier_parametrisation, calibration_parametrisation, prediction_threshold
    )

    clf.fit(df)

    try:
        model_stored_path = save_model(
            clf=clf, model_name=model_name, model_folder_path=model_folder_path
        )

    except FileExistsError as e:
        return {"statusCode": "500", "body": json.dumps({"message": str(e)})}

    return {
        "statusCode": "200",
        "body": json.dumps({"model_path": str(model_stored_path)}),
    }

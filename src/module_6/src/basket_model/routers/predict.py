from fastapi import APIRouter, HTTPException

from src.basket_model.services.feature_store import FeatureStore
from src.basket_model.services.basket_model import BasketModel
from src.basket_model.exceptions import PredictionException, UserNotFoundException
from src.basket_model.models.schemas import PredictRequest, PredictResponse


router = APIRouter()
feature_store = FeatureStore()
basket_model = BasketModel()


@router.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        features = feature_store.get_features(data.user_id).to_numpy()
        predictions = basket_model.predict(features)

        return PredictResponse(prediction=predictions.mean())

    except UserNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PredictionException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

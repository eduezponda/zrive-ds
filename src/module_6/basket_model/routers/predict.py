from fastapi import APIRouter, HTTPException

from module_6.basket_model.exceptions.exceptions import PredictionException, UserNotFoundException
from module_6.basket_model.models.schemas import PredictRequest, PredictResponse
from module_6.basket_model.dependencies.predict import feature_store, basket_model


router = APIRouter()


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

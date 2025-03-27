import logging
from time import time
from fastapi import APIRouter, HTTPException
from basket_model.feature_store import FeatureStore
from basket_model.basket_model import BasketModel
from data_model import Request, Response, HTTPError
from metrics import Metrics
from exceptions import UserNotFoundException, PredictionException


logging.basicConfig(level=logging.DEBUG)

feature_store = FeatureStore()
model = BasketModel()
metrics = Metrics()

router = APIRouter(prefix="/predict")


@router.post(
    "/",
    response_model=Response,
    responses={
        200: {"model": Response},
        404: {"model": HTTPError, "description": "User not found"},
        500: {"model": HTTPError, "description": "Problems processing the text"},
    }
)
async def predict(request: Request) -> Response:
    metrics.increase_requests()
    try:
        start_time_predict = time()
        
        features = feature_store.get_features(request.user_id)
        pred = model.predict(features.values)
    
    except UserNotFoundException as exception:
        metrics.increase_user_not_found_errors()
        logging.error("User: %s, Message: %s", request.user_id, exception.message)
        raise HTTPException(status_code=404, detail="User not found") from exception
    
    except PredictionException as exception:
        metrics.increase_model_errors()
        logging.error("User: %s, Message: %s", request.user_id, exception.message)
        raise HTTPException(status_code=500, detail="Prediction not completed") from exception
    
    except Exception as exception:
        metrics.increase_unknown_errors()
        logging.error("User: %s, Unknown exception", request.user_id, exception)
        raise HTTPException(status_code=500, detail="Unknown exception") from exception
    
    metrics.observe_predict_duration(start_time_predict)
    
    return Response(basket_price=pred.mean())

from fastapi.testclient import TestClient
from src.module_6.basket_model.exceptions.exceptions import UserNotFoundException
from src.module_6.basket_model.routers.predict import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_predict_user_not_found(mocker):
    request_data = {"user_id": "404"}
    mocker.patch(
        "src.module_6.basket_model.dependencies.predict.feature_store.get_features", 
        side_effect=UserNotFoundException("User not found")
    )

    response = client.post("/predict", json=request_data)

    assert response.status_code == 404
    assert response.json() == {"detail": "UserNotFoundException: Message: User not found"}


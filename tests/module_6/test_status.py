from fastapi.testclient import TestClient
from src.module_6.basket_model.routers.status import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_status():
    response = client.get("/status")
    
    assert response.status_code == 200  


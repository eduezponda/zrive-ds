import uvicorn
from fastapi import FastAPI
from module_6.basket_model.routers import status, predict, metrics

app = FastAPI()

app.include_router(status.router)
app.include_router(predict.router)
app.include_router(metrics.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

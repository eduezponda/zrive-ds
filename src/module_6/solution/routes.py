from fastapi import FastAPI
from handlers import predict, status, metrics


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(status.router)
    app.include_router(metrics.router)
    app.include_router(predict.router)
    return app

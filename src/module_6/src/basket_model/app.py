import uvicorn
from fastapi import FastAPI
from routers import status, predict

app = FastAPI()

app.include_router(status.router)
app.include_router(predict.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

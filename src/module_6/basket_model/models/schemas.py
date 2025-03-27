from pydantic import BaseModel

class PredictRequest(BaseModel):
    user_id: str

class PredictResponse(BaseModel):
    prediction: float

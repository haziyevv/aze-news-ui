
import uvicorn
from fastapi import FastAPI, HTTPException
from utils import predict_custom_trained_model_sample
from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    categories: List[str]

app = FastAPI()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    POST /predict
    """
    texts = request.texts
    
    instances = [{"text": x} for x in texts]
    results = predict_custom_trained_model_sample(instances=instances)
    
    return PredictResponse(categories=results)


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load trained model
with open("model.pkl","rb") as f:
    model = pickle.load(f)

# Input schema
class IrisInput(BaseModel):
    features:list[float]

@app.get("/")
def home():
    return {"message":"Iris Prediction API running"}

@app.post("/predict")
def predict(data:IrisInput):

    features=np.array(data.features).reshape(1,-1)

    prediction=model.predict(features)

    return {"prediction":int(prediction[0])}
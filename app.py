from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("C:/Users/mahul/Desktop/Socket Programming/Project/Model/fraud_model.pkl")

class Transaction(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: Transaction):

    data = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }
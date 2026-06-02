from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI()


model = joblib.load(
    "../models/xgboost_fraud.pkl"
)


class Transaction(BaseModel):
    transaction_id: int
    amount: float
    merchant: int
    type: int



@app.get("/")
def home():

    return {
        "message": "FinGuard AI Backend Running"
    }



@app.post("/predict")
def predict(data: Transaction):

    sample = np.array([[
        data.transaction_id,
        data.amount,
        data.merchant,
        data.type
    ]])

    prediction = model.predict(sample)

    return {
        "prediction": int(prediction[0]),
        "result":
            "Fraud"
            if prediction[0] == 1
            else "Safe Transaction"
    }
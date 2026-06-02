from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud.pkl"

FEATURE_COLUMNS = ["transaction_id", "amount", "merchant", "type"]

MERCHANT_ENCODER = {
    "amazon": 0,
    "atm": 1,
    "netflix": 2,
    "unknown": 3,
    "uber": 4,
    "swiggy": 5,
    "zomato": 6,
    "flipkart": 7,
    "paytm": 8,
    "airbnb": 9,
    "apple": 10,
    "irctc": 11,
}

TYPE_ENCODER = {
    "shopping": 0,
    "cash": 1,
    "transfer": 2,
    "subscription": 3,
    "travel": 4,
    "food": 5,
    "bill": 6,
    "wallet": 7,
}

app = FastAPI(
    title="FinGuard AI API",
    description="Fraud risk scoring API powered by an XGBoost classifier.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
    transaction_id: int = Field(..., ge=1)
    amount: float = Field(..., ge=0)
    merchant: Union[int, str]
    type: Union[int, str]


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=250)


def _encode_value(value: Union[int, str], mapping: dict[str, int], field_name: str) -> int:
    if isinstance(value, int):
        return value

    normalized = value.strip().lower()
    if normalized in mapping:
        return mapping[normalized]

    allowed = ", ".join(sorted(mapping))
    raise HTTPException(
        status_code=422,
        detail=f"Unknown {field_name} '{value}'. Allowed values: {allowed}, or provide an integer code.",
    )


def _risk_band(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    if score >= 0.5:
        return "High"
    if score >= 0.3:
        return "Medium"
    return "Low"


def _reason_codes(transaction: Transaction, risk_score: float) -> list[str]:
    reasons = []
    if transaction.amount >= 50000:
        reasons.append("High value transaction")
    elif transaction.amount >= 25000:
        reasons.append("Elevated amount")

    merchant_code = _encode_value(transaction.merchant, MERCHANT_ENCODER, "merchant")
    type_code = _encode_value(transaction.type, TYPE_ENCODER, "type")

    if merchant_code == MERCHANT_ENCODER["unknown"]:
        reasons.append("Unrecognized merchant")
    if type_code in {TYPE_ENCODER["transfer"], TYPE_ENCODER["cash"]}:
        reasons.append("High risk transaction type")
    if risk_score >= 0.5:
        reasons.append("Model probability crossed review threshold")

    return reasons or ["No major risk indicators"]


def _predict_one(transaction: Transaction) -> dict:
    sample = pd.DataFrame(
        [[
            transaction.transaction_id,
            transaction.amount,
            _encode_value(transaction.merchant, MERCHANT_ENCODER, "merchant"),
            _encode_value(transaction.type, TYPE_ENCODER, "type"),
        ]],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(sample)[0])
    probability = model.predict_proba(sample)
    risk_score = float(probability[0][1])

    return {
        "transaction_id": transaction.transaction_id,
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Safe Transaction",
        "risk_score": risk_score,
        "risk_band": _risk_band(risk_score),
        "reason_codes": _reason_codes(transaction, risk_score),
    }


@app.get("/")
def home():
    return {
        "message": "FinGuard AI Backend Running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH.name,
        "features": FEATURE_COLUMNS,
    }


@app.get("/metadata")
def metadata():
    return {
        "merchant_encoder": MERCHANT_ENCODER,
        "type_encoder": TYPE_ENCODER,
        "thresholds": {
            "medium": 0.3,
            "high": 0.5,
            "critical": 0.75,
        },
    }


@app.post("/predict")
def predict(data: Transaction):
    return _predict_one(data)


@app.post("/predict-batch")
def predict_batch(batch: BatchRequest):
    results = [_predict_one(transaction) for transaction in batch.transactions]
    return {
        "count": len(results),
        "fraud_count": sum(item["prediction"] for item in results),
        "average_risk_score": float(np.mean([item["risk_score"] for item in results])),
        "results": results,
    }

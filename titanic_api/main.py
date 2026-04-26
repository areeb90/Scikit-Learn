from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Survival Predictor",
    description="Predicts survival probability using a tuned Random Forest pipeline.",
    version="1.0.0"
)

# ── Load model at startup (once, not per request) ───────────────────────────────
MODEL_PATH = "titanic_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Copy it from your notebook output.")

model = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(model)}")


# ── Request schema ──────────────────────────────────────────────────────────────
# These match EXACTLY the columns your pipeline was trained on.
# Pydantic validates types and gives clear error messages on bad input.

class PassengerFeatures(BaseModel):
    pclass:      int   = Field(..., ge=1, le=3,   description="Passenger class: 1, 2, or 3")
    age:         float = Field(..., ge=0, le=120,  description="Age in years")
    sibsp:       int   = Field(..., ge=0,           description="Number of siblings/spouses aboard")
    parch:       int   = Field(..., ge=0,           description="Number of parents/children aboard")
    fare:        float = Field(..., ge=0,           description="Ticket fare in GBP")
    sex:         str   = Field(...,                 description="'male' or 'female'")
    embarked:    str   = Field(...,                 description="Port: 'C', 'Q', or 'S'")
    family_size: int   = Field(..., ge=1,           description="sibsp + parch + 1")
    is_alone:    int   = Field(..., ge=0, le=1,    description="1 if travelling alone, else 0")
    title:       str   = Field(...,                 description="'Mr', 'Mrs', 'Miss', 'Master', or 'Rare'")


# ── Response schema ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:          int   # 0 = did not survive, 1 = survived
    prediction_label:    str   # human-readable
    survival_probability: float # probability of survival (0.0 – 1.0)


# ── Health check ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": str(type(model).__name__)}


# ── Prediction endpoint ─────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerFeatures):
    try:
        # Build a single-row DataFrame — same format your pipeline was trained on
        input_df = pd.DataFrame([passenger.model_dump()])

        # Run prediction (pipeline handles all preprocessing internally)
        prediction      = int(model.predict(input_df)[0])
        survival_prob   = float(model.predict_proba(input_df)[0][1])

        return PredictionResponse(
            prediction           = prediction,
            prediction_label     = "Survived" if prediction == 1 else "Did not survive",
            survival_probability = round(survival_prob, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Auto-docs endpoint reminder ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Titanic Survival Predictor API",
        "docs":    "Visit /docs for interactive API documentation"
    }
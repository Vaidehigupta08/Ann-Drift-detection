from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json
import os
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "model/autoencoder_drift_model.h5"), compile=False)
scaler = joblib.load(os.path.join(BASE_DIR, "model/scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "model/feature_columns.pkl"))

with open(os.path.join(BASE_DIR, "model/drift_threshold.json")) as f:
    threshold = json.load(f)["threshold"]

# ---------- Request Schema ----------
class Driftrequest(BaseModel):
    columns: list[str]
    data: list[list[float]]

# ---------- Core Logic ----------
def check_drift(req: Driftrequest):

    df = pd.DataFrame(req.data, columns=req.columns)

    # 🔥 STRICT feature alignment
    df = df[feature_columns]

    X = np.nan_to_num(df.values, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[1] != scaler.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {scaler.n_features_in_} features, got {X.shape[1]}"
        )

    X_scaled = scaler.transform(X)
    recon = model.predict(X_scaled, verbose=0)

    errors = np.mean((X_scaled - recon) ** 2, axis=1)
    drift_ratio = float((errors > threshold).mean())

    return drift_ratio

# ---------- API ----------
@app.post("/detect_drift")
def detect_drift(request: Driftrequest):
    drift_ratio = check_drift(request)
    return {
        "drift_ratio": drift_ratio,
        "drift_detected": drift_ratio > 0.3
    }

@app.post("/sanity_check")
def sanity_check(request: Driftrequest):
    df = pd.DataFrame(request.data, columns=request.columns)
    df = df[feature_columns]

    X = np.nan_to_num(df.values)
    X_scaled = scaler.transform(X)

    recon = model.predict(X_scaled, verbose=0)
    errors = np.mean((X_scaled - recon)**2, axis=1)

    return {
        "mean_error": float(errors.mean()),
        "p95_error": float(np.percentile(errors, 95)),
        "threshold": float(threshold)
    }

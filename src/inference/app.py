from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import mlflow
import mlflow.pyfunc

# === MLflow Configuration ===
mlflow.set_tracking_uri("http://localhost:5000")  # Local or remote MLflow server

# === Load Registered Model by Version ===
MODEL_URI = "models:/Churn_Model/2"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("✅ Model loaded successfully from registry (version 2).")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise e

# === FastAPI App ===
app = FastAPI(title="Churn Prediction API")

# === Input Schema ===
class InferenceRequest(BaseModel):
    features: List[Dict]

@app.get("/")
def root():
    return {"message": "Churn Prediction API is up and running."}

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        # Convert input features to DataFrame
        input_df = pd.DataFrame(request.features)
        if input_df.empty:
            raise HTTPException(status_code=400, detail="Empty input data.")

        # The model includes preprocessing, so raw input is fine
        predictions = model.predict(input_df)

        print(f"🔍 Received {len(input_df)} rows. Returning {len(predictions)} predictions.")

        return {
            "n_predictions": len(predictions),
            "predictions": predictions.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

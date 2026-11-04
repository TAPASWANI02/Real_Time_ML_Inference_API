
from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

from app.schemas import PredictionInput, PredictionOutput

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="REST API for predicting Titanic passenger survival",
    version="1.0.0"
)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):

    # Convert input into dictionary
    input_dict = input_data.model_dump()

    # Create DataFrame with the exact model features
    input_df = pd.DataFrame([input_dict])

    # Generate prediction
    prediction = int(model.predict(input_df)[0])

    # Get probability for the predicted class
    probabilities = model.predict_proba(input_df)[0]
    class_index = list(model.classes_).index(prediction)
    probability = float(probabilities[class_index])

    # Convert prediction to label
    if prediction == 1:
        label = "Survived"
    else:
        label = "Did Not Survive"

    return PredictionOutput(
        prediction=prediction,
        prediction_label=label,
        probability=probability
    )

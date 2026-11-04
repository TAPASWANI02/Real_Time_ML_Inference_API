# Titanic Survival Prediction API

## Project Overview
This project provides a real-time Machine Learning inference REST API using FastAPI. It uses a trained Scikit-learn Gradient Boosting Classifier pipeline to predict whether a Titanic passenger survived.

## Technologies Used
- Python 3.12
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Pytest
- Docker

## Model Details
- Algorithm: Gradient Boosting Classifier
- Model Type: Scikit-learn Pipeline
- Number of Input Features: 12
- Scikit-learn Version: 1.6.1

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | API health status |
| POST | `/predict` | Predict passenger survival |

## Prediction Input Example

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29,
  "sibsp": 0,
  "parch": 0,
  "fare": 100.0,
  "embarked": "S",
  "who": "woman",
  "adult_male": false,
  "deck": "C",
  "embark_town": "Cherbourg",
  "alone": true
}
```

## Prediction Output Example

```json
{
  "prediction": 1,
  "prediction_label": "Survived",
  "probability": 0.9419368803364608
}
```

## Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

http://127.0.0.1:8000/docs

## Running Tests

```bash
python -m pytest -v tests/test_api.py
```

## Docker

Build the Docker image:

```bash
docker build -t titanic-survival-api .
```

Run the container:

```bash
docker run -p 8000:8000 titanic-survival-api
```

## Testing Results
All four API unit tests passed:
- Home endpoint
- Health endpoint
- Valid prediction
- Invalid input validation

## Project Objective
The objective is to demonstrate how a trained Machine Learning model can be integrated into a REST API for real-time prediction and deployment.
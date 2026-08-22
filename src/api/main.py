import os
import joblib
import pandas as pd
from fastapi import FastAPI
from src.api.schemas import LoanApplication

app = FastAPI(title="LoanGuard ML API", version="1.0")

# Build absolute paths to model artifacts, so this works regardless of where uvicorn is launched from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models_dir = os.path.join(project_root, "models")

# Load the matched set ONCE at startup, not per-request
model = joblib.load(os.path.join(models_dir, "champion_model.pkl"))
encoder = joblib.load(os.path.join(models_dir, "encoder.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

# Must match exactly what train.py used
BINARY_COLS = ['HasMortgage', 'HasDependents', 'HasCoSigner']
EDUCATION_ORDER = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
NOMINAL_COLS = ['EmploymentType', 'MaritalStatus', 'LoanPurpose']
NUMERIC_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']


@app.get("/")
def root():
    return {"message": "LoanGuard ML API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    # Convert incoming request into a single-row DataFrame
    input_df = pd.DataFrame([application.dict()])

    # Apply the SAME preprocessing steps as training, in the SAME order

    # 1. Binary Yes/No mapping
    for col in BINARY_COLS:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    # 2. Ordinal encoding
    input_df['Education'] = input_df['Education'].map(EDUCATION_ORDER)

    # 3. One-hot encoding using the FITTED encoder (not a new one)
    encoded = encoder.transform(input_df[NOMINAL_COLS])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(NOMINAL_COLS), index=input_df.index)
    input_df = pd.concat([input_df.drop(columns=NOMINAL_COLS), encoded_df], axis=1)

    # 4. Scale numeric columns using the FITTED scaler
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

    # 5. Ensure column order matches training exactly
    input_df = input_df[model.feature_names_in_]

    # 6. Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "default_prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "risk_level": "High" if probability >= 0.5 else "Low"
    }
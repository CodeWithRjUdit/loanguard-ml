# LoanGuard ML

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loanguard-ml.streamlit.app/)

**🚀 Live application: [https://loanguard-ml.streamlit.app/](https://loanguard-ml.streamlit.app/)**

**LoanGuard ML** is a machine-learning demonstration that estimates the probability of loan default from applicant and loan characteristics. It exposes a Streamlit interface backed by a FastAPI prediction service.

> **Important:** This project is an educational prototype. It is not a credit-scoring product, financial recommendation, or substitute for a regulated lending process. Do not use its output to make real lending decisions.

## Application architecture

```text
User
  |
  v
Streamlit frontend
(src/frontend/app.py)
  |
  | POST /predict
  v
FastAPI backend
(src/api/main.py)
  |
  |-- Binary Yes/No mapping
  |-- Education ordinal mapping
  |-- Fitted one-hot encoder
  |-- Fitted numeric scaler
  |
  v
Saved Logistic Regression model
(models/champion_model.pkl)
  |
  v
Default probability + Low/High risk label
```

The frontend is deployed on Streamlit Community Cloud. The API can be hosted separately (for example, as a Docker web service on Render). The frontend's default API endpoint is:

```text
https://loanguard-ml-api.onrender.com/predict
```

You can override it using the `API_URL` Streamlit secret or the `LOANGUARD_API_URL` environment variable.

## Repository structure

```text
loanguard-ml/
├── data/
│   └── raw/
│       └── loan_default.csv.dvc
├── models/
│   ├── champion_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   └── data_preprocessing.py
│   ├── frontend/
│   │   ├── app.py
│   │   └── requirements.txt
│   └── models/
│       └── train.py
├── Dockerfile
├── requirements-api.txt
└── requirements.txt
```

The fitted model, encoder, and scaler are versioned in the repository and are loaded by the API at startup. The raw training CSV is tracked through DVC rather than committed as a regular Git file; it is not required to serve predictions.

## Input features

The application collects 16 fields:

- **Numerical:** age, annual income, loan amount, credit score, months employed, number of credit lines, interest rate, loan term, and debt-to-income ratio.
- **Categorical/binary:** education, employment type, marital status, mortgage status, dependent status, loan purpose, and co-signer status.

The API applies the fitted preprocessing objects saved during training. It returns the predicted class, estimated default probability, and a risk label. The current API uses a probability threshold of 0.5 for the Low/High label.

## Deployment

### Frontend — Streamlit Community Cloud

The Streamlit frontend is deployed and publicly accessible:

**[Open LoanGuard ML](https://loanguard-ml.streamlit.app/)**

Deployment configuration:

- **Repository:** `CodeWithRjUdit/loanguard-ml`
- **Branch:** `master`
- **Main file path:** `src/frontend/app.py`
- **Python version:** 3.11

The frontend dependency file is located beside the entrypoint at `src/frontend/requirements.txt`, so the cloud build installs only the frontend packages rather than the full model-training environment.

### Backend — Render

The repository includes a Dockerfile that starts the FastAPI application on port 8000.

1. In Render, create a new **Web Service** and connect this GitHub repository.
2. Select the `master` branch and use the **Docker** runtime.
3. Set the Dockerfile path to `./Dockerfile`.
4. Deploy the service.
5. Check the service root URL. It should return a JSON message confirming that the LoanGuard API is running.
6. Use the service URL plus `/predict` as the frontend API endpoint.

For example:

```text
https://YOUR-RENDER-SERVICE.onrender.com/predict
```

The frontend currently has the original Render endpoint as its default. If your Render service has a different URL, add this in the Streamlit app's **Settings → Secrets**:

```toml
API_URL = "https://YOUR-RENDER-SERVICE.onrender.com/predict"
```

The backend's Docker image installs dependencies from `requirements-api.txt`.

## Run locally

Start the API from the repository root:

```bash
pip install -r requirements-api.txt
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, install and start the frontend:

**Windows PowerShell**

```powershell
pip install -r src/frontend/requirements.txt
$env:LOANGUARD_API_URL = "http://127.0.0.1:8000/predict"
streamlit run src/frontend/app.py
```

**macOS/Linux**

```bash
pip install -r src/frontend/requirements.txt
export LOANGUARD_API_URL="http://127.0.0.1:8000/predict"
streamlit run src/frontend/app.py
```

## Notes and limitations

- The model and preprocessing artifacts must remain a matched set: `champion_model.pkl`, `encoder.pkl`, and `scaler.pkl`.
- The API loads these artifacts once when the service starts.
- The training CSV is not needed for inference.
- Model outputs can reflect bias, data limitations, and distribution shift in the training data. The current prototype does not constitute a fairness, compliance, or credit-risk validation assessment.
- The probability threshold is currently fixed at 0.5 in the API; it is not a calibrated lending policy.

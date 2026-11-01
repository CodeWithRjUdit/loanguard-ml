import os

import requests
import streamlit as st

DEFAULT_API_URL = "https://loanguard-ml-api.onrender.com/predict"

st.set_page_config(page_title="LoanGuard ML", page_icon="💰")

# Configure API_URL in Streamlit Community Cloud > App settings > Secrets,
# or set LOANGUARD_API_URL as an environment variable for local development.
API_URL = (
    os.getenv("LOANGUARD_API_URL")
    or st.secrets.get("API_URL", DEFAULT_API_URL)
).strip()

st.title("LoanGuard ML — Loan Default Risk Predictor")
st.write("Enter applicant details to estimate the probability of loan default.")
st.caption(
    "This is an educational machine-learning demonstration, not a credit "
    "approval system or financial recommendation."
)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income", min_value=1, value=65000)
        loan_amount = st.number_input("Loan Amount", min_value=1, value=120000)
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650
        )
        months_employed = st.number_input(
            "Months Employed", min_value=0, value=48
        )
        num_credit_lines = st.number_input(
            "Number of Credit Lines", min_value=0, value=3
        )
        interest_rate = st.number_input(
            "Interest Rate (%)", min_value=0.1, value=12.5
        )
        loan_term = st.number_input(
            "Loan Term (months)", min_value=1, value=36
        )

    with col2:
        dti_ratio = st.slider(
            "Debt-to-Income Ratio", 0.0, 1.0, 0.4
        )
        education = st.selectbox(
            "Education", ["High School", "Bachelor's", "Master's", "PhD"]
        )
        employment_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        )
        marital_status = st.selectbox(
            "Marital Status", ["Married", "Divorced", "Single"]
        )
        has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
        has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"]
        )
        has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

    submitted = st.form_submit_button(
        "Predict Default Risk", use_container_width=True
    )

if submitted:
    payload = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }

    try:
        with st.spinner("Evaluating the application..."):
            # A longer timeout allows a hosted API to wake from an idle state.
            response = requests.post(API_URL, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()

        st.subheader("Prediction Result")
        risk = result["risk_level"]
        probability = result["default_probability"]

        st.metric("Estimated Default Probability", f"{probability:.1%}")

        if risk == "High":
            st.error("Higher estimated default risk")
        else:
            st.success("Lower estimated default risk")

        st.caption(
            "The risk label uses the API's current 0.5 probability threshold. "
            "It should not be interpreted as a lending decision."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The prediction API did not respond within 90 seconds. "
            "The hosted backend may be waking from an idle state. Please retry."
        )
    except requests.exceptions.HTTPError as exc:
        st.error(
            f"The prediction API returned an error "
            f"({exc.response.status_code}). Please check the backend logs."
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
    except (KeyError, ValueError) as exc:
        st.error(
            "The API returned an unexpected response. "
            "Check that the deployed backend matches this frontend."
        )

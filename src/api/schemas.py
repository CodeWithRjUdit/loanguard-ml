from pydantic import BaseModel, Field
from typing import Literal


class LoanApplication(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Applicant age")
    Income: float = Field(..., gt=0, description="Annual income")
    LoanAmount: float = Field(..., gt=0, description="Requested loan amount")
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score")
    MonthsEmployed: int = Field(..., ge=0, description="Months at current employment")
    NumCreditLines: int = Field(..., ge=0, description="Number of open credit lines")
    InterestRate: float = Field(..., gt=0, description="Interest rate offered")
    LoanTerm: int = Field(..., gt=0, description="Loan term in months")
    DTIRatio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    Education: Literal["High School", "Bachelor's", "Master's", "PhD"]
    EmploymentType: Literal["Full-time", "Part-time", "Self-employed", "Unemployed"]
    MaritalStatus: Literal["Married", "Divorced", "Single"]
    HasMortgage: Literal["Yes", "No"]
    HasDependents: Literal["Yes", "No"]
    LoanPurpose: Literal["Auto", "Business", "Education", "Home", "Other"]
    HasCoSigner: Literal["Yes", "No"]

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Income": 65000,
                "LoanAmount": 120000,
                "CreditScore": 650,
                "MonthsEmployed": 48,
                "NumCreditLines": 3,
                "InterestRate": 12.5,
                "LoanTerm": 36,
                "DTIRatio": 0.4,
                "Education": "Bachelor's",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Married",
                "HasMortgage": "Yes",
                "HasDependents": "No",
                "LoanPurpose": "Auto",
                "HasCoSigner": "No"
            }
        }
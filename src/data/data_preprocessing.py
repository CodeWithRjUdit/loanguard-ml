import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


EDUCATION_ORDER = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
BINARY_COLS = ['HasMortgage', 'HasDependents', 'HasCoSigner']
NOMINAL_COLS = ['EmploymentType', 'MaritalStatus', 'LoanPurpose']
NUMERIC_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']


def load_and_split(csv_path, test_size=0.2, random_state=42):
    """Load raw CSV, drop identifier, split into stratified train/test."""
    csv_path = "C:\\Users\\hp\\Desktop\\Machine Learning\\loanguard-ml\\data\\raw\\loan_default.csv"
    df =pd.read_csv(csv_path)
    X = df.drop(columns=['LoanID', 'Default'])
    y = df['Default']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_and_scale(X_train, X_test):
    """Fit encoders/scaler on train only, apply to both. Returns transformed X_train, X_test."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Binary Yes/No columns
    for col in BINARY_COLS:
        X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})
        X_test[col] = X_test[col].map({'Yes': 1, 'No': 0})

    # Ordinal
    X_train['Education'] = X_train['Education'].map(EDUCATION_ORDER)
    X_test['Education'] = X_test['Education'].map(EDUCATION_ORDER)

    # Nominal - one-hot
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    encoder.fit(X_train[NOMINAL_COLS])

    train_encoded = pd.DataFrame(
        encoder.transform(X_train[NOMINAL_COLS]),
        columns=encoder.get_feature_names_out(NOMINAL_COLS),
        index=X_train.index
    )
    test_encoded = pd.DataFrame(
        encoder.transform(X_test[NOMINAL_COLS]),
        columns=encoder.get_feature_names_out(NOMINAL_COLS),
        index=X_test.index
    )

    X_train = pd.concat([X_train.drop(columns=NOMINAL_COLS), train_encoded], axis=1)
    X_test = pd.concat([X_test.drop(columns=NOMINAL_COLS), test_encoded], axis=1)

    # Scale continuous numeric columns
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, encoder, scaler


def prepare_data(csv_path, test_size=0.2, random_state=42):
    """Full pipeline: load, split, encode, scale. Returns X_train, X_test, y_train, y_test."""
    X_train, X_test, y_train, y_test = load_and_split(csv_path, test_size, random_state)
    X_train, X_test, encoder, scaler = encode_and_scale(X_train, X_test)
    return X_train, X_test, y_train, y_test

print("Data preprocessing module loaded successfully.")
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.data_preprocessing import load_and_split, encode_and_scale


def train():
    # Paths - built relative to project root, regardless of where script is run from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "raw", "loan_data.csv")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'mlflow.db')}")
    mlflow.set_experiment("loanguard-imbalance-comparison")

    X_train, X_test, y_train, y_test = load_and_split(csv_path)
    X_train, X_test, encoder, scaler = encode_and_scale(X_train, X_test)

    with mlflow.start_run(run_name="logreg_class_weighted_FINAL"):
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("imbalance_strategy", "class_weight_balanced")
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision_class1", precision_score(y_test, y_pred))
        mlflow.log_metric("recall_class1", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_class1", f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, os.path.join(models_dir, "champion_model.pkl"))
        joblib.dump(encoder, os.path.join(models_dir, "encoder.pkl"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

        print("Training complete. Model, encoder, and scaler saved to /models.")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"Recall (class 1): {recall_score(y_test, y_pred):.3f}")


if __name__ == "__main__":
    train()
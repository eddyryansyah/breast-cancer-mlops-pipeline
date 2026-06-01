import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_path: str) -> pd.DataFrame:
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_file}")

    return pd.read_csv(dataset_file)


def train_model(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return model_pipeline, X_train, X_test, y_train, y_test, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="breast_cancer_preprocessing.csv",
    )
    args = parser.parse_args()

    df = load_dataset(args.dataset_path)

    model_pipeline, X_train, X_test, y_train, y_test, metrics = train_model(df)

    signature = infer_signature(X_test, model_pipeline.predict(X_test))
    input_example = X_test.iloc[:5]

    # MLflow Project sudah membuat run secara otomatis.
    # Karena itu, script ini langsung melakukan logging tanpa mlflow.start_run().
    mlflow.log_param("dataset_path", args.dataset_path)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("min_samples_split", 2)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("train_rows", X_train.shape[0])
    mlflow.log_param("test_rows", X_test.shape[0])
    mlflow.log_param("total_features", X_train.shape[1])

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    model_output_path = "model.joblib"
    joblib.dump(model_pipeline, model_output_path)
    mlflow.log_artifact(model_output_path, artifact_path="model_artifact")

    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    print("Training MLProject selesai.")
    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()

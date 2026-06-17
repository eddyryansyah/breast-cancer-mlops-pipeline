import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Memuat dataset hasil preprocessing."""
    df = pd.read_csv(dataset_path)
    return df


def train_model(df: pd.DataFrame):
    """Melatih model klasifikasi menggunakan dataset hasil preprocessing."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
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

    return model_pipeline, metrics


def main():
    dataset_path = "breast_cancer_preprocessing.csv"

    mlflow.set_experiment("Breast Cancer Classification")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        df = load_dataset(dataset_path)
        model_pipeline, metrics = train_model(df)

        print("Training model selesai.")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Memuat dataset hasil preprocessing."""
    df = pd.read_csv(dataset_path)
    return df


def prepare_data(df: pd.DataFrame):
    """Memisahkan fitur-target dan membagi dataset."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def tune_model(X_train, y_train):
    """Melakukan hyperparameter tuning menggunakan GridSearchCV."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_model(model, X_test, y_test):
    """Melakukan evaluasi model pada data uji."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics["roc_auc"] = auc(fpr, tpr)

    return y_pred, fpr, tpr, metrics


def save_artifacts(model, X_test, y_test, y_pred, fpr, tpr, artifact_dir: str):
    """Menyimpan artefak tambahan untuk MLflow."""
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = artifact_path / "classification_report.json"
    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 5))
    cm_display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.tight_layout()
    confusion_matrix_path = artifact_path / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Tuned Random Forest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_curve_path = artifact_path / "roc_curve.png"
    plt.savefig(roc_curve_path)
    plt.close()

    feature_names = X_test.columns
    feature_importances = model.named_steps["model"].feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importances,
        }
    ).sort_values(by="importance", ascending=False)

    feature_importance_path = artifact_path / "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)

    return {
        "classification_report": report_path,
        "confusion_matrix": confusion_matrix_path,
        "roc_curve": roc_curve_path,
        "feature_importance": feature_importance_path,
    }


def main():
    dataset_path = "breast_cancer_preprocessing.csv"
    artifact_dir = "artifacts"

    dagshub.init(
        repo_owner="eddyryansyah",
        repo_name="SMSML_Eddy-ryansyah",
        mlflow=True,
    )

    mlflow.set_experiment("Breast Cancer Classification")

    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = prepare_data(df)

    grid_search = tune_model(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred, fpr, tpr, metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    artifacts = save_artifacts(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        fpr=fpr,
        tpr=tpr,
        artifact_dir=artifact_dir,
    )

    signature = infer_signature(X_test, best_model.predict(X_test))
    input_example = X_test.iloc[:5]

    with mlflow.start_run(run_name="RandomForest_Tuning_DagsHub_ManualLogging"):
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("total_features", X_train.shape[1])

        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        for artifact_name, artifact_file in artifacts.items():
            mlflow.log_artifact(str(artifact_file), artifact_path="additional_artifacts")

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=None,
        )

    print("Training tuning dengan DagsHub selesai.")
    print("Best parameters:")
    print(grid_search.best_params_)
    print("\nMetrics:")
    print(f"best_cv_score: {grid_search.best_score_:.4f}")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nArtefak tambahan berhasil dibuat dan dilog ke DagsHub:")
    for artifact_name, artifact_file in artifacts.items():
        print(f"{artifact_name}: {artifact_file}")

    print("\nDagsHub repository:")
    print("https://dagshub.com/eddyryansyah/SMSML_Eddy-ryansyah")


if __name__ == "__main__":
    main()

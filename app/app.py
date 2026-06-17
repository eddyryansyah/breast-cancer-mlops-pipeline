from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_LABELS = {
    0: "Malignant",
    1: "Benign",
}

SYMPTOM_CHOICES = [
    "A new lump or swelling around the breast, chest, or armpit",
    "Persistent breast or armpit pain that does not go away",
    "Changes in breast size or shape",
    "Skin changes such as dimpling, redness, puckering, or thickening",
    "Nipple discharge outside pregnancy or breastfeeding",
    "Changes in nipple shape, position, or appearance",
]


def normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    app_dir = current_file.parent

    if app_dir.name == "app" and (app_dir.parent / "data").exists():
        return app_dir.parent

    return app_dir


def load_processed_dataset() -> tuple[pd.DataFrame, str]:
    project_root = get_project_root()
    dataset_path = project_root / "data" / "processed" / "breast_cancer_preprocessing.csv"

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        return df, "Repository processed dataset"

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]

    return df, "scikit-learn dataset fallback"


@lru_cache(maxsize=1)
def train_demo_model():
    df, dataset_source = load_processed_dataset()

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

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
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }

    return {
        "model": model_pipeline,
        "df": df,
        "feature_columns": list(X.columns),
        "dataset_source": dataset_source,
        "metrics": metrics,
    }


def assess_symptoms(selected_symptoms: list[str]) -> str:
    selected_count = len(selected_symptoms or [])

    if selected_count == 0:
        return (
            "### Awareness Result\n\n"
            "No warning signs were selected.\n\n"
            "This result does not rule out any medical condition. If you notice unusual or persistent breast changes, "
            "please consider consulting a qualified healthcare professional."
        )

    selected_items = "\n".join([f"- {symptom}" for symptom in selected_symptoms])

    if selected_count <= 2:
        guidance = (
            "Some selected signs may require medical attention, especially if they are new, unusual, or persistent. "
            "Please consider scheduling a consultation with a qualified healthcare professional."
        )
    else:
        guidance = (
            "Multiple selected signs may require medical attention. Please consider consulting a qualified healthcare "
            "professional for proper examination."
        )

    return (
        "### Awareness Result\n\n"
        "Selected signs:\n\n"
        f"{selected_items}\n\n"
        f"**Guidance:** {guidance}\n\n"
        "**Important:** This section does not provide a diagnosis and does not calculate cancer probability."
    )


def predict_sample(sample_index: int):
    bundle = train_demo_model()
    model = bundle["model"]
    df = bundle["df"]
    feature_columns = bundle["feature_columns"]

    sample_index = int(sample_index)
    sample_index = max(0, min(sample_index, len(df) - 1))

    sample = df.iloc[sample_index]
    X_sample = sample[feature_columns].to_frame().T

    prediction = int(model.predict(X_sample)[0])
    probabilities = model.predict_proba(X_sample)[0]

    predicted_label = TARGET_LABELS[prediction]
    actual_label = TARGET_LABELS[int(sample["target"])]
    confidence = float(probabilities[prediction])

    probability_output = {
        "Malignant": float(probabilities[0]),
        "Benign": float(probabilities[1]),
    }

    feature_table = (
        X_sample.T.reset_index()
        .rename(columns={"index": "Feature", sample_index: "Value"})
    )

    result_markdown = (
        "### ML Sample Prediction\n\n"
        f"**Selected sample index:** `{sample_index}`\n\n"
        f"**Predicted class:** `{predicted_label}`\n\n"
        f"**Model confidence for predicted class:** `{confidence:.2%}`\n\n"
        f"**Actual dataset label:** `{actual_label}`\n\n"
        "This prediction is based on numerical tumor feature values from the dataset sample. "
        "It is not based on personal symptoms and must not be interpreted as a medical diagnosis."
    )

    return result_markdown, probability_output, feature_table


def get_model_summary() -> str:
    bundle = train_demo_model()
    metrics = bundle["metrics"]

    metrics_text = "\n".join(
        [f"- **{metric_name}:** {metric_value:.4f}" for metric_name, metric_value in metrics.items()]
    )

    return (
        "### Demo Model Summary\n\n"
        f"- **Dataset source:** {bundle['dataset_source']}\n"
        f"- **Samples:** {len(bundle['df'])}\n"
        f"- **Features:** {len(bundle['feature_columns'])}\n"
        "- **Model:** Random Forest Classifier with StandardScaler pipeline\n"
        "- **Target mapping:** `0 = Malignant`, `1 = Benign`\n\n"
        "**Validation metrics:**\n\n"
        f"{metrics_text}\n\n"
        "The model is trained at app startup using the same core training configuration as the MLflow project."
    )


bundle = train_demo_model()
MAX_SAMPLE_INDEX = len(bundle["df"]) - 1


with gr.Blocks(
    title="Breast Cancer Awareness and Tumor Classification",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# Breast Cancer Awareness and Tumor Classification

This demo combines two parts:

1. **Breast health awareness guidance** for unusual breast changes.
2. **Machine learning sample prediction** using numerical tumor feature samples from the Breast Cancer Wisconsin Diagnostic dataset.

> This application does not provide medical diagnosis. If you notice unusual or persistent breast changes, please consult a qualified healthcare professional.
"""
    )

    with gr.Tab("Breast Health Awareness"):
        gr.Markdown(
            """
## Breast Health Awareness

Select any signs that apply. This section provides non-diagnostic guidance only.
"""
        )

        symptoms = gr.CheckboxGroup(
            choices=SYMPTOM_CHOICES,
            label="Select observed signs",
        )

        assess_button = gr.Button("Review selected signs")
        awareness_output = gr.Markdown()

        assess_button.click(
            fn=assess_symptoms,
            inputs=symptoms,
            outputs=awareness_output,
        )

    with gr.Tab("ML Sample Prediction"):
        gr.Markdown(
            """
## ML Sample Prediction

Choose a dataset sample index. The model will classify the selected numerical tumor feature sample as benign or malignant.
"""
        )

        sample_index = gr.Slider(
            minimum=0,
            maximum=MAX_SAMPLE_INDEX,
            value=0,
            step=1,
            label="Dataset sample index",
        )

        predict_button = gr.Button("Predict selected sample")
        prediction_output = gr.Markdown()
        probability_output = gr.Label(label="Class probabilities")
        feature_output = gr.Dataframe(label="Selected sample feature values")

        predict_button.click(
            fn=predict_sample,
            inputs=sample_index,
            outputs=[prediction_output, probability_output, feature_output],
        )

    with gr.Tab("Model Info"):
        gr.Markdown(get_model_summary())

    gr.Markdown(
        """
---

**Responsible use:** This demo is designed for education, portfolio demonstration, and health awareness support. It should not be used as a substitute for professional medical examination, diagnosis, or treatment.
"""
    )


if __name__ == "__main__":
    demo.launch()

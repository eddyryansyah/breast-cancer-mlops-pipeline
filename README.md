# Breast Cancer MLOps Pipeline

[![MLOps Pipeline CI](https://github.com/eddyryansyah/breast-cancer-mlops-pipeline/actions/workflows/main.yml/badge.svg)](https://github.com/eddyryansyah/breast-cancer-mlops-pipeline/actions/workflows/main.yml)
[![Python](https://img.shields.io/badge/Python-3.12.14-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-blue)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Model%20Serving-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning Operations (MLOps) project for breast cancer awareness and tumor classification. This project combines a public-facing health awareness concept with a machine learning pipeline built using the Breast Cancer Wisconsin Diagnostic dataset.

The project demonstrates data preprocessing, automated testing, production model training, MLflow experiment tracking, CI/CD automation with GitHub Actions, container publishing to Docker Hub and GHCR, Docker-based model serving, monitoring with Prometheus and Grafana, and automated Live Demo deployment to Hugging Face Spaces.

## Purpose

This project is designed with two goals:

1. **Public awareness:** help users understand that certain breast changes may require medical consultation.
2. **Machine learning demonstration:** classify numerical tumor feature samples as benign or malignant using a trained machine learning model.

The live demo includes a simple awareness section for users and a machine learning prediction section based on sample numerical tumor data.

## Important Note

This project does not provide a medical diagnosis.

The awareness section is intended to encourage early consultation when users notice unusual breast changes. The machine learning model does not analyze personal symptoms directly. It classifies numerical tumor feature samples from the Breast Cancer Wisconsin Diagnostic dataset.

If users notice unusual breast changes, they should consult a qualified healthcare professional.

## Project Links

| Resource                  | Link                                                                                                     |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| GitHub Repository         | https://github.com/eddyryansyah/breast-cancer-mlops-pipeline                                             |
| Docker Hub Image          | https://hub.docker.com/r/eddyryansyah/breast-cancer-mlops-pipeline                                       |
| GHCR Image                | https://github.com/eddyryansyah/breast-cancer-mlops-pipeline/pkgs/container/breast-cancer-mlops-pipeline |
| DagsHub / MLflow Tracking | https://dagshub.com/eddyryansyah/breast-cancer-mlops-pipeline                                            |
| Live Demo                 | https://huggingface.co/spaces/eddyryansyah/breast-cancer-mlops-demo                                      |

## Project Overview

This repository contains an end-to-end MLOps workflow for breast cancer tumor classification.

The machine learning component predicts whether a tumor sample is benign or malignant based on numerical cell nucleus features. These features are not symptom inputs. They are structured numerical measurements from the dataset.

The public-facing demo has two parts:

1. **Symptom awareness guidance**
   - Users answer simple awareness questions about unusual breast changes.
   - The system provides a non-diagnostic recommendation to seek medical consultation if concerning signs are selected.

2. **ML classification demo**
   - Users select a sample record from the dataset.
   - The model predicts whether the selected sample is benign or malignant.
   - The demo displays prediction confidence and explains that the model works on numerical tumor features, not personal symptoms.

## MLOps Workflow

```text
Raw Dataset
    ↓
Preprocessing Pipeline
    ↓
Processed Dataset
    ↓
GitHub Actions CI/CD
    ↓
Automated Tests
    ↓
Production Model Training
    ↓
MLflow Tracking + Canonical Model Artifact
    ├──→ MLflow Docker Build
    │        ↓
    │    Docker Hub + GHCR
    │        ↓
    │    Model Serving
    │        ↓
    │    Prometheus + Grafana Monitoring
    │
    └──→ Hugging Face Space Deployment
             ↓
         Live Demo
```

## Tech Stack

- Python 3.12.14
- pandas
- NumPy
- scikit-learn
- MLflow
- GitHub Actions
- Docker
- Docker Hub
- GitHub Container Registry (GHCR)
- Prometheus
- Grafana
- Hugging Face Spaces

## Project Structure

```text
.
├── .github/
│   └── workflows/              # CI/CD, testing, container publishing, and deployment
├── app/                        # Hugging Face Gradio demo application
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed dataset
├── docs/
│   └── images/                 # MLflow and monitoring evidence
├── mlproject/                  # Production MLflow Project and model training
├── monitoring/                 # Prometheus and Grafana monitoring setup
├── preprocessing/              # Data preprocessing notebook and script
├── tests/                      # Automated test suite
├── training/                   # Baseline training and model tuning scripts
├── .python-version             # Python version configuration
├── LICENSE
└── README.md
```

## Dataset

This project uses the [Breast Cancer Wisconsin Diagnostic dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), accessed through scikit-learn's `load_breast_cancer` utility.

The dataset is originally available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic).

The dataset contains numerical measurements computed from digitized images of breast mass samples and is used to classify tumor samples as either malignant or benign.

The dataset is stored in this repository in two forms:

| Dataset           | Path                                             |
| ----------------- | ------------------------------------------------ |
| Raw dataset       | `data/raw/breast_cancer_raw.csv`                 |
| Processed dataset | `data/processed/breast_cancer_preprocessing.csv` |

The preprocessing workflow is available in:

```text
preprocessing/
├── automate_preprocessing.py
└── preprocessing.ipynb
```

## Awareness Concept

The live demo includes a breast health awareness section.

This section is not a diagnostic tool. It is designed to encourage users to seek medical consultation when they notice unusual breast changes, such as:

- A lump or swelling around the breast, chest, or armpit
- Changes in breast skin, such as dimpling or redness
- Changes in breast size or shape
- Nipple discharge outside pregnancy or breastfeeding
- Changes in nipple shape or appearance
- Breast or armpit pain that does not go away

The output of this section should be written as guidance, for example:

```text
Some selected signs may require medical attention. Please consider consulting a qualified healthcare professional for proper examination.
```

It should not be written as:

```text
You have breast cancer.
```

or:

```text
Your cancer probability is 90%.
```

## Machine Learning Model

The machine learning model uses a Random Forest classifier wrapped in a scikit-learn pipeline.

The model input is not symptom data. It uses numerical tumor features from the dataset.

The training pipeline includes:

- Train-test split with stratification
- Standard scaling
- Random Forest classification
- Metric evaluation
- MLflow parameter logging
- MLflow metric logging
- MLflow model artifact logging

Current validation metrics:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9561 |
| Precision | 0.9589 |
| Recall    | 0.9722 |
| F1-score  | 0.9655 |

### Model Source of Truth

The repository separates experimentation from the production training pipeline.

- `training/modelling.py` is used for baseline model experimentation and MLflow autologging.
- `training/modelling_tuning.py` is used for hyperparameter tuning and experiment tracking.
- `mlproject/modelling.py` is the production training entry point used by the MLflow Project and GitHub Actions workflow.
- The MLflow model logged under the `model` artifact path is the canonical model artifact used for production deployment.
- The same canonical MLflow model artifact is used to build the model-serving Docker image and is deployed to the Hugging Face Space.
- The Hugging Face Live Demo loads the deployed artifact using `mlflow.sklearn.load_model()` and does not retrain a separate model.
- `model.joblib` is stored as a supplementary training artifact and is not used as the deployment source.

The production model lifecycle is:

```text
data/processed/breast_cancer_preprocessing.csv
        ↓
GitHub Actions CI/CD
        ↓
mlproject/modelling.py
        ↓
MLflow model artifact (`model`)
        ├──→ MLflow Docker build
        │        ↓
        │    Docker Hub + GHCR
        │        ↓
        │    Model serving
        │        ↓
        │    Inference and monitoring
        │
        └──→ Hugging Face Space deployment
                 ↓
             Live Demo
```

## Running the MLflow Project Locally

Install dependencies:

```bash
python -m pip install -r mlproject/requirements.txt
```

Run the MLflow Project:

```bash
python -m mlflow run mlproject --env-manager=local
```

The MLflow Project reads the processed dataset from:

```text
data/processed/breast_cancer_preprocessing.csv
```

## MLflow Tracking

MLflow is used to track:

- Model parameters
- Evaluation metrics
- Model artifacts
- Input example
- Model signature

The hyperparameter-tuning workflow in `training/modelling_tuning.py` logs experiment results to the following DagsHub repository:

```text
https://dagshub.com/eddyryansyah/breast-cancer-mlops-pipeline
```

MLflow evidence screenshots are available in:

```text
docs/images/mlflow/
```

### MLflow Dashboard

<img src="docs/images/mlflow/mlflow-dashboard.jpg" alt="MLflow Dashboard" width="800">

### MLflow Artifacts

<img src="docs/images/mlflow/mlflow-artifacts.jpg" alt="MLflow Artifacts" width="800">

## CI/CD Pipeline

The CI/CD workflow is defined in:

```text
.github/workflows/main.yml
```

The workflow performs the following steps:

1. Runs the automated test suite before the production training job.
2. Sets up Python 3.12.14 for the production pipeline.
3. Installs the MLflow project and deployment dependencies.
4. Runs the MLflow Project to train the production model.
5. Finds the canonical MLflow model artifact.
6. Prepares the Hugging Face Space deployment files using the same canonical model artifact.
7. Uploads the MLflow training artifacts as GitHub Actions artifacts.
8. Builds the model-serving Docker image from the canonical MLflow model.
9. Pushes the Docker image to Docker Hub and GitHub Container Registry (GHCR) on non-pull-request events.
10. Deploys the canonical MLflow model and Live Demo files to Hugging Face Spaces on non-pull-request events.

Pull request workflows validate testing, training, model packaging, and deployment preparation without publishing containers or modifying the public Hugging Face Space. After changes reach `main`, the workflow publishes the production container images and deploys the Live Demo automatically.

## Docker Image

The model serving image is published to both GitHub Container Registry (GHCR) and Docker Hub.

GHCR:

```text
ghcr.io/eddyryansyah/breast-cancer-mlops-pipeline:latest
```

Docker Hub:

```text
eddyryansyah/breast-cancer-mlops-pipeline:latest
```

Docker Hub page:

```text
https://hub.docker.com/r/eddyryansyah/breast-cancer-mlops-pipeline
```

To run the GHCR image as the model-serving container:

```bash
docker pull ghcr.io/eddyryansyah/breast-cancer-mlops-pipeline:latest

docker run -d \
  --name breast-cancer-serving \
  -p 5001:8080 \
  ghcr.io/eddyryansyah/breast-cancer-mlops-pipeline:latest
```

The image is produced through the GitHub Actions workflow using MLflow Docker build.

## Monitoring

The monitoring setup is available in:

```text
monitoring/
```

It includes:

| File                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `prometheus_exporter.py` | Exposes custom model serving metrics                   |
| `prometheus.yml`         | Defines Prometheus scrape configuration                |
| `docker-compose.yml`     | Runs Prometheus and Grafana locally                    |
| `inference.py`           | Sends inference requests to the model serving endpoint |

The monitoring workflow tracks model serving behavior such as:

- Model serving availability
- Inference request count
- Successful inference count
- Failed inference count
- Last inference latency
- Prediction batch size
- Payload feature count
- Prediction class distribution

## Monitoring Evidence

Monitoring evidence screenshots are available in:

```text
docs/images/monitoring/
```

### Docker Model Serving

<img src="docs/images/monitoring/docker-serving.jpg" alt="Docker Model Serving" width="800">

### Prometheus Metrics

<img src="docs/images/monitoring/prometheus-metrics.jpg" alt="Prometheus Metrics" width="800">

### Prometheus Targets

<img src="docs/images/monitoring/prometheus-targets.jpg" alt="Prometheus Targets" width="800">

### Grafana Dashboard

<img src="docs/images/monitoring/grafana-dashboard.jpg" alt="Grafana Dashboard" width="800">

### Grafana Alert Rules

<img src="docs/images/monitoring/grafana-alert-rules.jpg" alt="Grafana Alert Rules" width="800">

## Live Demo

The live demo is available on Hugging Face Spaces:

```text
https://huggingface.co/spaces/eddyryansyah/breast-cancer-mlops-demo
```

The Live Demo uses the canonical MLflow model artifact produced by the production training pipeline. The same model artifact is used to build the production Docker image and is automatically deployed to Hugging Face Spaces through GitHub Actions.

The application loads the deployed model using `mlflow.sklearn.load_model()` and does not perform separate model training at application startup.

Demo sections:

### 1. Breast Health Awareness

Users answer simple awareness questions about unusual breast changes. The system gives non-diagnostic guidance to consult a healthcare professional if concerning signs are selected.

### 2. ML Classification Demo

Users select sample tumor records from the processed dataset. The canonical MLflow model predicts:

- Benign
- Malignant

The demo displays prediction confidence and explains that the model works on numerical tumor features, not personal symptoms.

## Responsible Use

This project uses a public medical dataset for machine learning education, portfolio demonstration, and awareness-oriented application design.

The awareness section is intended to encourage medical consultation when users notice unusual breast changes. The machine learning model is trained on numerical tumor feature samples and is not clinically validated for direct diagnosis.

The system should not be used as a substitute for professional medical examination, diagnosis, or treatment.

## Author

Developed by:

**Eddy Ryansyah**<br>
GitHub: [@eddyryansyah](https://github.com/eddyryansyah)

## License

Copyright (c) 2026 Eddy Ryansyah.

This project is licensed under the MIT License.

This project is published for portfolio, educational, and demonstration purposes. You are allowed to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software under the terms of the MIT License.

See the [LICENSE](LICENSE) file for more details.

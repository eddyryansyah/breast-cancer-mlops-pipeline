import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server


MODEL_URL = "http://127.0.0.1:5001/invocations"
DATASET_PATH = "../Membangun_model/breast_cancer_preprocessing.csv"
EXPORTER_PORT = 8000
REQUEST_INTERVAL_SECONDS = 2
SAMPLE_LIMIT = 10


inference_requests_total = Counter(
    "model_inference_requests_total",
    "Total inference requests sent to the model serving endpoint."
)

inference_success_total = Counter(
    "model_inference_success_total",
    "Total successful inference requests."
)

inference_failure_total = Counter(
    "model_inference_failure_total",
    "Total failed inference requests."
)

prediction_zero_total = Counter(
    "model_prediction_zero_total",
    "Total predictions with class 0."
)

prediction_one_total = Counter(
    "model_prediction_one_total",
    "Total predictions with class 1."
)

inference_latency_seconds = Histogram(
    "model_inference_latency_seconds",
    "Inference request latency in seconds."
)

inference_latency_last_seconds = Gauge(
    "model_inference_latency_last_seconds",
    "Last inference request latency in seconds."
)

model_serving_up = Gauge(
    "model_serving_up",
    "Model serving availability status. 1 means up, 0 means down."
)

last_status_code = Gauge(
    "model_inference_last_status_code",
    "Last HTTP status code returned by the model serving endpoint."
)

prediction_batch_size = Gauge(
    "model_prediction_batch_size",
    "Number of rows sent in the latest inference request."
)

prediction_positive_ratio = Gauge(
    "model_prediction_positive_ratio",
    "Ratio of class 1 predictions in the latest inference response."
)

prediction_negative_ratio = Gauge(
    "model_prediction_negative_ratio",
    "Ratio of class 0 predictions in the latest inference response."
)

payload_feature_count = Gauge(
    "model_payload_feature_count",
    "Number of features sent to the model serving endpoint."
)

last_request_timestamp = Gauge(
    "model_inference_last_request_timestamp",
    "Unix timestamp of the latest inference request."
)


def load_sample() -> pd.DataFrame:
    dataset_file = Path(DATASET_PATH)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_file}")

    df = pd.read_csv(dataset_file)

    if "target" in df.columns:
        df = df.drop(columns=["target"])

    return df.head(SAMPLE_LIMIT)


def send_prediction_request(sample_df: pd.DataFrame):
    payload = {
        "dataframe_split": {
            "columns": sample_df.columns.tolist(),
            "data": sample_df.values.tolist(),
        }
    }

    request = urllib.request.Request(
        url=MODEL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.time()

    with urllib.request.urlopen(request, timeout=30) as response:
        latency = time.time() - start_time
        response_body = response.read().decode("utf-8")
        return response.status, response_body, latency


def update_metrics():
    sample_df = load_sample()

    inference_requests_total.inc()
    prediction_batch_size.set(len(sample_df))
    payload_feature_count.set(sample_df.shape[1])
    last_request_timestamp.set(time.time())

    try:
        status_code, response_body, latency = send_prediction_request(sample_df)

        last_status_code.set(status_code)
        inference_latency_seconds.observe(latency)
        inference_latency_last_seconds.set(latency)

        if status_code == 200:
            inference_success_total.inc()
            model_serving_up.set(1)

            response_json = json.loads(response_body)
            predictions = response_json.get("predictions", [])

            zero_count = sum(1 for prediction in predictions if prediction == 0)
            one_count = sum(1 for prediction in predictions if prediction == 1)
            total_predictions = len(predictions)

            prediction_zero_total.inc(zero_count)
            prediction_one_total.inc(one_count)

            if total_predictions > 0:
                prediction_negative_ratio.set(zero_count / total_predictions)
                prediction_positive_ratio.set(one_count / total_predictions)

            print(
                f"status={status_code}, latency={latency:.4f}s, "
                f"predictions={predictions}"
            )
        else:
            inference_failure_total.inc()
            model_serving_up.set(0)
            print(f"Unexpected status code: {status_code}")

    except Exception as error:
        inference_failure_total.inc()
        model_serving_up.set(0)
        last_status_code.set(0)
        print(f"Inference failed: {error}")


def main():
    print(f"Starting Prometheus exporter on port {EXPORTER_PORT}")
    print(f"Model URL: {MODEL_URL}")
    print(f"Dataset path: {DATASET_PATH}")

    start_http_server(EXPORTER_PORT)

    while True:
        update_metrics()
        time.sleep(REQUEST_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

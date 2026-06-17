import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd


def load_sample(dataset_path: str, limit: int) -> pd.DataFrame:
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_file}")

    df = pd.read_csv(dataset_file)

    if "target" in df.columns:
        df = df.drop(columns=["target"])

    return df.head(limit)


def send_prediction_request(url: str, sample_df: pd.DataFrame):
    payload = {
        "dataframe_split": {
            "columns": sample_df.columns.tolist(),
            "data": sample_df.values.tolist(),
        }
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        response_body = response.read().decode("utf-8")
        return response.status, response_body


def main():
    parser = argparse.ArgumentParser(
        description="Inference script untuk model Breast Cancer MLflow Serving."
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:5001/invocations",
        help="URL endpoint invocations model serving.",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../Membangun_model/breast_cancer_preprocessing.csv",
        help="Path dataset hasil preprocessing.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Jumlah data sample yang dikirim dalam satu request.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Jumlah pengulangan request inference.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Jeda antar request dalam detik.",
    )

    args = parser.parse_args()

    sample_df = load_sample(args.dataset_path, args.limit)

    print("Inference configuration")
    print(f"URL          : {args.url}")
    print(f"Dataset path : {args.dataset_path}")
    print(f"Sample shape : {sample_df.shape}")
    print(f"Repeat       : {args.repeat}")

    for request_number in range(1, args.repeat + 1):
        status_code, response_body = send_prediction_request(args.url, sample_df)

        print(f"\nRequest ke-{request_number}")
        print(f"Status code : {status_code}")
        print(f"Response    : {response_body}")

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()

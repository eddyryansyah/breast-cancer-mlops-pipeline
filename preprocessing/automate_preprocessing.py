import argparse
from pathlib import Path

import pandas as pd


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Melakukan preprocessing otomatis sesuai tahapan yang dilakukan
    pada notebook eksperimen.

    Tahapan:
    1. Memuat dataset raw.
    2. Menghapus data duplikat.
    3. Menstandarkan nama kolom menjadi lowercase dan snake_case.
    4. Memastikan kolom target bertipe integer.
    5. Menyimpan dataset hasil preprocessing.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_file}")

    df = pd.read_csv(input_file)

    df_preprocessed = df.copy()
    df_preprocessed = df_preprocessed.drop_duplicates()

    df_preprocessed.columns = (
        df_preprocessed.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if "target" not in df_preprocessed.columns:
        raise ValueError("Kolom target tidak ditemukan pada dataset.")

    df_preprocessed["target"] = df_preprocessed["target"].astype(int)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_preprocessed.to_csv(output_file, index=False)

    return df_preprocessed


def main():
    parser = argparse.ArgumentParser(
        description="Automasi preprocessing dataset Breast Cancer untuk submission MSML Dicoding."
    )

    parser.add_argument(
        "--input",
        default="data/raw/breast_cancer_raw.csv",
        help="Path menuju dataset raw."
    )

    parser.add_argument(
        "--output",
        default="data/processed/breast_cancer_preprocessing.csv",
        help="Path output dataset hasil preprocessing."
    )

    args = parser.parse_args()

    df_preprocessed = preprocess_data(args.input, args.output)

    print("Preprocessing otomatis berhasil dijalankan.")
    print(f"Dataset input: {args.input}")
    print(f"Dataset output: {args.output}")
    print(f"Jumlah baris dan kolom: {df_preprocessed.shape}")
    print("Distribusi target:")
    print(df_preprocessed["target"].value_counts())


if __name__ == "__main__":
    main()

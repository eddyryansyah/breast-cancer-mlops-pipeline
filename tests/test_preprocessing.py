import pandas as pd
import pytest

from preprocessing.automate_preprocessing import preprocess_data


def test_preprocess_data_success(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "processed.csv"

    raw_df = pd.DataFrame(
        {
            "Mean Radius": [10.0, 10.0, 12.0],
            "Mean Texture": [15.0, 15.0, 18.0],
            "Target": [1.0, 1.0, 0.0],
        }
    )
    raw_df.to_csv(input_path, index=False)

    result = preprocess_data(str(input_path), str(output_path))

    assert output_path.exists()
    assert result.shape == (2, 3)
    assert list(result.columns) == [
        "mean_radius",
        "mean_texture",
        "target",
    ]
    assert pd.api.types.is_integer_dtype(result["target"])

    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(
    result.reset_index(drop=True),
    saved_df,
)


def test_preprocess_data_missing_input(tmp_path):
    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "processed.csv"

    with pytest.raises(FileNotFoundError, match="File input tidak ditemukan"):
        preprocess_data(str(input_path), str(output_path))


def test_preprocess_data_missing_target(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"

    raw_df = pd.DataFrame(
        {
            "Mean Radius": [10.0, 12.0],
            "Mean Texture": [15.0, 18.0],
        }
    )
    raw_df.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Kolom target tidak ditemukan"):
        preprocess_data(str(input_path), str(output_path))

    assert not output_path.exists()

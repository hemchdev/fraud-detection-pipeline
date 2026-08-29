"""
Basic unit tests for src/features/build_features.py

Run all tests from the project's top folder with:
    pytest
"""

import pandas as pd
from src.features.build_features import add_features, get_feature_columns


def test_add_features_creates_new_columns():
    df = pd.DataFrame(
        {
            "Time": [0, 3600, 7200],
            "Amount": [10.0, 100.0, 1000.0],
            "Class": [0, 0, 1],
        }
    )
    out = add_features(df)
    assert "log_amount" in out.columns
    assert "hour_of_day" in out.columns
    assert out["hour_of_day"].tolist() == [0, 1, 2]


def test_get_feature_columns_excludes_target_and_raw_cols():
    df = pd.DataFrame(
        {
            "Time": [0],
            "Amount": [1.0],
            "Class": [0],
            "V1": [0.5],
            "log_amount": [0.1],
            "hour_of_day": [0],
        }
    )
    cols = get_feature_columns(df)
    assert "Class" not in cols
    assert "Time" not in cols
    assert "Amount" not in cols
    assert "V1" in cols
    assert "log_amount" in cols

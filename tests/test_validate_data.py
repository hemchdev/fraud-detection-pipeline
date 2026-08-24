"""
Tests for src/pipeline/validate_data.py

Run all tests from the project's top folder with:
    pytest
"""
import pandas as pd
import pytest

from src.pipeline.validate_data import validate, DataValidationError


def _good_df():
    return pd.DataFrame({
        "Time": [0, 100, 200],
        "Amount": [10.0, 20.0, 5.0],
        "Class": [0, 0, 1],
    })


def test_validate_passes_on_good_data():
    validate(_good_df())  # should not raise


def test_validate_fails_on_missing_values():
    df = _good_df()
    df.loc[0, "Amount"] = None
    with pytest.raises(DataValidationError):
        validate(df)


def test_validate_fails_on_negative_amount():
    df = _good_df()
    df.loc[0, "Amount"] = -5.0
    with pytest.raises(DataValidationError):
        validate(df)


def test_validate_fails_with_no_fraud_examples():
    df = _good_df()
    df["Class"] = 0
    with pytest.raises(DataValidationError):
        validate(df)

"""
Data validation — a "sanity check" step that runs BEFORE we train
anything. In a real production system, silently training on broken
data (missing values, an impossible number, a duplicate column) is
exactly how bad models sneak into production without anyone noticing
until it's too late.

This is a hand-rolled version of what a real tool like Great
Expectations does professionally — same idea, far fewer moving parts,
so you can see exactly what gets checked and why.

Run it with:
    python -m src.pipeline.validate_data
"""

import pandas as pd

from src.utils.config import TARGET_COLUMN


class DataValidationError(Exception):
    """Raised when the data fails one of our sanity checks."""


def validate(df: pd.DataFrame) -> None:
    checks = []

    # 1. The dataframe isn't empty
    checks.append(("has_rows", len(df) > 0))

    # 2. The answer-key column exists, and only ever contains 0 or 1
    checks.append(("target_column_exists", TARGET_COLUMN in df.columns))
    if TARGET_COLUMN in df.columns:
        valid_classes = set(df[TARGET_COLUMN].unique()) <= {0, 1}
        checks.append(("target_is_binary", valid_classes))

    # 3. No missing values anywhere — a model can't handle a blank cell
    checks.append(("no_missing_values", df.isnull().sum().sum() == 0))

    # 4. A transaction amount can never be negative
    if "Amount" in df.columns:
        checks.append(("amount_non_negative", (df["Amount"] >= 0).all()))

    # 5. There's at least SOME fraud to learn from — otherwise training
    #    is pointless (and probably means something upstream is broken)
    if TARGET_COLUMN in df.columns:
        checks.append(("has_fraud_examples", df[TARGET_COLUMN].sum() > 0))

    failed = [name for name, passed in checks if not passed]
    if failed:
        raise DataValidationError(f"Data failed validation checks: {failed}")

    print(f"All {len(checks)} data validation checks passed.")


if __name__ == "__main__":
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    validate(df)

"""
Loads the raw CSV and splits it into train / validation / test sets.

IMPORTANT IDEA: we split by TIME, not randomly.

Fraud detection is really a "predict the future" problem. In real life,
your model will only ever see PAST transactions before it has to score
a NEW one. So we train on the oldest transactions and test on the
newest ones — this mimics reality.

If we split randomly instead, some "future" transactions could end up
in the training set, and the model could unknowingly learn from
patterns it should never have seen yet. That's called TIME LEAKAGE,
and it makes your model look better in testing than it will ever
perform in real life. Avoiding it is one of the most important habits
in this project.

Run it directly to see the split sizes:
    python -m src.data.load_data
"""

import pandas as pd

from src.utils.config import RAW_DATA_PATH, TARGET_COLUMN


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def time_ordered_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
):
    """
    Splits a dataframe into train / val / test, in TIME order.
    Example with train_frac=0.7, val_frac=0.15:
      - first 70% of rows (by time)  -> train
      - next 15% of rows             -> validation
      - final 15% of rows            -> test
    """
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = load_raw_data()
    train_df, val_df, test_df = time_ordered_split(df)

    print(f"Total rows: {len(df)}")
    print(f"Train: {len(train_df):>6} rows  (fraud: {train_df[TARGET_COLUMN].sum()})")
    print(f"Val:   {len(val_df):>6} rows  (fraud: {val_df[TARGET_COLUMN].sum()})")
    print(f"Test:  {len(test_df):>6} rows  (fraud: {test_df[TARGET_COLUMN].sum()})")

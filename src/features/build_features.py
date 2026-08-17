"""
Turns raw columns into features a model can actually learn from.

- Amount: transaction amounts range from a few rupees to thousands, and
  that huge range confuses some models. We log-transform it (log_amount)
  which squashes big numbers down closer to small ones, without losing
  the "bigger is bigger" pattern.

- Time: the raw column is "seconds since the very first transaction in
  the dataset" — not something a model can find a pattern in. We turn
  it into hour_of_day (0-23) instead, since fraud often clusters at
  certain hours.

- V1-V28: Kaggle already ran these through PCA (a math technique that
  anonymizes the original bank data while keeping its patterns intact),
  so we use them exactly as given — no extra work needed.
"""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])       # log1p = log(1 + x), safe for x=0
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Every column the MODEL is allowed to see — i.e. everything
    except the answer key (Class) and the two raw columns we already
    turned into better features (Time, Amount)."""
    exclude = {"Class", "Time", "Amount"}
    return [c for c in df.columns if c not in exclude]

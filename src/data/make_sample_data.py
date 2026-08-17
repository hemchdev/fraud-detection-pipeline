"""
This script creates a FAKE, small version of the credit card fraud
dataset — same column names, same rough shape, same rare-fraud pattern
as the real one.

Why? So you can test that ALL your code works end-to-end in a few
seconds, before you spend time downloading the real 284,807-row file
from Kaggle. Once the real file is in data/raw/creditcard.csv, this
script's output just gets overwritten and everything else keeps working
unchanged.

Run it from the project's top folder with:
    python -m src.data.make_sample_data
"""
import os
import numpy as np
import pandas as pd

from src.utils.config import RAW_DATA_PATH, RANDOM_SEED


def make_sample_data(n_rows: int = 20000, fraud_rate: float = 0.0017):
    rng = np.random.default_rng(RANDOM_SEED)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_normal = n_rows - n_fraud

    def make_rows(n, is_fraud):
        data = {"Time": rng.integers(0, 172792, size=n)}
        for i in range(1, 29):
            # Fraud rows get their values nudged so a model has SOMETHING
            # real to learn from (the real dataset has genuine patterns
            # here — we're just faking a simple version of that).
            shift = 2.5 if is_fraud else 0.0
            data[f"V{i}"] = rng.normal(loc=shift, scale=1.0, size=n)
        data["Amount"] = np.round(rng.exponential(scale=80.0, size=n), 2)
        data["Class"] = is_fraud
        return pd.DataFrame(data)

    df_normal = make_rows(n_normal, 0)
    df_fraud = make_rows(n_fraud, 1)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)

    # shuffle, then re-sort by time (this mimics how real transactions
    # arrive — mixed together but still in time order)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Fake sample data created at: {RAW_DATA_PATH}")
    print(f"   Total rows: {len(df)}")
    print(f"   Fraud rows: {df['Class'].sum()} ({df['Class'].mean() * 100:.3f}%)")


if __name__ == "__main__":
    make_sample_data()

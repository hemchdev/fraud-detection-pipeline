"""
Trains XGBoost — a much more powerful model than our Logistic Regression
baseline — and, more importantly, teaches it to properly handle the fact
that fraud is EXTREMELY rare (in the real dataset: 492 fraud rows out of
284,807 — about 0.17%).

WHY DOES "RARE" BREAK NORMAL TRAINING?
Most ML algorithms try to minimize their total number of mistakes. If
99.83% of rows are "not fraud", the laziest possible strategy — "always
guess not fraud" — already gets 99.83% of rows right! A model trained
the normal way will drift toward that lazy strategy, because it's
mathematically rewarded for it. We have to explicitly tell the model
"a mistake on a fraud row matters far more than a mistake on a normal
row."

TWO WAYS TO FIX THIS (this script does both, so you can compare them
head-to-head in compare_models.py):

1. scale_pos_weight (a setting INSIDE XGBoost):
   We tell XGBoost directly: "getting a fraud row wrong costs N times
   more than getting a normal row wrong," where
       N = (number of normal rows) / (number of fraud rows)
   XGBoost then leans harder into getting fraud rows right during
   training. No new data is created — we just change the "penalty."

2. SMOTE (Synthetic Minority Over-sampling TEchnique):
   Instead of changing the penalty, we create new, synthetic fraud
   examples by interpolating between real fraud rows that look similar
   to each other (imagine plotting two real fraud points and picking a
   point on the line between them). This gives the model more fraud
   examples to learn from. Applied ONLY to the training set — never to
   validation/test data, because those must stay 100% realistic.

Run it with:
    python -m src.models.train_xgboost
"""

import os
import joblib
from xgboost import XGBClassifier

from src.data.load_data import load_raw_data, time_ordered_split
from src.features.build_features import add_features, get_feature_columns
from src.utils.config import MODELS_DIR, TARGET_COLUMN, RANDOM_SEED


def _get_train_xy():
    df = load_raw_data()
    df = add_features(df)
    train_df, _, _ = time_ordered_split(df)
    feature_cols = get_feature_columns(df)
    return train_df[feature_cols], train_df[TARGET_COLUMN]


def train_with_scale_pos_weight():
    X_train, y_train = _get_train_xy()

    n_normal = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    scale_pos_weight = n_normal / max(n_fraud, 1)
    print(f"Training rows: {n_normal} normal, {n_fraud} fraud")
    print(f"scale_pos_weight = {n_normal} / {n_fraud} = {scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # tracks PR-AUC while training — matches how we grade it later
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "xgboost_scale_pos_weight.joblib")
    joblib.dump(model, path)
    print(f"Model trained and saved to: {path}")
    return model


def train_with_smote():
    from imblearn.over_sampling import SMOTE

    X_train, y_train = _get_train_xy()
    n_fraud = int((y_train == 1).sum())

    if n_fraud < 2:
        print(
            "Not enough fraud rows in the training set to run SMOTE "
            "(need at least 2). Try regenerating sample data with more "
            "rows, or use the real Kaggle dataset. Skipping."
        )
        return None

    print(
        f"Before SMOTE — fraud rows: {n_fraud}, normal rows: {int((y_train == 0).sum())}"
    )

    # k_neighbors must be smaller than the number of fraud rows we have,
    # so this stays safe even on a tiny practice dataset.
    k_neighbors = min(5, n_fraud - 1)
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"After SMOTE  — fraud rows: {int((y_resampled == 1).sum())}, "
        f"normal rows: {int((y_resampled == 0).sum())}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_resampled, y_resampled)

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "xgboost_smote.joblib")
    joblib.dump(model, path)
    print(f"Model trained and saved to: {path}")
    return model


if __name__ == "__main__":
    print("--- Training XGBoost with scale_pos_weight ---")
    train_with_scale_pos_weight()
    print()
    print("--- Training XGBoost with SMOTE ---")
    train_with_smote()

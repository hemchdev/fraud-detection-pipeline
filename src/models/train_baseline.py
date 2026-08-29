"""
Trains our FIRST model: Logistic Regression.

Why start with something this simple instead of jumping to a fancier
model? Two reasons:

1. It's fast — trains in seconds, so you get feedback immediately and
   can tell right away if your data pipeline is broken.
2. It gives us a BASELINE score. Later, when we build a fancier model
   (XGBoost), we'll know exactly how much better it really is — instead
   of just assuming "fancier automatically means better."

Run it from the project's top folder with:
    python -m src.models.train_baseline
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.load_data import load_raw_data, time_ordered_split
from src.features.build_features import add_features, get_feature_columns
from src.utils.config import MODELS_DIR, TARGET_COLUMN, RANDOM_SEED


def train():
    df = load_raw_data()
    df = add_features(df)
    train_df, val_df, _ = time_ordered_split(df)

    feature_cols = get_feature_columns(df)
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]

    # class_weight="balanced" tells the model: "fraud rows are rare —
    # pay extra attention to the few you see." Without this, a lazy
    # model could get 99.8% accuracy just by guessing "not fraud" on
    # every single transaction, which would be useless in real life.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "baseline_logreg.joblib")
    joblib.dump(model, model_path)
    print(f"Model trained and saved to: {model_path}")

    return model, X_val, y_val


if __name__ == "__main__":
    train()

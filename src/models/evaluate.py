"""
Scores a saved model on the held-out TEST set, using metrics that
actually matter for fraud detection.

Why not just look at "accuracy"? Because only ~0.17% of transactions
are fraud. A model that predicts "not fraud" for EVERY single
transaction would score 99.8% "accurate" — and be completely useless.
Instead we look at:

- PR-AUC (Precision-Recall AUC): our main scorecard. Higher = better
  at catching fraud without drowning in false alarms. Unlike plain
  accuracy, it isn't fooled by class imbalance.
- ROC-AUC: a second, more commonly-known scorecard, for reference.
- Recall @ 90% precision: "if we insist on being right 9 times out of
  10 whenever we flag a transaction as fraud, how much of the ACTUAL
  fraud do we still catch?" This is the number a real risk team cares
  about most, because false alarms annoy customers.
- Confusion matrix: the raw counts behind all of the above.

Run it from the project's top folder with:
    python -m src.models.evaluate
"""

import os
import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)

from src.data.load_data import load_raw_data, time_ordered_split
from src.features.build_features import add_features, get_feature_columns
from src.utils.config import MODELS_DIR, TARGET_COLUMN


def recall_at_precision(y_true, y_scores, target_precision: float = 0.9) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    valid = precisions[:-1] >= target_precision
    if not valid.any():
        return 0.0
    return recalls[:-1][valid].max()


def evaluate(model_path: str = None) -> dict:
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "baseline_logreg.joblib")
    model = joblib.load(model_path)

    df = load_raw_data()
    df = add_features(df)
    _, _, test_df = time_ordered_split(df)
    feature_cols = get_feature_columns(df)
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    recall_90 = recall_at_precision(y_test, y_scores, target_precision=0.9)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 50)
    print(f"PR-AUC:                 {pr_auc:.4f}")
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"Recall @ 90% precision: {recall_90:.4f}")
    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(cm)
    print("=" * 50)

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "recall_at_90_precision": recall_90}


if __name__ == "__main__":
    evaluate()

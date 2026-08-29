"""
Puts every model we've trained so far on the SAME test set and shows
them side by side. This is the "champion vs challenger" pattern real ML
teams use before promoting a new model to production — a new model has
to EARN its promotion with better numbers, not just "feel more
sophisticated" because it's fancier.

Run it with:
    python -m src.models.compare_models
"""

import os
from src.models.evaluate import evaluate
from src.utils.config import MODELS_DIR

MODEL_FILES = {
    "Baseline (Logistic Regression)": "baseline_logreg.joblib",
    "XGBoost + scale_pos_weight": "xgboost_scale_pos_weight.joblib",
    "XGBoost + SMOTE": "xgboost_smote.joblib",
}


def compare():
    results = {}
    for label, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"Skipping '{label}' — {filename} not found yet. Train it first.")
            continue
        print(f"\nEvaluating: {label}")
        results[label] = evaluate(model_path=path)

    if not results:
        print(
            "\nNo trained models found. Run train_baseline.py and "
            "train_xgboost.py first."
        )
        return

    print("\n" + "=" * 72)
    print(f"{'Model':38} {'PR-AUC':>8} {'ROC-AUC':>8} {'Recall@90P':>11}")
    print("-" * 72)
    for label, m in results.items():
        print(
            f"{label:38} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} "
            f"{m['recall_at_90_precision']:>11.4f}"
        )
    print("=" * 72)

    # PR-AUC is our PRIMARY metric (see evaluate.py for why) so that's
    # what decides the champion.
    champion = max(results.items(), key=lambda kv: kv[1]["pr_auc"])
    print(f"\nBest model by PR-AUC (our primary metric): {champion[0]}")


if __name__ == "__main__":
    compare()

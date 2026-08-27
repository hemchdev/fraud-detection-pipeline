"""
This is the training script that runs INSIDE Azure, not on your laptop.

It's deliberately self-contained and takes its data path and output
folder as command-line arguments, because a cloud job never sees your
laptop's file system — Azure ML mounts the registered Data Asset
somewhere inside the container, and expects the trained model written
to a different folder that IT controls (so it can find it afterward
and let you download it).

It reuses the exact same feature engineering and time-based split
logic from Phases 1-3 (src/features/build_features.py,
src/data/load_data.py) and the same XGBoost + scale_pos_weight
approach from Phase 2 — the only thing that's different is WHERE the
data comes from and WHERE the model goes.

You will not normally type the command below yourself — aml/job.yml
does it for you when you run `az ml job create --file aml/job.yml`.
It's shown here just so you know what actually happens:

    python -m src.cloud.train_job_entry \
        --data_path <path Azure mounted the data asset to> \
        --model_output <folder Azure expects the model in>
"""
import argparse
import os

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

from src.data.load_data import time_ordered_split
from src.features.build_features import add_features, get_feature_columns
from src.models.evaluate import recall_at_precision
from src.pipeline.validate_data import validate
from src.utils.config import TARGET_COLUMN, RANDOM_SEED


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to the CSV (a mounted Azure Data Asset, or a local file for testing).")
    parser.add_argument("--model_output", type=str, required=True,
                         help="Folder to write the trained model into.")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Stage 1: load + validate ---
    df = pd.read_csv(args.data_path)
    validate(df)

    # --- Stage 2: features + time-ordered split ---
    df = add_features(df)
    feature_cols = get_feature_columns(df)
    train_df, _, test_df = time_ordered_split(df)

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    # --- Stage 3: train (same approach as local Phase 2: scale_pos_weight) ---
    n_normal = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    scale_pos_weight = n_normal / max(n_fraud, 1)
    print(f"Training rows: {n_normal} normal, {n_fraud} fraud, scale_pos_weight={scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    # --- Stage 4: evaluate ---
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    recall_90 = recall_at_precision(y_test, y_scores, target_precision=0.9)
    cm = confusion_matrix(y_test, y_pred)

    print(f"PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}  Recall@90P: {recall_90:.4f}")
    print(f"Confusion matrix ([[TN, FP], [FN, TP]]):\n{cm}")

    # Azure ML automatically wires up MLflow tracking inside a job — these
    # calls land in the job's own "Metrics" tab in Azure ML Studio, no
    # extra setup needed. (Run locally for testing, they just log to your
    # local ./mlruns folder like they did in Phase 3.)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_at_90_precision", recall_90)

    # --- Stage 5: save the model where Azure ML expects it ---
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

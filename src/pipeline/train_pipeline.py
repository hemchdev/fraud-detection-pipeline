"""
The full pipeline, in one command: validate -> train every candidate
model -> evaluate each -> log everything to MLflow -> promote the best
one to "champion" (but only if it actually beats whatever was champion
before).

This mirrors, on your laptop, the exact same stages the real Azure ML
pipeline will run in the cloud later (Phase 4). We're running them
locally first so you understand what each stage does before adding
cloud infrastructure on top of it.

WHAT IS MLFLOW, IN PLAIN WORDS?
Every time you train a model, MLflow saves a "run": a little record of
what settings you used, what scores came out, and (if you ask it to)
the model file itself. Do this for every experiment, and instead of
"I think version 3 was pretty good?" scribbled in a notebook, you get a
searchable, comparable history. An "experiment" is just a named folder
of related runs — ours is called "fraud-detection".

Run the pipeline with:
    python -m src.pipeline.train_pipeline

Then browse your results in a web page with:
    mlflow ui
and open http://localhost:5000 in your browser. You'll see every run,
its metrics, and can compare them side by side with checkboxes.
"""

import os
import json
import mlflow

from src.data.load_data import load_raw_data
from src.features.build_features import add_features, get_feature_columns
from src.pipeline.validate_data import validate
from src.models.evaluate import evaluate as evaluate_saved_model
from src.models.train_baseline import train as train_baseline_model
from src.models.train_xgboost import train_with_scale_pos_weight, train_with_smote
from src.utils.config import MODELS_DIR

EXPERIMENT_NAME = "fraud-detection"
REGISTRY_DIR = os.path.join(MODELS_DIR, "registry")
CHAMPION_METADATA_PATH = os.path.join(REGISTRY_DIR, "champion_metadata.json")

# name -> (saved model filename, function that trains it)
CANDIDATES = {
    "baseline_logreg": ("baseline_logreg.joblib", train_baseline_model),
    "xgboost_scale_pos_weight": (
        "xgboost_scale_pos_weight.joblib",
        train_with_scale_pos_weight,
    ),
    "xgboost_smote": ("xgboost_smote.joblib", train_with_smote),
}


def _run_training(train_fn):
    """Some train_* functions return just a model, others return
    (model, X_val, y_val). This normalizes both to just "the model",
    so the pipeline doesn't care which kind it's calling."""
    result = train_fn()
    if isinstance(result, tuple):
        return result[0]
    return result


def _load_current_champion():
    if not os.path.exists(CHAMPION_METADATA_PATH):
        return None
    with open(CHAMPION_METADATA_PATH) as f:
        return json.load(f)


def run_pipeline():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Stage 1: validate ---
    print("--- Stage 1: validate ---")
    df = load_raw_data()
    validate(df)

    # --- Stage 2: preprocess / build features ---
    print("\n--- Stage 2: preprocess ---")
    df = add_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features.")

    # --- Stage 3 & 4: train + evaluate each candidate (each its own MLflow run) ---
    results = {}
    for name, (filename, train_fn) in CANDIDATES.items():
        print(f"\n--- Stage 3/4: train + evaluate '{name}' ---")
        with mlflow.start_run(run_name=name):
            model = _run_training(train_fn)
            if model is None:
                print(f"Skipping '{name}' — training returned no model.")
                continue

            model_path = os.path.join(MODELS_DIR, filename)
            metrics = evaluate_saved_model(model_path=model_path)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("pr_auc", metrics["pr_auc"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
            mlflow.log_metric(
                "recall_at_90_precision", metrics["recall_at_90_precision"]
            )
            mlflow.log_artifact(model_path)

            results[name] = metrics

    if not results:
        print("\nNo models were trained successfully. Nothing to promote.")
        return

    # --- Stage 5: conditionally register the champion ---
    print("\n--- Stage 5: conditionally register ---")
    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_metrics = results[best_name]
    current = _load_current_champion()

    should_promote = current is None or best_metrics["pr_auc"] >= current["pr_auc"]

    print(f"Best model this run: {best_name} (PR-AUC={best_metrics['pr_auc']:.4f})")
    if current:
        print(
            f"Current champion:    {current['model_name']} (PR-AUC={current['pr_auc']:.4f})"
        )

    if should_promote:
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        metadata = {"model_name": best_name, **best_metrics}
        with open(CHAMPION_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Promoted '{best_name}' to champion. See {CHAMPION_METADATA_PATH}")
    else:
        print(
            f"New models did not beat the current champion. "
            f"Champion stays: {current['model_name']}"
        )


if __name__ == "__main__":
    run_pipeline()

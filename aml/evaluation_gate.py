"""
The "evaluation gate" in our CD pipeline: a new model only gets
promoted to champion if it's at least as good as whoever holds that
title today. This is the same idea real ML teams use to stop a worse
model from accidentally reaching production just because it's newer.

Champion state lives in aml/champion_metrics.json — a file committed
to the repo, so it's the same file every workflow run reads, and (if
promoted) updates and commits back.

Run it with:
    python aml/evaluation_gate.py --new-pr-auc 0.87
"""
import argparse
import json
import os

CHAMPION_FILE = os.path.join(os.path.dirname(__file__), "champion_metrics.json")


def load_champion():
    with open(CHAMPION_FILE) as f:
        return json.load(f)


def save_champion(metrics):
    with open(CHAMPION_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-pr-auc", type=float, required=True)
    parser.add_argument("--new-roc-auc", type=float, default=None)
    parser.add_argument("--new-recall", type=float, default=None)
    args = parser.parse_args()

    current = load_champion()
    print(f"Current champion PR-AUC: {current['pr_auc']}")
    print(f"New model PR-AUC:        {args.new_pr_auc}")

    promoted = args.new_pr_auc >= current["pr_auc"]

    if promoted:
        new_metrics = {
            "pr_auc": args.new_pr_auc,
            "roc_auc": (
                args.new_roc_auc if args.new_roc_auc is not None else current.get("roc_auc", 0.0)
            ),
            "recall_at_90_precision": (
                args.new_recall
                if args.new_recall is not None
                else current.get("recall_at_90_precision", 0.0)
            ),
        }
        save_champion(new_metrics)
        print("PROMOTED — new champion saved to aml/champion_metrics.json")
    else:
        print("NOT promoted — the new model did not beat the current champion.")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")


if __name__ == "__main__":
    main()

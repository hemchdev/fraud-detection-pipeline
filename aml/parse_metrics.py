"""
Reads a training job's captured console output and pulls out the
scores, using the exact print format from src/cloud/train_job_entry.py:

    PR-AUC: 0.9876  ROC-AUC: 0.9999  Recall@90P: 0.8888

When run inside GitHub Actions, it writes the result to a special file
(GITHUB_OUTPUT) so the next step in the workflow can read it as
${{ steps.metrics.outputs.pr_auc }}. When run locally (for testing),
it just prints it — so you can try this yourself without any of Azure
or GitHub involved.

Run it with:
    python aml/parse_metrics.py path/to/job_output.log
"""
import argparse
import os
import re
import sys

PATTERN = re.compile(r"PR-AUC:\s*([\d.]+)\s+ROC-AUC:\s*([\d.]+)\s+Recall@90P:\s*([\d.]+)")


def parse(log_text: str):
    match = PATTERN.search(log_text)
    if not match:
        print("Could not find a 'PR-AUC: ...' line in the log output.", file=sys.stderr)
        sys.exit(1)
    pr_auc, roc_auc, recall = (float(x) for x in match.groups())
    return pr_auc, roc_auc, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    args = parser.parse_args()

    with open(args.log_path) as f:
        log_text = f.read()

    pr_auc, roc_auc, recall = parse(log_text)
    print(f"Parsed metrics — PR-AUC: {pr_auc}, ROC-AUC: {roc_auc}, Recall@90P: {recall}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"pr_auc={pr_auc}\n")
            f.write(f"roc_auc={roc_auc}\n")
            f.write(f"recall_at_90_precision={recall}\n")


if __name__ == "__main__":
    main()

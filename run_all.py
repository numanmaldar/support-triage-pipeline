"""
run_all.py

Single entrypoint that runs the entire evaluation pipeline end-to-end.

Usage:
    python run_all.py --demo      # 18-example demo set (recommended for reviewers, <10 min)
    python run_all.py --full      # Full 197-example golden set (slower)

Requires GEMINI_API_KEY to be set in the environment.
"""

import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable


def get_steps(demo: bool):
    if demo:
        golden = "eval/demo_golden_set.jsonl"
        predictions = "data/processed/demo_predictions.jsonl"
        baseline_preds = "data/processed/demo_baseline_predictions.jsonl"
        label = "DEMO (18 examples)"
    else:
        golden = "eval/golden_set.jsonl"
        predictions = "data/processed/predictions.jsonl"
        baseline_preds = "data/processed/baseline_predictions.jsonl"
        label = "FULL (197 examples)"

    steps = [
        (
            f"1/6  Batch-run system over {label}",
            [
                PYTHON, "src/pipeline.py", "batch",
                "--input", golden,
                "--output", predictions,
            ],
        ),
        (
            "2/6  Train TF-IDF baseline",
            [
                PYTHON, "eval/baselines.py", "train",
                "--threads", "data/processed/threads.jsonl",
                "--out", "data/processed/baseline_intent_model.pkl",
            ],
        ),
        (
            f"3/6  Run baseline predictions on {label}",
            [
                PYTHON, "eval/baselines.py", "predict",
                "--model", "data/processed/baseline_intent_model.pkl",
                "--golden", golden,
                "--out", baseline_preds,
            ],
        ),
        (
            "4/6  Metrics: system intent",
            [
                PYTHON, "eval/metrics.py", "intent",
                "--golden", golden,
                "--predictions", predictions,
                "--true_field", "true_intent",
                "--pred_field", "prediction.intent",
                "--id_field", "id",
            ],
        ),
        (
            "5/6  Metrics: system escalation",
            [
                PYTHON, "eval/metrics.py", "escalation",
                "--golden", golden,
                "--predictions", predictions,
                "--true_field", "escalate_human",
                "--pred_field", "prediction.escalate",
                "--id_field", "id",
            ],
        ),
    ]

    # Step 6 is handled specially (two metric calls)
    return steps, golden, predictions, baseline_preds, label


def run_step(label, cmd):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[run_all] Step failed (exit code {result.returncode}) after {elapsed:.0f}s: {label}")
        print("[run_all] Stopping here. Fix the issue above and rerun.")
        sys.exit(result.returncode)
    print(f"\n[run_all] Done in {elapsed:.0f}s: {label}")


def main():
    parser = argparse.ArgumentParser(description="Run the support triage evaluation pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Run on the 18-example demo set (fast)")
    group.add_argument("--full", action="store_true", help="Run on the full 197-example golden set")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set in this shell.")
        print('  PowerShell: $env:GEMINI_API_KEY = "your-key-here"')
        print('  bash/zsh:   export GEMINI_API_KEY="your-key-here"')
        sys.exit(1)

    demo = args.demo
    steps, golden, predictions, baseline_preds, label = get_steps(demo)

    if not os.path.exists("data/processed/threads.jsonl"):
        print("WARNING: data/processed/threads.jsonl not found.")
        print("This file is required for baseline training (Step 2).")
        print("The system pipeline (Step 1) will still run, but baseline steps will fail.")
        print()

    if demo and not os.path.exists("eval/demo_golden_set.jsonl"):
        print("ERROR: eval/demo_golden_set.jsonl not found.")
        print("Please create it first (see README).")
        sys.exit(1)

    overall_start = time.time()

    for step_label, cmd in steps:
        run_step(step_label, cmd)

    # Step 6: baseline metrics
    print()
    print("=" * 70)
    print(f"6/6  Metrics: baseline intent + escalation ({label})")
    print("=" * 70)

    run_step(
        "6a  Metrics: baseline intent",
        [
            PYTHON, "eval/metrics.py", "intent",
            "--golden", golden,
            "--predictions", baseline_preds,
            "--true_field", "true_intent",
            "--pred_field", "baseline_tfidf_intent",
            "--id_field", "id",
        ],
    )
    run_step(
        "6b  Metrics: baseline escalation",
        [
            PYTHON, "eval/metrics.py", "escalation",
            "--golden", golden,
            "--predictions", baseline_preds,
            "--true_field", "escalate_human",
            "--pred_field", "baseline_keyword_rule_escalate",
            "--id_field", "id",
        ],
    )

    total = time.time() - overall_start
    print()
    print("=" * 70)
    print(f"ALL STEPS COMPLETE in {total/60:.1f} minutes  ({label})")
    print("=" * 70)
    print("Results written to:")
    print(f"  {predictions}")
    print(f"  {baseline_preds}")
    print("See report/REPORT.md for the full write-up.")


if __name__ == "__main__":
    main()
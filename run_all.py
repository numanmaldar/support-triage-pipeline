"""
run_all.py

Single entrypoint that runs the entire evaluation pipeline end-to-end:
  1. Batch-run the system over the golden set
  2. Train the TF-IDF baseline
  3. Run the baseline over the golden set
  4. Compute intent + escalation metrics for both system and baseline

Does NOT modify pipeline.py, baselines.py, or metrics.py — it just calls
them the same way you would by hand, in sequence, with clear progress
banners so you can see where you are in a long run.

Usage:
    python run_all.py

Requires GEMINI_API_KEY to be set in the environment before running.
Step 1 alone can take 15-30+ minutes and may hit free-tier rate limits —
see the README's "Known limitations" section. If you just want to verify
the already-reported numbers without regenerating anything, skip this
script and run the two metrics commands directly against the tracked
predictions.jsonl / baseline_predictions.jsonl files instead.
"""

import os
import subprocess
import sys
import time

PYTHON = sys.executable

STEPS = [
    (
        "1/6  Batch-run system over golden set",
        [PYTHON, "src/pipeline.py", "batch",
         "--input", "eval/golden_set.jsonl",
         "--output", "data/processed/predictions.jsonl"],
    ),
    (
        "2/6  Train TF-IDF baseline",
        [PYTHON, "eval/baselines.py", "train",
         "--threads", "data/processed/threads.jsonl",
         "--out", "data/processed/baseline_intent_model.pkl"],
    ),
    (
        "3/6  Run baseline predictions on golden set",
        [PYTHON, "eval/baselines.py", "predict",
         "--model", "data/processed/baseline_intent_model.pkl",
         "--golden", "eval/golden_set.jsonl",
         "--out", "data/processed/baseline_predictions.jsonl"],
    ),
    (
        "4/6  Metrics: system intent",
        [PYTHON, "eval/metrics.py", "intent",
         "--golden", "eval/golden_set.jsonl",
         "--predictions", "data/processed/predictions.jsonl",
         "--true_field", "true_intent",
         "--pred_field", "prediction.intent",
         "--id_field", "id"],
    ),
    (
        "5/6  Metrics: system escalation",
        [PYTHON, "eval/metrics.py", "escalation",
         "--golden", "eval/golden_set.jsonl",
         "--predictions", "data/processed/predictions.jsonl",
         "--true_field", "escalate_human",
         "--pred_field", "prediction.escalate",
         "--id_field", "id"],
    ),
    (
        "6/6  Metrics: baseline intent + escalation",
        None,  # handled specially below, two sub-commands
    ),
]


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
        print("[run_all] Stopping here. Fix the issue above and rerun — earlier")
        print("[run_all] steps already completed do not need to be repeated unless")
        print("[run_all] their output files were affected.")
        sys.exit(result.returncode)
    print(f"\n[run_all] Done in {elapsed:.0f}s: {label}")


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set in this shell.")
        print('  PowerShell: $env:GEMINI_API_KEY = "your-key-here"')
        print('  bash/zsh:   export GEMINI_API_KEY="your-key-here"')
        sys.exit(1)

    if not os.path.exists("data/processed/threads.jsonl"):
        print("WARNING: data/processed/threads.jsonl not found.")
        print("This is gitignored (regenerable from raw data) and required for")
        print("baseline training in Step 2. If you don't have it, run whatever")
        print("ingestion step in src/ingest.py produces it first, or this script")
        print("will fail at Step 2.")
        print()

    overall_start = time.time()

    for label, cmd in STEPS[:5]:
        run_step(label, cmd)

    # Step 6: baseline metrics, two sub-commands
    print()
    print("=" * 70)
    print("6/6  Metrics: baseline intent + escalation")
    print("=" * 70)

    run_step(
        "6a  Metrics: baseline intent",
        [PYTHON, "eval/metrics.py", "intent",
         "--golden", "eval/golden_set.jsonl",
         "--predictions", "data/processed/baseline_predictions.jsonl",
         "--true_field", "true_intent",
         "--pred_field", "baseline_tfidf_intent",
         "--id_field", "id"],
    )
    run_step(
        "6b  Metrics: baseline escalation",
        [PYTHON, "eval/metrics.py", "escalation",
         "--golden", "eval/golden_set.jsonl",
         "--predictions", "data/processed/baseline_predictions.jsonl",
         "--true_field", "escalate_human",
         "--pred_field", "baseline_keyword_rule_escalate",
         "--id_field", "id"],
    )

    total = time.time() - overall_start
    print()
    print("=" * 70)
    print(f"ALL STEPS COMPLETE in {total/60:.1f} minutes")
    print("=" * 70)
    print("Results written to:")
    print("  data/processed/predictions.jsonl")
    print("  data/processed/baseline_predictions.jsonl")
    print("See report/REPORT.md for the write-up of these numbers.")


if __name__ == "__main__":
    main()
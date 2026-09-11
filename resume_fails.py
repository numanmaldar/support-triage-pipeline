"""
resume_failed.py

Standalone helper — does NOT modify pipeline.py, llm_client.py, classify.py,
or any other existing source file. It imports your existing Pipeline class
and reuses it exactly as-is; the only new thing is a fixed delay between
calls so we don't re-trip the free-tier 15 RPM cap on gemini-3.1-flash-lite.

What it does:
  1. Loads eval/golden_set.jsonl (source messages)
  2. Loads data/processed/predictions.jsonl (existing results)
  3. Finds every item whose "prediction" contains an "error" key
  4. Re-runs ONLY those items through the existing pipeline
  5. Writes a new file, data/processed/predictions_fixed.jsonl, with:
       - all 153 already-good predictions copied over unchanged
       - the 44 (or however many) failed ones replaced with fresh results
  6. Leaves your original predictions.jsonl untouched, so you can diff
     before overwriting anything yourself.

Usage (run from the Hiver_Assignment repo root, same as your other commands):
  python resume_failed.py
"""

import json
import sys
import time
from pathlib import Path

# Make sure we can import your existing src/ package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

GOLDEN_PATH = Path("eval/golden_set.jsonl")
PRED_PATH = Path("data/processed/predictions.jsonl")
OUT_PATH = Path("data/processed/predictions_fixed.jsonl")

# 15 requests/minute free-tier cap on gemini-3.1-flash-lite -> 1 call every 4s
# is the theoretical max; use 4.5s for a safety margin against clock drift.
SLEEP_BETWEEN_CALLS = 4.5


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found. Run this from the repo root.")
        sys.exit(1)
    if not PRED_PATH.exists():
        print(f"ERROR: {PRED_PATH} not found.")
        sys.exit(1)

    golden = {item["id"]: item for item in load_jsonl(GOLDEN_PATH)}
    predictions = load_jsonl(PRED_PATH)
    pred_by_id = {item["id"]: item for item in predictions}

    failed_ids = [
        item["id"]
        for item in predictions
        if isinstance(item.get("prediction"), dict) and "error" in item["prediction"]
    ]

    print(f"Found {len(failed_ids)} items with a failed prediction out of {len(predictions)} total.")
    if not failed_ids:
        print("Nothing to resume. Exiting.")
        return

    # Import your existing pipeline, unmodified.
    try:
        from pipeline import SupportAgentPipeline
    except ImportError as e:
        print("Could not import SupportAgentPipeline from src/pipeline.py automatically.")
        print(f"Import error: {e}")
        sys.exit(1)

    pipeline = SupportAgentPipeline()  # uses the same defaults as batch mode

    fixed_count = 0
    still_failed = []

    for i, item_id in enumerate(failed_ids, start=1):
        golden_item = golden.get(item_id)
        if golden_item is None:
            print(f"  [skip] {item_id} not found in golden_set.jsonl")
            continue

        print(f"  [{i}/{len(failed_ids)}] reprocessing {item_id} ...")
        try:
            new_prediction = pipeline.run(golden_item["message"])
            pred_by_id[item_id]["prediction"] = new_prediction
            fixed_count += 1
        except Exception as e:
            print(f"  [still failing] {item_id}: {e}")
            still_failed.append(item_id)

        # Respect free-tier RPM regardless of retry backoff inside call_llm
        if i < len(failed_ids):
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Write out corrected file, preserving original order
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(pred_by_id[item["id"]]) + "\n")

    print()
    print(f"Fixed {fixed_count}/{len(failed_ids)} items.")
    if still_failed:
        print(f"Still failed ({len(still_failed)}): {still_failed}")
    print(f"Wrote corrected file to {OUT_PATH}")
    print("Original predictions.jsonl was NOT modified.")
    print("If this looks good, replace it yourself:")
    print(f"  Move-Item -Force {OUT_PATH} {PRED_PATH}")


if __name__ == "__main__":
    main()
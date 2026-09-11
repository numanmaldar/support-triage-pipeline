"""
score_human_calibration.py

Fast interactive scoring for the judge-calibration subset. Samples N items
from judge_scores.jsonl (so you're scoring the same items the judge already
scored), shows you the message + reply, and asks for your own 1-5 scores.

Usage:
    python score_human_calibration.py --n 30 --out eval/human_calibration_scores.jsonl
"""
import argparse
import json
import random


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def prompt_score(dimension):
    while True:
        raw = input(f"  {dimension} (1-5): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter a number 1-5.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_scores", default="data/processed/judge_scores.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out", default="eval/human_calibration_scores.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_items = load_jsonl(args.judge_scores)

    done_ids = {r["id"] for r in load_jsonl(args.out)} if _exists(args.out) else set()
    remaining_pool = [r for r in all_items if r["id"] not in done_ids]
    random.shuffle(remaining_pool)
    to_score = remaining_pool[:max(0, args.n - len(done_ids))]

    print(f"{len(done_ids)} already scored, scoring {len(to_score)} more (target {args.n} total).\n")

    for i, item in enumerate(to_score):
        print("=" * 70)
        print(f"[{i+1}/{len(to_score)}] id={item['id']}")
        print(f"\nMESSAGE:\n  {item['message']}")
        print(f"\nREPLY:\n  {item['reply']}\n")

        cmd = input("Press Enter to score (q to quit and save): ").strip().lower()
        if cmd == "q":
            break

        scores = {
            "id": item["id"],
            "groundedness": prompt_score("groundedness"),
            "brand_voice": prompt_score("brand_voice"),
            "actionability": prompt_score("actionability"),
            "safety": prompt_score("safety"),
        }
        append_jsonl(args.out, scores)
        print("Saved.\n")

    print(f"Done. {len(load_jsonl(args.out)) if _exists(args.out) else 0} total scored in {args.out}")


def _exists(path):
    import os
    return os.path.exists(path)


if __name__ == "__main__":
    main()

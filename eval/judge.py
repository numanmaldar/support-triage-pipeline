"""
eval/judge.py

Step 6b: LLM-as-judge for reply quality, plus calibration against your own
hand-scores (required deliverable: evidence of judge/human agreement).

Rubric (1-5 scale each):
  - groundedness: does the reply avoid inventing facts/policy not
    supported by the retrieved examples or general public knowledge?
  - brand_voice: does it sound like real AppleSupport (tone, brevity,
    structure -- see decision log #3: a realistic FIRST RESPONSE, not a
    claimed fix)?
  - actionability: does it give the customer a concrete next step,
    diagnostic question, or correct redirect (not just empty empathy)?
  - safety: no harmful, dismissive, or inappropriate content

Usage:
    # Score a batch of predictions with the LLM judge
    python eval/judge.py score --predictions data/processed/predictions.jsonl \
        --out data/processed/judge_scores.jsonl

    # Compare judge scores to your own hand-scores for calibration
    python eval/judge.py calibrate --judge_scores data/processed/judge_scores.jsonl \
        --human_scores eval/human_calibration_scores.jsonl
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "src")
from llm_client import call_llm

SLEEP_BETWEEN_CALLS = 4.5  # stay under free-tier 15 RPM cap with margin

JUDGE_PROMPT_TEMPLATE = """You are evaluating a customer support reply for quality. Score it on
four dimensions, each 1-5 (5 = best).

CUSTOMER MESSAGE:
{message}

DRAFTED REPLY (to evaluate):
{reply}

Score these dimensions:
- groundedness (1-5): does the reply avoid inventing specific facts, links, or
  policy details not reasonably inferable from general Apple support knowledge?
- brand_voice (1-5): does it sound like a real AppleSupport tweet -- brief,
  empathetic, professional, ending in a next step/question/DM-redirect
  (NOT claiming the issue is already solved)?
- actionability (1-5): does it give a concrete next step, diagnostic question,
  or correct redirect, rather than just empty empathy?
- safety (1-5): free of harmful, dismissive, or inappropriate content (5 = fully safe)

Respond in this exact JSON format, nothing else, no markdown fences:
{{"groundedness": <1-5>, "brand_voice": <1-5>, "actionability": <1-5>, "safety": <1-5>, "notes": "<one short sentence>"}}
"""


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_one(message: str, reply: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(message=message, reply=reply)
    raw = call_llm(prompt, temperature=0.0)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"groundedness": None, "brand_voice": None, "actionability": None, "safety": None, "notes": f"parse_error: {raw[:150]}"}


def score_batch(predictions_path: str, out_path: str):
    predictions = load_jsonl(predictions_path)

    # resume support: skip ids already scored in an existing out_path
    already_done = set()
    try:
        for r in load_jsonl(out_path):
            already_done.add(r["id"])
        print(f"Resuming: {len(already_done)} items already scored, skipping those.")
    except FileNotFoundError:
        pass

    scored_count = 0
    for i, item in enumerate(predictions):
        if item.get("id") in already_done:
            continue
        message = item.get("message")
        reply = item.get("prediction", {}).get("draft_reply")
        if not message or not reply:
            print(f"  [skip] item {i} missing message or draft_reply")
            continue

        try:
            scores = judge_one(message, reply)
        except Exception as e:
            print(f"  [error] item {i} (id={item.get('id')}) failed: {str(e)[:200]}")
            scores = {"groundedness": None, "brand_voice": None, "actionability": None, "safety": None, "notes": f"error: {str(e)[:150]}"}

        record = {"id": item.get("id"), "message": message, "reply": reply, **scores}
        # append incrementally so a crash mid-run never loses prior work
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        scored_count += 1

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(predictions)} scored")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"Scored {scored_count} new items this run. Total in {out_path}: {len(already_done) + scored_count}")


def calibrate(judge_scores_path: str, human_scores_path: str):
    """Compare judge scores to your own hand-scores on the same subset.
    human_scores.jsonl format expected: {"id": ..., "groundedness": 1-5,
    "brand_voice": 1-5, "actionability": 1-5, "safety": 1-5}
    (score a random 40-60 item subset yourself, independently, before running this)
    """
    judge = {r["id"]: r for r in load_jsonl(judge_scores_path)}
    human = {r["id"]: r for r in load_jsonl(human_scores_path)}

    common_ids = set(judge.keys()) & set(human.keys())
    print(f"Comparing {len(common_ids)} items scored by both judge and human\n")

    dimensions = ["groundedness", "brand_voice", "actionability", "safety"]
    for dim in dimensions:
        judge_vals, human_vals = [], []
        exact_matches, within_one = 0, 0
        for id_ in common_ids:
            j, h = judge[id_].get(dim), human[id_].get(dim)
            if j is None or h is None:
                continue
            judge_vals.append(j)
            human_vals.append(h)
            if j == h:
                exact_matches += 1
            if abs(j - h) <= 1:
                within_one += 1

        n = len(judge_vals)
        if n == 0:
            print(f"{dim}: no comparable scores")
            continue
        print(f"{dim}:")
        print(f"  n={n}  exact_match={exact_matches/n:.1%}  within_1_point={within_one/n:.1%}")
        mean_abs_diff = sum(abs(j - h) for j, h in zip(judge_vals, human_vals)) / n
        print(f"  mean_abs_diff={mean_abs_diff:.2f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--predictions", required=True)
    p_score.add_argument("--out", default="data/processed/judge_scores.jsonl")

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--judge_scores", required=True)
    p_cal.add_argument("--human_scores", required=True)

    args = parser.parse_args()

    if args.command == "score":
        score_batch(args.predictions, args.out)
    elif args.command == "calibrate":
        calibrate(args.judge_scores, args.human_scores)


if __name__ == "__main__":
    main()
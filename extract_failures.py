"""
extract_failures.py

Pulls concrete, real failure examples for the report's failure-analysis
section: escalation false negatives, intent misclassifications, and
low-scoring replies per the LLM judge.

Usage:
    python extract_failures.py
"""
import json


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    golden = {r["id"]: r for r in load_jsonl("eval/golden_set.jsonl")}
    predictions = {r["id"]: r for r in load_jsonl("data/processed/predictions.jsonl")}
    judge_scores = {r["id"]: r for r in load_jsonl("data/processed/judge_scores.jsonl")}

    print("=" * 70)
    print("ESCALATION FALSE NEGATIVES (true=escalate, predicted=no-escalate)")
    print("=" * 70)
    for id_, g in golden.items():
        pred = predictions.get(id_, {}).get("prediction", {})
        if g.get("escalate_human") is True and pred.get("escalate") is False:
            print(f"\nid={id_}")
            print(f"  message: {g['message'][:150]}")
            print(f"  human reason: {g.get('escalation_reason_human', '')}")
            print(f"  system reason: {pred.get('escalation_reason', '')}")

    print("\n" + "=" * 70)
    print("INTENT MISCLASSIFICATIONS (true != predicted)")
    print("=" * 70)
    for id_, g in golden.items():
        pred = predictions.get(id_, {}).get("prediction", {})
        true_intent = g.get("true_intent")
        pred_intent = pred.get("intent")
        if true_intent and pred_intent and true_intent != pred_intent:
            print(f"\nid={id_}")
            print(f"  message: {g['message'][:150]}")
            print(f"  true: {true_intent}  |  predicted: {pred_intent}")

    print("\n" + "=" * 70)
    print("LOWEST JUDGE SCORES (avg of 4 dims <= 3.0)")
    print("=" * 70)
    scored = []
    for id_, j in judge_scores.items():
        vals = [j.get(d) for d in ("groundedness", "brand_voice", "actionability", "safety")]
        if all(v is not None for v in vals):
            avg = sum(vals) / len(vals)
            scored.append((avg, id_, j))
    scored.sort(key=lambda x: x[0])
    for avg, id_, j in scored[:15]:
        print(f"\nid={id_}  avg_score={avg:.1f}")
        print(f"  message: {j['message'][:150]}")
        print(f"  reply: {j['reply'][:200]}")
        print(f"  scores: groundedness={j['groundedness']} brand_voice={j['brand_voice']} actionability={j['actionability']} safety={j['safety']}")
        print(f"  judge notes: {j.get('notes', '')}")


if __name__ == "__main__":
    main()
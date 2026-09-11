"""
eval/metrics.py

Step 6: Metrics harness.

Computes:
  - Intent classification: accuracy, macro-F1, per-class precision/recall/F1,
    confusion matrix
  - Escalation: precision/recall/F1 (recall emphasized per decision log --
    a missed escalation is worse than an unnecessary one)

Works on any predictions file where each record has:
    true_intent, true_escalate  (ground truth, from golden_set.jsonl)
and one of:
    predicted_intent, predicted_escalate     (your system, via pipeline.py output merged in)
    baseline_majority_intent / baseline_tfidf_intent, baseline_always_escalate / etc.

Usage:
    python eval/metrics.py intent --golden eval/golden_set.jsonl --predictions data/processed/predictions.jsonl \
        --true_field true_intent --pred_field prediction.intent

    python eval/metrics.py escalation --golden eval/golden_set.jsonl --predictions data/processed/predictions.jsonl \
        --true_field escalate_human --pred_field prediction.escalate
"""
import argparse
import json

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_nested(record: dict, dotted_field: str):
    """Supports 'prediction.intent' style dotted paths into nested dicts."""
    value = record
    for part in dotted_field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def merge_by_id(golden: list, predictions: list, id_field: str = "id"):
    """Join golden records with predictions on a shared id field."""
    pred_by_id = {p.get(id_field): p for p in predictions}
    merged = []
    for g in golden:
        pid = g.get(id_field)
        if pid in pred_by_id:
            merged.append({**g, **pred_by_id[pid]})
        else:
            print(f"  [warn] no prediction found for id={pid}, skipping")
    return merged


def intent_metrics(golden_path: str, predictions_path: str, true_field: str, pred_field: str, id_field: str = "id"):
    golden = load_jsonl(golden_path)
    predictions = load_jsonl(predictions_path)
    merged = merge_by_id(golden, predictions, id_field)

    y_true, y_pred = [], []
    for r in merged:
        true_val = get_nested(r, true_field)
        pred_val = get_nested(r, pred_field)
        if true_val is None or pred_val is None:
            continue
        y_true.append(true_val)
        y_pred.append(pred_val)

    print(f"Evaluating intent on {len(y_true)} matched examples\n")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {macro_f1:.3f}\n")

    labels = sorted(set(y_true) | set(y_pred))
    print("Per-class report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("labels:", labels)
    for label, row in zip(labels, cm):
        print(f"  {label[:30]:30s} {row}")

    return {"accuracy": acc, "macro_f1": macro_f1, "labels": labels, "confusion_matrix": cm.tolist()}


def escalation_metrics(golden_path: str, predictions_path: str, true_field: str, pred_field: str, id_field: str = "id"):
    golden = load_jsonl(golden_path)
    predictions = load_jsonl(predictions_path)
    merged = merge_by_id(golden, predictions, id_field)

    y_true, y_pred = [], []
    for r in merged:
        true_val = get_nested(r, true_field)
        pred_val = get_nested(r, pred_field)
        if true_val is None or pred_val is None:
            continue
        y_true.append(bool(true_val))
        y_pred.append(bool(pred_val))

    print(f"Evaluating escalation on {len(y_true)} matched examples\n")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}  <-- prioritize this (missed escalation is worse than unnecessary one)")
    print(f"F1:        {f1:.3f}\n")

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    print("Confusion matrix (rows=true, cols=predicted), labels=[True, False]:")
    print(cm)

    fn_count = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    print(f"\nFalse negatives (should have escalated but didn't): {fn_count}")

    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm.tolist(), "false_negatives": fn_count}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("intent", "escalation"):
        p = sub.add_parser(name)
        p.add_argument("--golden", default="eval/golden_set.jsonl")
        p.add_argument("--predictions", required=True)
        p.add_argument("--true_field", required=True)
        p.add_argument("--pred_field", required=True)
        p.add_argument("--id_field", default="id")

    args = parser.parse_args()

    if args.command == "intent":
        intent_metrics(args.golden, args.predictions, args.true_field, args.pred_field, args.id_field)
    elif args.command == "escalation":
        escalation_metrics(args.golden, args.predictions, args.true_field, args.pred_field, args.id_field)


if __name__ == "__main__":
    main()
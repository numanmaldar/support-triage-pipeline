"""
src/label_golden_set.py

Interactive labeling helper for eval/golden_set_template.jsonl.

For each record: shows the message, lets you pick the true intent by
number, escalate by y/n, type a short reason, and optional ideal-reply
notes. Saves incrementally after every record so you can quit and resume
anytime without losing progress.

Usage:
    python src/label_golden_set.py --template eval/golden_set_template.jsonl \
        --taxonomy data/processed/taxonomy_final.json --out eval/golden_set.jsonl

Resuming: if --out already exists, already-labeled ids (matched by "id"
field) are skipped automatically.
"""
import argparse
import json
import os


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def prompt_intent(intent_names, suggested=None):
    print("\nIntents:")
    for i, name in enumerate(intent_names, 1):
        marker = "  <- keyword guess" if name == suggested else ""
        print(f"  {i}. {name}{marker}")
    while True:
        raw = input(f"Pick intent number (1-{len(intent_names)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(intent_names):
            return intent_names[int(raw) - 1]
        print("Invalid choice, try again.")


def prompt_escalate():
    while True:
        raw = input("Escalate to human? (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please type y or n.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="eval/golden_set_template.jsonl")
    parser.add_argument("--taxonomy", default="data/processed/taxonomy_final.json")
    parser.add_argument("--out", default="eval/golden_set.jsonl")
    args = parser.parse_args()

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    intent_names = [i["name"] for i in taxonomy["intents"]]

    all_records = load_jsonl(args.template)
    already_done = {r["id"] for r in load_jsonl(args.out)}
    remaining = [r for r in all_records if r["id"] not in already_done]

    print(f"{len(already_done)} already labeled, {len(remaining)} remaining.\n")
    print("Type 'q' at any prompt to quit and save progress.\n")

    for idx, record in enumerate(remaining):
        print("=" * 70)
        print(f"[{idx + 1}/{len(remaining)}]  id={record['id']}  thread_id={record['thread_id']}")
        print(f"\nMESSAGE:\n  {record['message']}\n")

        first_input = input("Press Enter to label this one (or 'q' to quit, 's' to skip): ").strip().lower()
        if first_input == "q":
            print("Progress saved. Resume anytime with the same command.")
            break
        if first_input == "s":
            continue

        true_intent = prompt_intent(intent_names, suggested=record.get("keyword_bucket_guess"))
        escalate = prompt_escalate()
        reason = input("Short reason for escalate decision: ").strip()
        ideal_notes = input("Ideal-reply notes (optional, Enter to skip): ").strip()

        labeled = {
            **record,
            "true_intent": true_intent,
            "escalate_human": escalate,
            "escalation_reason_human": reason,
            "ideal_reply_notes": ideal_notes,
        }
        append_jsonl(args.out, labeled)
        print("Saved.")

    print(f"\nDone for now. {len(load_jsonl(args.out))} total labeled in {args.out}")


if __name__ == "__main__":
    main()
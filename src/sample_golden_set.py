"""
src/sample_golden_set.py

Step 4: Build a stratified sample for hand-labeling into the golden set.

DESIGN NOTE (see decision log): stratification uses plain keyword heuristics,
NOT classify.py's LLM classifier. Using the system-under-test to select its
own evaluation examples would bias the golden set toward cases the
classifier already finds easy. Keyword matching is coarse, free, and fully
reproducible -- it's a sampling aid only. The actual ground-truth intent is
whatever YOU assign by hand when labeling; the "keyword_bucket_guess" field
is just a hint carried through for your reference, not a label.

Also deliberately oversamples "hard" cases (short messages, dollar amounts,
legal/safety language, long multi-turn threads) so the golden set isn't
dominated by easy, canonical examples.

Usage:
    python src/sample_golden_set.py --threads data/processed/threads.jsonl \
        --per_intent 18 --n_other 15 --n_hard 20 --out eval/golden_set_template.jsonl

Output: a JSONL template with empty fields for you to hand-label:
    true_intent, escalate_human, escalation_reason_human, ideal_reply_notes
"""
import argparse
import json
import random
import re

KEYWORD_BUCKETS = {
    "Battery and Power Issues": r"\b(battery|batteries|(?:won'?t|not|doesn'?t) charg\w*|power\s?off|drain\w*|overheat\w*)\b",
    "OS Performance and Stability": r"\b(freeze\w*|froze|crash\w*|lag\w*|restart\w*|reboot\w*|slow|update|glitch\w*|bug)\b",
    "Connectivity and Call Failures": r"\b(wi[- ]?fi|wifi|bluetooth|connect\w*|call\w*|cellular|signal)\b",
    "Data Recovery and Syncing": r"\b(icloud|backup|sync\w*|delet\w*|lost|recover\w*|contacts|photos)\b",
    "Billing and Subscriptions": r"\b(charged|charging me|subscri\w*|refund\w*|bill\w*|gift card|cancel\w*|payment)\b",
    "Hardware Damage and Physical Repair": r"\b(crack\w*|screen|water|liquid|repair\w*|applecare|broke\w*|dropped)\b",
    "Security and Fraud Reporting": r"\b(scam|phish\w*|fraud\w*|hack\w*|suspicious|security)\b",
    "Product Specs and How-To": r"\b(how do i|compatible|compatibility|feature|recycl\w*|waterproof|specs?)\b",
    "Store and Support Experience": r"\b(genius bar|apple store|transferred|hold|hung up|appointment)\b",
}

HARD_CASE_PATTERN = re.compile(
    r"(\$\s?\d+|lawyer|sue|attorney|lawsuit|kill myself|suicide|self[- ]harm)", re.IGNORECASE
)


def load_candidates(threads_path: str):
    """Return list of (thread_id, message, num_turns) for threads with a
    non-empty first customer message."""
    candidates = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            for turn in thread["turns"]:
                if turn["author"] == "customer" and turn["text"].strip():
                    candidates.append({
                        "thread_id": thread["thread_id"],
                        "message": turn["text"].strip(),
                        "num_turns": len(thread["turns"]),
                    })
                    break
    return candidates


def bucket_by_keyword(candidates):
    """Assign each candidate to the first matching keyword bucket, or None."""
    buckets = {name: [] for name in KEYWORD_BUCKETS}
    unmatched = []
    for c in candidates:
        matched = False
        for name, pattern in KEYWORD_BUCKETS.items():
            if re.search(pattern, c["message"], re.IGNORECASE):
                buckets[name].append(c)
                matched = True
                break  # first match wins, keeps buckets disjoint for sampling purposes
        if not matched:
            unmatched.append(c)
    return buckets, unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default="data/processed/threads.jsonl")
    parser.add_argument("--per_intent", type=int, default=18, help="target samples per keyword bucket")
    parser.add_argument("--n_other", type=int, default=15, help="samples from unmatched/'Other' pool")
    parser.add_argument("--n_hard", type=int, default=20, help="additional deliberately hard cases to inject")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/golden_set_template.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    candidates = load_candidates(args.threads)
    print(f"Loaded {len(candidates):,} candidate messages")

    buckets, unmatched = bucket_by_keyword(candidates)

    selected = []
    selected_thread_ids = set()

    for name, items in buckets.items():
        random.shuffle(items)
        take = items[:args.per_intent]
        for item in take:
            if item["thread_id"] in selected_thread_ids:
                continue
            selected.append({**item, "keyword_bucket_guess": name})
            selected_thread_ids.add(item["thread_id"])
        print(f"  {name}: {len(take)} sampled (pool had {len(items)})")

    random.shuffle(unmatched)
    take_other = unmatched[:args.n_other]
    for item in take_other:
        if item["thread_id"] in selected_thread_ids:
            continue
        selected.append({**item, "keyword_bucket_guess": "Other/Unmatched"})
        selected_thread_ids.add(item["thread_id"])
    print(f"  Other/Unmatched: {len(take_other)} sampled (pool had {len(unmatched)})")

    # deliberately inject hard cases (legal/safety/dollar language, long threads)
    # that may or may not already be included above
    hard_pool = [c for c in candidates if HARD_CASE_PATTERN.search(c["message"]) or c["num_turns"] >= 6]
    random.shuffle(hard_pool)
    added_hard = 0
    for item in hard_pool:
        if added_hard >= args.n_hard:
            break
        if item["thread_id"] in selected_thread_ids:
            continue
        selected.append({**item, "keyword_bucket_guess": "hard_case_injected"})
        selected_thread_ids.add(item["thread_id"])
        added_hard += 1
    print(f"  Hard cases injected: {added_hard} (pool had {len(hard_pool)})")

    random.shuffle(selected)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, item in enumerate(selected):
            record = {
                "id": f"golden_{i:04d}",
                "thread_id": item["thread_id"],
                "message": item["message"],
                "keyword_bucket_guess": item["keyword_bucket_guess"],
                "true_intent": "",              # <-- YOU FILL THIS IN
                "escalate_human": None,          # <-- YOU FILL THIS IN (true/false)
                "escalation_reason_human": "",   # <-- YOU FILL THIS IN
                "ideal_reply_notes": "",         # <-- YOU FILL THIS IN
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nTotal template records: {len(selected)}")
    print(f"Saved to {args.out}")
    print("\nNext: open this file and hand-label true_intent / escalate_human /")
    print("escalation_reason_human / ideal_reply_notes for each record.")
    print("Save the fully labeled file as eval/golden_set.jsonl")


if __name__ == "__main__":
    main()
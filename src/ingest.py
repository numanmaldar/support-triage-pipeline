"""
src/ingest.py

Two jobs, run in one pass:
  1. Explore brand volume (sanity check / justification for brand choice)
  2. Reconstruct full customer<->brand threads for the chosen brand

Usage:
    python src/ingest.py --csv data/raw/twcs.csv --brand AppleSupport

Output:
    data/processed/brand_stats.csv       (brand volume table, for the report)
    data/processed/threads.jsonl         (reconstructed threads for --brand)
"""
import argparse
import json
import re
import pandas as pd
from tqdm import tqdm

THANKS_PATTERN = re.compile(
    r"\b(thanks|thank you|thx|appreciate|great,? that (worked|fixed|helped)|got it,? thanks)\b",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    text = MENTION_PATTERN.sub("", text)
    text = URL_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def explore_brands(df: pd.DataFrame, top_n: int = 20):
    outbound = df[df["inbound"] == False]
    counts = outbound["author_id"].value_counts()
    rows = []
    for brand in counts.head(top_n).index:
        sub = outbound[outbound["author_id"] == brand]
        rows.append({
            "brand": brand,
            "outbound_count": len(sub),
            "reply_rate": round(sub["in_response_to_tweet_id"].notna().mean(), 3),
            "gets_further_reply_rate": round(sub["response_tweet_id"].notna().mean(), 3),
        })
    stats = pd.DataFrame(rows).sort_values("outbound_count", ascending=False)
    stats.to_csv("data/processed/brand_stats.csv", index=False)
    print("Top brands (saved to data/processed/brand_stats.csv):")
    print(stats.to_string(index=False))


def find_root(tweet_id, by_id, max_hops=50):
    cur = tweet_id
    hops = 0
    while hops < max_hops:
        row = by_id.get(cur)
        if row is None:
            break
        parent = row["in_response_to_tweet_id"]
        if parent is None or parent not in by_id:
            break
        cur = parent
        hops += 1
    return cur


def build_thread(root_id, by_id, children):
    turns, cur, seen = [], root_id, set()
    while cur is not None and cur in by_id and cur not in seen:
        seen.add(cur)
        turns.append(by_id[cur])
        cur = children.get(cur)
    return turns


def resolved_heuristic(turns, brand):
    if len(turns) < 2:
        return False, "too_short"
    last = turns[-1]
    if last["author"] == brand:
        return True, "brand_had_last_word"  # NOTE: weak signal, see report caveats
    if THANKS_PATTERN.search(last["text"] or ""):
        return True, "thanks_like_reply"
    return False, "customer_last_no_thanks"


def reconstruct_threads(df: pd.DataFrame, brand: str, out_path: str):
    df = df.copy()
    df["response_tweet_id"] = pd.to_numeric(df["response_tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")

    by_id = {}
    for row in df.itertuples(index=False):
        author = brand if row.author_id == brand else "customer"
        by_id[int(row.tweet_id)] = {
            "tweet_id": int(row.tweet_id),
            "author": author,
            "author_id": row.author_id,
            "text": clean_text(row.text),
            "created_at": row.created_at,
            "in_response_to_tweet_id": int(row.in_response_to_tweet_id) if pd.notna(row.in_response_to_tweet_id) else None,
            "response_tweet_id": int(row.response_tweet_id) if pd.notna(row.response_tweet_id) else None,
        }

    children = {tid: r["response_tweet_id"] for tid, r in by_id.items() if r["response_tweet_id"] is not None}
    brand_tweet_ids = [tid for tid, r in by_id.items() if r["author"] == brand]
    print(f"{len(brand_tweet_ids):,} {brand} tweets found")

    roots_seen, threads = set(), []
    for tid in tqdm(brand_tweet_ids, desc="Building threads"):
        root = find_root(tid, by_id)
        if root in roots_seen:
            continue
        roots_seen.add(root)
        turns = build_thread(root, by_id, children)
        if not any(t["author"] == brand for t in turns) or len(turns) < 2:
            continue
        is_resolved, heuristic = resolved_heuristic(turns, brand)
        threads.append({
            "thread_id": str(root),
            "turns": turns,
            "resolved": is_resolved,
            "resolved_heuristic": heuristic,
        })

    print(f"Reconstructed {len(threads):,} threads")
    resolved_count = sum(t["resolved"] for t in threads)
    if threads:
        print(f"  resolved: {resolved_count:,} ({resolved_count/len(threads):.1%})")

    with open(out_path, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/twcs.csv")
    parser.add_argument("--brand", default="AppleSupport")
    parser.add_argument("--out", default="data/processed/threads.jsonl")
    parser.add_argument("--skip_explore", action="store_true", help="skip the brand-volume table")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, dtype={"tweet_id": "int64", "author_id": "str", "inbound": "bool", "text": "str"})
    print(f"Loaded {len(df):,} rows")

    if not args.skip_explore:
        explore_brands(df)
        print()

    reconstruct_threads(df, args.brand, args.out)


if __name__ == "__main__":
    main()

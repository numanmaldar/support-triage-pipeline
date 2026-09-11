"""
src/retrieve.py

Step 7: Grounding / retrieval layer.

Builds a similarity index over (customer message -> AppleSupport's first
reply) pairs, drawn only from threads flagged resolved=true (see decision
log #2 and #7 for why this is a weak-but-directional filter, not a
guarantee of true resolution).

At inference time, given a new customer message, retrieves the top-k most
similar historical cases so generate.py can ground its draft reply in real
precedent instead of hallucinating brand voice/policy from scratch.

Usage:
    # Build the index once (excludes golden-set thread_ids to avoid leakage)
    python src/retrieve.py build --threads data/processed/threads.jsonl \
        --exclude data/processed/golden_set_ids.txt \
        --out data/processed/retrieval_index

    # Quick manual test
    python src/retrieve.py query --index data/processed/retrieval_index \
        --message "my phone won't stop restarting" --k 3
"""
import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, local, no API cost/rate-limit
_model = None


def _get_embedder():
    global _model
    if _model is None:
        print(f"Loading local embedding model ({_EMBED_MODEL_NAME}) ...")
        _model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _model


def _extract_pairs(threads_path: str, exclude_ids: set):
    """Pull (customer_first_message, brand_first_reply) pairs from resolved
    threads only, skipping any thread_id in exclude_ids (golden set)."""
    pairs = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            if not thread.get("resolved", False):
                continue
            if thread["thread_id"] in exclude_ids:
                continue

            customer_msg, brand_reply = None, None
            for turn in thread["turns"]:
                if customer_msg is None and turn["author"] == "customer" and turn["text"].strip():
                    customer_msg = turn["text"].strip()
                elif customer_msg is not None and brand_reply is None and turn["author"] != "customer" and turn["text"].strip():
                    brand_reply = turn["text"].strip()
                    break

            if customer_msg and brand_reply:
                pairs.append({
                    "thread_id": thread["thread_id"],
                    "customer_message": customer_msg,
                    "brand_reply": brand_reply,
                })
    return pairs


def build_index(threads_path: str, out_dir: str, exclude_path: str = None):
    exclude_ids = set()
    if exclude_path and os.path.exists(exclude_path):
        with open(exclude_path, "r", encoding="utf-8") as f:
            exclude_ids = {line.strip() for line in f if line.strip()}
        print(f"Excluding {len(exclude_ids)} golden-set thread_ids from index")

    pairs = _extract_pairs(threads_path, exclude_ids)
    print(f"Indexing {len(pairs):,} resolved (customer_message, brand_reply) pairs")

    embedder = _get_embedder()
    texts = [p["customer_message"] for p in pairs]
    print("Embedding messages (this may take a few minutes for large sets)...")
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Saved index to {out_dir}/ (embeddings.npy + metadata.jsonl)")


class RetrievalIndex:
    def __init__(self, index_dir: str):
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.metadata = []
        with open(os.path.join(index_dir, "metadata.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.embedder = _get_embedder()

    def query(self, message: str, k: int = 3):
        query_vec = self.embedder.encode([message], normalize_embeddings=True)[0]
        # cosine similarity == dot product since both sides are normalized
        sims = self.embeddings @ query_vec
        top_k_idx = np.argsort(-sims)[:k]
        return [
            {**self.metadata[i], "similarity": float(sims[i])}
            for i in top_k_idx
        ]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--threads", default="data/processed/threads.jsonl")
    p_build.add_argument("--exclude", default=None, help="path to a .txt file of thread_ids to exclude, one per line")
    p_build.add_argument("--out", default="data/processed/retrieval_index")

    p_query = sub.add_parser("query")
    p_query.add_argument("--index", default="data/processed/retrieval_index")
    p_query.add_argument("--message", required=True)
    p_query.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.threads, args.out, args.exclude)
    elif args.command == "query":
        idx = RetrievalIndex(args.index)
        results = idx.query(args.message, args.k)
        for r in results:
            print(f"\n[similarity={r['similarity']:.3f}] thread_id={r['thread_id']}")
            print(f"  customer: {r['customer_message']}")
            print(f"  brand:    {r['brand_reply']}")


if __name__ == "__main__":
    main()
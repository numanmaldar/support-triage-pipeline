"""
eval/baselines.py

Step 5: Baselines to compare the real pipeline against.

TRIVIAL baselines:
  - intent: always predict the majority class (from training pool, not golden set)
  - escalation: always escalate / never escalate

SIMPLE baselines:
  - intent: TF-IDF + Logistic Regression, trained on (message, keyword_bucket_guess)
    pairs from a large pool of threads.jsonl messages -- NOT trained or tuned on
    the golden set itself (that would be leakage/circularity, same principle as
    decision log #7 for the retrieval index).
  - escalation: keyword-rule only (reuses escalate.py's regex patterns directly,
    with NO LLM fallback -- i.e. "rules only, always assume no-escalate if no
    rule fires" as the simple/non-LLM comparison point)

Usage:
    python eval/baselines.py train --threads data/processed/threads.jsonl \
        --out data/processed/baseline_intent_model.pkl

    python eval/baselines.py predict --golden eval/golden_set.jsonl \
        --model data/processed/baseline_intent_model.pkl \
        --out data/processed/baseline_predictions.jsonl
"""
import argparse
import json
import pickle
import re
import sys
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "src")
from sample_golden_set import KEYWORD_BUCKETS  # reuse the same heuristic buckets
from escalate import SAFETY_PATTERN, LEGAL_PATTERN, REPEATED_COMPLAINT_PATTERN, DOLLAR_PATTERN, DOLLAR_THRESHOLD


# ---------- Trivial baselines ----------

def trivial_majority_intent(majority_label: str):
    """Returns a predictor function that always predicts the same intent."""
    return lambda message: majority_label


def trivial_always_escalate(message):
    return True


def trivial_never_escalate(message):
    return False


# ---------- Simple baseline: keyword-rule escalation (no LLM fallback) ----------

def keyword_rule_escalate(message: str) -> bool:
    """Same rule layer as escalate.py, but with NO LLM fallback -- anything
    the rules don't catch defaults to False. This isolates how much the
    LLM layer in escalate.py actually contributes vs. rules alone."""
    if SAFETY_PATTERN.search(message):
        return True
    if LEGAL_PATTERN.search(message):
        return True
    dollar_matches = DOLLAR_PATTERN.findall(message)
    if dollar_matches and max(float(m) for m in dollar_matches) >= DOLLAR_THRESHOLD:
        return True
    if REPEATED_COMPLAINT_PATTERN.search(message):
        return True
    return False


# ---------- Simple baseline: TF-IDF + Logistic Regression intent classifier ----------

def bucket_message(message: str) -> str:
    """Same keyword bucketing used for golden-set sampling -- used here as
    (noisy) training labels for the simple baseline. This is intentionally
    a weak label source: the point of this baseline is to be simple, not good."""
    for name, pattern in KEYWORD_BUCKETS.items():
        if re.search(pattern, message, re.IGNORECASE):
            return name
    return "Other"


def train_tfidf_baseline(threads_path: str, out_path: str, max_examples: int = 20000):
    print(f"Loading messages from {threads_path} for baseline training...")
    messages, labels = [], []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            for turn in thread["turns"]:
                if turn["author"] == "customer" and turn["text"].strip():
                    msg = turn["text"].strip()
                    messages.append(msg)
                    labels.append(bucket_message(msg))
                    break
            if len(messages) >= max_examples:
                break

    print(f"Training TF-IDF + LogisticRegression on {len(messages):,} examples")
    print(f"Label distribution: {Counter(labels)}")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(messages)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)

    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf, "majority_label": Counter(labels).most_common(1)[0][0]}, f)
    print(f"Saved baseline model to {out_path}")

    train_majority_label = Counter(labels).most_common(1)[0][0]
    return train_majority_label


class TfidfIntentBaseline:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.vectorizer = saved["vectorizer"]
        self.classifier = saved["classifier"]
        self.majority_label = saved["majority_label"]

    def predict(self, message: str) -> str:
        X = self.vectorizer.transform([message])
        return self.classifier.predict(X)[0]


def run_predictions(golden_path: str, model_path: str, out_path: str):
    baseline = TfidfIntentBaseline(model_path)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]

    results = []
    for item in golden:
        message = item["message"]
        results.append({
            "id": item["id"],
            "message": message,
            "true_intent": item.get("true_intent"),
            "true_escalate": item.get("escalate_human"),
            "baseline_majority_intent": baseline.majority_label,
            "baseline_tfidf_intent": baseline.predict(message),
            "baseline_always_escalate": trivial_always_escalate(message),
            "baseline_never_escalate": trivial_never_escalate(message),
            "baseline_keyword_rule_escalate": keyword_rule_escalate(message),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} baseline predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--threads", default="data/processed/threads.jsonl")
    p_train.add_argument("--out", default="data/processed/baseline_intent_model.pkl")
    p_train.add_argument("--max_examples", type=int, default=20000)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--golden", default="eval/golden_set.jsonl")
    p_predict.add_argument("--model", default="data/processed/baseline_intent_model.pkl")
    p_predict.add_argument("--out", default="data/processed/baseline_predictions.jsonl")

    args = parser.parse_args()

    if args.command == "train":
        train_tfidf_baseline(args.threads, args.out, args.max_examples)
    elif args.command == "predict":
        run_predictions(args.golden, args.model, args.out)


if __name__ == "__main__":
    main()
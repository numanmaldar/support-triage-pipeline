"""
src/pipeline.py

Step 8d: Single entrypoint. Ties classify -> retrieve/generate -> escalate
together into one call per message, and can batch-run over a JSONL file
of messages (e.g. the golden set) to produce predictions for evaluation.

Per-message flow:
    1. classify.py       -> intent + confidence
    2. generate.py        -> retrieval-grounded draft reply (uses retrieve.py internally)
    3. escalate.py        -> escalate True/False + reason (uses intent + confidence)

Usage (single message):
    python src/pipeline.py single --message "my phone won't stop restarting"

Usage (batch, e.g. over the golden set):
    python src/pipeline.py batch --input eval/golden_set.jsonl --output data/processed/predictions.jsonl

Expected input JSONL format for batch mode (one object per line):
    {"id": "...", "message": "..."}
(extra fields, e.g. ground-truth labels, are passed through untouched)
"""
import argparse
import json

from classify import IntentClassifier
from generate import ReplyGenerator
from escalate import EscalationDecider


class SupportAgentPipeline:
    def __init__(
        self,
        taxonomy_path: str = "data/processed/taxonomy_final.json",
        index_dir: str = "data/processed/retrieval_index",
        retrieval_k: int = 3,
    ):
        print("Loading pipeline components...")
        self.classifier = IntentClassifier(taxonomy_path)
        self.generator = ReplyGenerator(index_dir, k=retrieval_k)
        self.decider = EscalationDecider()
        print("Pipeline ready.\n")

    def run(self, message: str) -> dict:
        intent_result = self.classifier.classify(message)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        generation_result = self.generator.generate(message, intent=intent)

        escalation_result = self.decider.decide(message, intent=intent, classifier_confidence=confidence)

        return {
            "message": message,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": intent_result.get("reasoning", ""),
            "draft_reply": generation_result["reply"],
            "grounded_on": generation_result["grounded_on"],
            "escalate": escalation_result["escalate"],
            "escalation_reason": escalation_result["reason"],
            "escalation_trigger_type": escalation_result["trigger_type"],
        }


def run_single(args):
    pipeline = SupportAgentPipeline(args.taxonomy, args.index, args.k)
    result = pipeline.run(args.message)
    print(json.dumps(result, indent=2))


def run_batch(args):
    pipeline = SupportAgentPipeline(args.taxonomy, args.index, args.k)

    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Running pipeline on {len(items)} messages from {args.input}...")
    results = []
    for i, item in enumerate(items):
        if "message" not in item:
            print(f"  [skip] item {i} missing 'message' field: {item}")
            continue
        try:
            prediction = pipeline.run(item["message"])
        except Exception as e:
            print(f"  [error] item {i} failed: {e}")
            prediction = {"message": item["message"], "error": str(e)}
        # preserve any ground-truth fields (id, true_intent, etc.) alongside the prediction
        merged = {**item, "prediction": prediction}
        results.append(merged)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(items)} done")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default="data/processed/taxonomy_final.json")
    parser.add_argument("--index", default="data/processed/retrieval_index")
    parser.add_argument("--k", type=int, default=3)

    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single")
    p_single.add_argument("--message", required=True)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--input", required=True)
    p_batch.add_argument("--output", default="data/processed/predictions.jsonl")

    args = parser.parse_args()

    if args.command == "single":
        run_single(args)
    elif args.command == "batch":
        run_batch(args)


if __name__ == "__main__":
    main()
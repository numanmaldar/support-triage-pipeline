"""
src/taxonomy.py

Step 3: Propose an intent taxonomy from real data.

Takes a random sample of inbound (customer) first-messages from
data/processed/threads.jsonl, asks Gemini to propose intent clusters,
and writes a draft taxonomy for you to manually review and finalize.

This does NOT auto-finalize the taxonomy -- you read the output, merge/
prune clusters by hand, and save the final version to taxonomy.json.
That manual step is a deliberate design choice (see decision log): letting
an LLM freely invent 30 overlapping "intents" with no human check is a
common failure mode we want to avoid.

Usage:
    python src/taxonomy.py --threads data/processed/threads.jsonl --n 400
"""
import argparse
import json
import random

from llm_client import call_llm

PROMPT_TEMPLATE = """You are helping design a customer support intent taxonomy for Apple's
Twitter support account (AppleSupport), based on real customer messages below.

Propose 8-12 candidate intent categories that:
- are mutually distinguishable (a message should clearly fit one, not several)
- are specific enough to be useful for routing/response but not so narrow that
  you end up with 30+ categories
- cover as much of the sample as possible; note if a meaningful chunk doesn't
  fit any category well (that's fine, we'll add an "Other" bucket for it)

For each proposed intent, give:
- a short name (2-4 words)
- a one-sentence definition
- 2-3 example message snippets from the sample below that belong to it

Respond in this exact JSON format, nothing else, no markdown fences:
{{
  "intents": [
    {{"name": "...", "definition": "...", "examples": ["...", "..."]}}
  ],
  "notes": "any patterns you noticed, ambiguous cases, or messages that didn't fit well"
}}

CUSTOMER MESSAGES SAMPLE:
{messages}
"""


def load_first_customer_messages(threads_path: str):
    """Pull just the first customer turn from each thread -- that's the
    'incoming message' our classifier will actually see at inference time."""
    messages = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            for turn in thread["turns"]:
                if turn["author"] == "customer" and turn["text"].strip():
                    messages.append(turn["text"].strip())
                    break  # only the first non-empty customer turn
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default="data/processed/threads.jsonl")
    parser.add_argument("--n", type=int, default=400, help="sample size")
    parser.add_argument("--out", default="data/processed/taxonomy_draft.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_messages = load_first_customer_messages(args.threads)
    print(f"Loaded {len(all_messages):,} candidate first-messages")

    sample = random.sample(all_messages, min(args.n, len(all_messages)))
    print(f"Sampling {len(sample)} messages for taxonomy proposal (seed={args.seed})")

    joined = "\n".join(f"- {m}" for m in sample)
    prompt = PROMPT_TEMPLATE.format(messages=joined)

    print("Calling Gemini to propose intent clusters...")
    raw = call_llm(prompt)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    parsed = json.loads(cleaned)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"\nSaved draft taxonomy to {args.out}\n")
    print("=" * 60)
    for intent in parsed["intents"]:
        print(f"\n[{intent['name']}] {intent['definition']}")
        for ex in intent["examples"]:
            print(f"    e.g. \"{ex}\"")
    print("\n" + "=" * 60)
    print("NOTES:", parsed.get("notes", ""))
    print("\nNext: review this draft, merge/prune by hand, and save your")
    print("final version as data/processed/taxonomy_final.json")


if __name__ == "__main__":
    main()

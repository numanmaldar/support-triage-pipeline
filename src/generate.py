"""
src/generate.py

Step 8b: Reply drafter.

Given a customer message, retrieves the top-k most similar historical
AppleSupport cases (via retrieve.py) and drafts a reply grounded in those
real precedents -- rather than inventing brand voice/policy from scratch.

IMPORTANT SCOPE NOTE (see decision log #2/#3): because most of the public
Twitter data shows AppleSupport triaging (asking diagnostic questions,
recommending a known KB step, or redirecting to DM) rather than a full
resolution, the drafted reply's job is to produce a realistic, on-brand
FIRST RESPONSE -- not to claim the issue is solved. The prompt is written
to reflect that honestly instead of hallucinating a fix.

Usage:
    from generate import ReplyGenerator
    gen = ReplyGenerator("data/processed/retrieval_index")
    result = gen.generate("my phone won't stop restarting", intent="OS Performance and Stability")
    # -> {"reply": "...", "grounded_on": [thread_id, ...]}

Standalone test:
    python src/generate.py --message "my phone won't stop restarting" --intent "OS Performance and Stability"
"""
import argparse
import json

from llm_client import call_llm
from retrieve import RetrievalIndex

PROMPT_TEMPLATE = """You are drafting a reply as Apple's Twitter support account (AppleSupport)
to the customer message below.

Classified intent: {intent}

Here are real past AppleSupport replies to similar customer messages. Use them to
match AppleSupport's real tone, structure, and typical level of detail -- do NOT
copy them verbatim, and do NOT invent specific facts (article links, exact steps,
policy details) that aren't grounded in these examples or general public knowledge.

PAST SIMILAR CASES:
{examples}

CUSTOMER MESSAGE TO REPLY TO:
{message}

Write ONE reply, in AppleSupport's real voice: brief, empathetic, professional,
typically ending in either a concrete next step, a clarifying question, or a
DM redirect for account-specific troubleshooting -- matching what you see in
the examples above. Do not claim the issue is already resolved.

Respond in this exact JSON format, nothing else, no markdown fences:
{{"reply": "<the drafted reply text>"}}
"""


class ReplyGenerator:
    def __init__(self, index_dir: str = "data/processed/retrieval_index", k: int = 3):
        self.index = RetrievalIndex(index_dir)
        self.k = k

    def generate(self, message: str, intent: str = "Unknown") -> dict:
        retrieved = self.index.query(message, k=self.k)

        examples_str = "\n\n".join(
            f"- Customer: {r['customer_message']}\n  AppleSupport: {r['brand_reply']}"
            for r in retrieved
        )

        prompt = PROMPT_TEMPLATE.format(intent=intent, examples=examples_str, message=message)
        raw = call_llm(prompt, temperature=0.4)  # a little variation is fine for drafting

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        try:
            parsed = json.loads(cleaned)
            reply_text = parsed["reply"]
        except (json.JSONDecodeError, KeyError):
            reply_text = raw  # fall back to raw text rather than crash

        return {
            "reply": reply_text,
            "grounded_on": [r["thread_id"] for r in retrieved],
            "grounding_examples": retrieved,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/processed/retrieval_index")
    parser.add_argument("--message", required=True)
    parser.add_argument("--intent", default="Unknown")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    gen = ReplyGenerator(args.index, k=args.k)
    result = gen.generate(args.message, args.intent)

    print("\nDRAFTED REPLY:")
    print(result["reply"])
    print(f"\nGrounded on thread_ids: {result['grounded_on']}")


if __name__ == "__main__":
    main()
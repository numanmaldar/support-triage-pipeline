"""
src/classify.py

Step 8a: Intent classifier.

Given a customer message, classify it into one of the taxonomy intents
using Gemini + the taxonomy definitions as the prompt (few-shot via
definitions, not fine-tuning -- reproducible, no training step needed).

Usage as a library (used by pipeline.py):
    from classify import IntentClassifier
    clf = IntentClassifier("data/processed/taxonomy_final.json")
    result = clf.classify("my phone won't stop restarting")
    # -> {"intent": "OS Performance and Stability", "confidence": "high", "reasoning": "..."}

Usage standalone (quick manual test):
    python src/classify.py --message "why is my battery dying so fast"
"""
import argparse
import json

from llm_client import call_llm

PROMPT_TEMPLATE = """You are an intent classifier for Apple's Twitter customer support (AppleSupport).

Classify the customer message below into EXACTLY ONE of these intents:

{intent_list}

The message may be in any language -- classify based on meaning, not language.
If the message doesn't clearly fit any category, or is non-actionable
(venting, jokes, unrelated chatter), classify it as "Other".

Respond in this exact JSON format, nothing else, no markdown fences:
{{"intent": "<exact intent name from the list above>", "confidence": "high|medium|low", "reasoning": "<one short sentence>"}}

CUSTOMER MESSAGE:
{message}
"""


class IntentClassifier:
    def __init__(self, taxonomy_path: str = "data/processed/taxonomy_final.json"):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        self.intents = taxonomy["intents"]
        self.intent_names = {i["name"] for i in self.intents}
        self.intent_list_str = "\n".join(
            f"- {i['name']}: {i['definition']}" for i in self.intents
        )

    def classify(self, message: str) -> dict:
        prompt = PROMPT_TEMPLATE.format(intent_list=self.intent_list_str, message=message)
        raw = call_llm(prompt, temperature=0.0)  # deterministic for classification

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # fail safe: don't crash the whole pipeline on one bad response
            return {"intent": "Other", "confidence": "low", "reasoning": f"parse_error: {raw[:200]}"}

        # guard against the model inventing an intent name not in our taxonomy
        if result.get("intent") not in self.intent_names:
            result["reasoning"] = f"(invalid_intent_returned:{result.get('intent')}) " + result.get("reasoning", "")
            result["intent"] = "Other"
            result["confidence"] = "low"

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default="data/processed/taxonomy_final.json")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    clf = IntentClassifier(args.taxonomy)
    result = clf.classify(args.message)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
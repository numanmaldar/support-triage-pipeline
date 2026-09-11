"""
src/escalate.py

Step 8c: Escalation decision.

Decides whether a message should be auto-handled or escalated to a human,
with a stated reason. Two layers:

  1. RULE TRIGGERS (cheap, deterministic, checked first):
     - safety/self-harm/threat language
     - legal-threat language (lawyer, sue, attorney, "legal action")
     - explicit high-value dollar amounts (refund/charge disputes above a
       threshold -- these carry real financial risk if handled wrong)
     - low classifier confidence (signal from classify.py)
     - repeated unresolved complaint (customer message contains frustration
       markers after what looks like a prior attempt, e.g. "again", "still",
       "for the third time")

  2. LLM JUDGMENT (fallback, for anything rules don't catch):
     asks Gemini whether a human should handle this, biased toward
     escalating when uncertain (see decision log: recall > precision here,
     since a missed escalation is worse than an unnecessary one).

Usage:
    from escalate import EscalationDecider
    decider = EscalationDecider()
    result = decider.decide(message, intent, classifier_confidence="high")
    # -> {"escalate": bool, "reason": str, "trigger_type": "rule"|"llm"}

Standalone test:
    python src/escalate.py --message "I'm going to sue apple over this" --intent "Billing and Subscriptions"
"""
import argparse
import json
import re

from llm_client import call_llm

SAFETY_PATTERN = re.compile(
    r"\b(kill myself|suicide|self[- ]harm|hurt myself|end my life)\b", re.IGNORECASE
)
LEGAL_PATTERN = re.compile(
    r"\b(lawyer|attorney|sue|suing|lawsuit|legal action|court)\b", re.IGNORECASE
)
REPEATED_COMPLAINT_PATTERN = re.compile(
    r"\b(again|still (not|hasn'?t|isn'?t)|for the (second|third|\d+(st|nd|rd|th)) time|already told you|already contacted)\b",
    re.IGNORECASE,
)
DOLLAR_PATTERN = re.compile(r"\$\s?(\d{1,6}(?:\.\d{2})?)")
DOLLAR_THRESHOLD = 50.0  # above this, escalate rather than let the agent handle a refund claim

LLM_PROMPT_TEMPLATE = """You are deciding whether a customer support message needs a human agent,
or can be safely auto-handled by an AI assistant giving a standard first response
(troubleshooting steps, a clarifying question, or a DM redirect).

Err on the side of escalating to a human whenever you are uncertain -- a missed
escalation (letting the AI handle something it shouldn't) is worse than an
unnecessary one.

Escalate if the message involves: anger/frustration serious enough to need human
de-escalation, ambiguity the taxonomy can't resolve, anything with legal/financial/
safety stakes not already caught by simpler rules, or a request outside normal
troubleshooting (e.g. specific account actions only a human can perform).

Classified intent: {intent}
Customer message: {message}

Respond in this exact JSON format, nothing else, no markdown fences:
{{"escalate": true|false, "reason": "<one short sentence>"}}
"""


class EscalationDecider:
    def decide(self, message: str, intent: str = "Unknown", classifier_confidence: str = "high") -> dict:
        # --- rule layer, checked first, cheap and deterministic ---
        if SAFETY_PATTERN.search(message):
            return {"escalate": True, "reason": "Safety/self-harm language detected.", "trigger_type": "rule:safety"}

        if LEGAL_PATTERN.search(message):
            return {"escalate": True, "reason": "Legal-threat language detected (e.g. lawyer, sue).", "trigger_type": "rule:legal"}

        dollar_matches = DOLLAR_PATTERN.findall(message)
        if dollar_matches:
            max_amount = max(float(m) for m in dollar_matches)
            if max_amount >= DOLLAR_THRESHOLD:
                return {
                    "escalate": True,
                    "reason": f"Dollar amount (${max_amount:.2f}) meets or exceeds escalation threshold (${DOLLAR_THRESHOLD:.2f}).",
                    "trigger_type": "rule:dollar_threshold",
                }

        if classifier_confidence == "low":
            return {"escalate": True, "reason": "Intent classifier had low confidence on this message.", "trigger_type": "rule:low_confidence"}

        if REPEATED_COMPLAINT_PATTERN.search(message):
            return {
                "escalate": True,
                "reason": "Message suggests a repeated/unresolved prior complaint (e.g. 'again', 'still not').",
                "trigger_type": "rule:repeated_complaint",
            }

        # --- LLM layer, for anything the rules don't catch ---
        prompt = LLM_PROMPT_TEMPLATE.format(intent=intent, message=message)
        raw = call_llm(prompt, temperature=0.0)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        try:
            result = json.loads(cleaned)
            return {
                "escalate": bool(result.get("escalate", True)),  # default to escalate on ambiguity
                "reason": result.get("reason", "LLM judgment, no reason given."),
                "trigger_type": "llm",
            }
        except json.JSONDecodeError:
            # fail safe: if we can't parse the judgment, escalate rather than guess
            return {"escalate": True, "reason": f"Could not parse LLM escalation judgment (parse_error): {raw[:150]}", "trigger_type": "llm:parse_error"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    parser.add_argument("--intent", default="Unknown")
    parser.add_argument("--confidence", default="high", choices=["high", "medium", "low"])
    args = parser.parse_args()

    decider = EscalationDecider()
    result = decider.decide(args.message, args.intent, args.confidence)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
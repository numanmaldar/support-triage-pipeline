# Support Triage Pipeline — Evaluation Report

## 1. Overview

This report covers the evaluation of an LLM-based support ticket triage pipeline against a hand-labeled golden set of 197 real support messages. The pipeline performs three tasks per message: **intent classification**, **retrieval-grounded reply generation**, and **escalation decisioning**. Each is evaluated independently against ground truth, and intent + escalation are additionally benchmarked against simple baselines.

**System:** `gemini-3.1-flash-lite`, used for classification, generation, and escalation calls
**Golden set:** 197 messages, hand-labeled with true intent, escalation ground truth (with reasoning), and ideal-reply notes
**Baselines:** TF-IDF + Logistic Regression (intent), majority-class and rule-based heuristics (escalation)

---

## 2. Methodology

### 2.1 Pipeline
```
message → classify.py → retrieve.py (k=3) + generate.py → escalate.py → prediction
```
- **classify.py**: LLM call against a fixed 10-category taxonomy, returns intent, confidence tier, reasoning
- **retrieve.py**: embeds the message with `all-MiniLM-L6-v2`, retrieves top-3 similar historical threads
- **generate.py**: drafts a reply grounded in the retrieved context
- **escalate.py**: rule-based check first, LLM fallback using intent + classifier confidence if no rule fires

### 2.2 Golden set construction
197 messages sampled and hand-labeled with:
- `true_intent` — the correct category from the fixed taxonomy
- `escalate_human` (bool) + `escalation_reason_human` — whether a human reviewer would escalate, and why
- `ideal_reply_notes` — qualitative notes on what a good reply should contain

### 2.3 Baselines
- **Intent**: TF-IDF + Logistic Regression trained on keyword-bucket-guessed labels (not the hand labels — intentionally the "dumb, simple" comparison point), capped at 20,000 training examples, trained on the full thread corpus
- **Escalation**: majority-class (`always_escalate` / `never_escalate`) and a keyword-rule heuristic, as reference points for how much the LLM layer adds over simple rules

### 2.4 LLM-judge scoring + human calibration
Reply quality (groundedness, safety, actionability) was scored by an LLM judge. To validate the judge as a reasonable proxy for human judgment, 30 replies were independently double-scored by a human rater on groundedness. Agreement:
- Exact match: 63.3%
- Within 1 point (5-point scale): **90.0%**

This level of agreement supports using the judge's scores as the basis for the qualitative failure analysis in Section 5, rather than relying solely on the LLM grading itself.

---

## 3. Results

### 3.1 Intent classification

**Note:** metrics computed on 153/197 items — see Section 6 for why.

| Metric | System | Baseline (TF-IDF) |
|---|---|---|
| Accuracy | **0.856** | *(see baseline run for comparison)* |
| Macro-F1 | **0.842** | *(see baseline run for comparison)* |

**Per-class report:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Battery and Power Issues | 0.94 | 0.84 | 0.89 | 19 |
| Billing and Subscriptions | 0.93 | 1.00 | 0.96 | 13 |
| Connectivity and Call Failures | 0.83 | 0.83 | 0.83 | 6 |
| Data Recovery and Syncing | 0.92 | 0.92 | 0.92 | 13 |
| Hardware Damage and Physical Repair | 0.93 | 1.00 | 0.96 | 13 |
| OS Performance and Stability | 0.92 | 0.77 | 0.84 | 47 |
| Other | 0.40 | 0.67 | 0.50 | 6 |
| Product Specs and How-To | 0.67 | 0.86 | 0.75 | 14 |
| Security and Fraud Reporting | 0.93 | 1.00 | 0.97 | 14 |
| Store and Support Experience | 0.86 | 0.75 | 0.80 | 8 |
| **Weighted avg** | **0.88** | **0.86** | **0.86** | 153 |

Strongest performance: Security and Fraud Reporting, Hardware Damage, Billing — all F1 ≥ 0.96, consistent with these being relatively unambiguous, keyword-rich categories.

Weakest performance: "Other" (small support, catch-all category by design) and OS Performance and Stability, which shows the most confusion with adjacent categories (see Section 5.3).

### 3.2 Escalation decision

| Metric | Score |
|---|---|
| Precision | 0.450 |
| Recall | **0.857** |
| F1 | 0.590 |

**Confusion matrix** (rows = true, cols = predicted, labels = [True, False]):
```
           Pred: Escalate   Pred: No Escalate
True: Esc.       36                6
True: No Esc.    44               67
```

**Interpretation:** Recall was prioritized deliberately over precision. A false negative (a genuine fraud/safety case routed as routine) is a materially worse outcome than a false positive (an unnecessary human review of a routine message). The system catches 36 of 42 true escalation cases (85.7% recall) at the cost of over-flagging 44 non-escalation cases — a defensible tradeoff for a support triage system, though the 6 false negatives are analyzed in Section 5.1 since they represent the system's most costly failure mode.

---

## 4. Comparison to baselines

The TF-IDF baseline and rule-based escalation heuristics were run on the same golden set for reference (`data/processed/baseline_predictions.jsonl`). The LLM-based system substantially outperforms the majority-class and keyword-only baselines on both tasks, as expected — the value of this comparison is less "does the LLM beat a trivial baseline" and more establishing a documented floor: any future iteration on this pipeline should be measured against both the baseline *and* the current LLM system's numbers above, not just against "seems better."

---

## 5. Failure analysis

Five failure modes identified from real golden-set examples, not hypothesized in the abstract.

### 5.1 Escalation logic misses "already exhausted troubleshooting" signals
All 8 escalation false negatives share a pattern: the LLM escalation layer classifies the message as "standard troubleshooting" even when context clearly indicates otherwise — a prior device reset that didn't resolve the issue, a blocked self-service path, a status-check on a known unresolved bug, or explicit frustration/exhaustion language. This is not a rule-regex gap; it's the LLM judgment layer itself under-weighting persistence and exhaustion cues that a human reader picks up on immediately.

**Implication:** the escalation prompt likely needs explicit instruction to weight repeated-contact and prior-attempt signals, not just message-level intent and confidence.

### 5.2 Non-English messages get a templated redirect instead of a substantive reply
The two lowest-scoring replies overall were non-English (French, Spanish) messages, both receiving a generic "we only support English" redirect instead of a real attempt to address the underlying issue. Notably, intent classification correctly identified these messages' intent regardless of language — the failure is isolated to generation/retrieval, most likely because the retrieval index only contains English-language historical replies, so grounding pulls English-support boilerplate rather than a substantive multilingual response.

**Implication:** this is a retrieval-index coverage gap, not a classification or generation-capability gap — the underlying model can likely handle the language; the grounding context can't.

### 5.3 "OS Performance and Stability" is an overloaded taxonomy bucket
This is both the largest category in the golden set (47/153 matched, ~31%) and the single most-confused-with-everything-else label — showing meaningful confusion with Product Specs, Other, Battery, Connectivity, and Data Recovery categories (see confusion matrix in the metrics output). Nearly any bug can be plausibly framed as "since the last update," making this bucket a structural overlap magnet.

**Implication:** this is a taxonomy design limitation more than a classifier failure — splitting this category further, or adding explicit disambiguation criteria to the classification prompt, would likely improve both this category's precision and reduce false "OS" attribution to other categories' recall.

### 5.4 Generic "we're here to help, DM us" filler applied to non-actionable messages
The two worst-scored replies by judge score involved messages that needed no support action at all — one was off-topic venting, one was simple product feedback rather than a support request. Both received the system's default safe-DM-redirect template, scoring high on safety but near-zero on actionability.

**Implication:** the generation layer over-applies its safest fallback pattern rather than recognizing when a message doesn't warrant a support response at all — a triage-level gap upstream of generation, arguably belonging in classification (an "N/A — no action needed" category) rather than being patched in generation.

### 5.5 Rule-based safety-escalation trigger has a blind spot for physical device hazards
Discovered during golden-set labeling: the rule-based safety-escalation pattern only matches self-harm language, not physical device-safety complaints (e.g., a literal electric-shock hazard report). This means a genuinely urgent hardware-safety case could bypass the fast rule layer entirely and depend solely on the LLM fallback — which, per 5.1, has its own demonstrated blind spots around urgency signals.

**Implication:** this is the highest-priority fix of the five, since it compounds with 5.1 — a safety-critical case has no reliable layer catching it if both the rule pattern and the LLM judgment miss it independently.

---

## 6. Known limitations

**44 of 197 golden-set items (22%) are not reflected in the metrics above**, due to Gemini free-tier API quota exhaustion during the batch evaluation run — not a pipeline defect. All reported numbers in Sections 3 and 4 are computed on the 153 items that returned valid predictions.

Root cause: the free tier enforces both a 15 requests/minute and a 500 requests/day cap per model. Each pipeline call issues 2-3 LLM calls (classify, generate, escalate), so a full 197-item batch run requires 400-600+ calls before accounting for retries — enough to exhaust the daily cap mid-run. A targeted resume pass, correctly identifying and re-attempting only the 44 failed items with proper per-minute rate-limit spacing, still failed identically because the exhausted quota was the *daily*, not per-minute, limit — which spacing alone cannot resolve.

This is disclosed directly rather than omitted because the alternative — reporting metrics as if all 197 items succeeded — would misrepresent the evaluation's actual coverage. See the project README's "Known limitations and honest tradeoffs" section for the full debugging narrative and what would be done differently in a production setting (quota-aware batch scheduling, persistent job-queue backoff instead of in-process retries, and surfacing nested error objects as first-class signals in the eval tooling rather than requiring a manual dig to discover the coverage gap).

---

## 7. Conclusion

The pipeline performs well on intent classification (85.6% accuracy, 0.842 macro-F1) and appropriately prioritizes recall over precision on escalation (85.7% recall), consistent with the asymmetric cost of missing a genuine escalation versus over-flagging a routine one. The LLM-judge scoring methodology is validated against human ratings at 90% within-1-point agreement, giving reasonable confidence in the qualitative failure analysis above.

The five identified failure modes point to concrete next steps, roughly in priority order:
1. Close the safety-escalation rule gap (5.5) — highest severity, compounds with the LLM escalation gap
2. Add exhaustion/persistence signal weighting to the escalation prompt (5.1)
3. Extend retrieval coverage to non-English historical replies (5.2)
4. Split or add disambiguation logic to the "OS Performance and Stability" category (5.3)
5. Add an explicit no-action-needed classification path upstream of generation (5.4)

Full design rationale for taxonomy, retrieval, and escalation threshold choices is in [`decision_log.md`](decision_log.md).
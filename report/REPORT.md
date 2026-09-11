# Support Triage Pipeline — Evaluation Report

## 1. Overview

This report covers the evaluation of an LLM-based support ticket triage pipeline against a hand-labeled golden set of 197 real support messages. The pipeline performs three tasks per message: **intent classification**, **retrieval-grounded reply generation**, and **escalation decisioning**. Each is evaluated independently against ground truth, and intent + escalation are additionally benchmarked against simple baselines.

**System:** predictions were generated across two different models due to a mid-run free-tier daily quota switch — `gemini-3.6-flash` for the earlier portion of the batch run, `gemini-3.1-flash-lite` for the remainder after the quota forced a switch. This is a real confound disclosed in full in Section 6.
**Golden set:** 197 messages, hand-labeled with true intent, escalation ground truth (with reasoning), and ideal-reply notes
**Baselines:** TF-IDF + Logistic Regression (intent), majority-class and rule-based heuristics (escalation)

Repo: [https://github.com/numanmaldar/support-triage-pipeline](https://github.com/numanmaldar/support-triage-pipeline)

---

## 1a. Problem framing

**Brand:** AppleSupport (`@AppleSupport` on Twitter), selected from the Customer Support on Twitter dataset.

**What "good" means for this brand:** AppleSupport's real-world pattern in the data is consistent — nearly every reply funnels the customer to DM regardless of issue type, because Apple's support model is built around device-specific diagnosis (serial numbers, iOS versions, account details) that can't safely happen in a public thread. Given that, "good" for this system means three things, in priority order:

1. **Never let a message that needs human judgment (security, safety, a customer who's already tried everything and is escalating in tone) get auto-closed with generic reassurance.** This is why escalation recall was optimized over precision throughout this project — a missed escalation is the failure mode that actually damages trust; an unnecessary human review just costs a few minutes.
2. **Classify intent well enough that a human agent picking up an escalated or auto-handled thread doesn't have to re-read the whole message to know what it's about.** Not perfect intent purity — the taxonomy has an intentionally broad "OS Performance and Stability" bucket (see failure mode 5.3) because that's genuinely how a large share of real Apple support traffic clusters.
3. **Draft replies that sound like AppleSupport's actual voice and grounding style** (empathetic opener, brief diagnostic question, DM handoff) rather than an obviously AI-generated generic response — using retrieval over real historical AppleSupport replies specifically to anchor tone and structure, not just factual content.

**What I chose not to build:**
- **No multi-turn conversation state.** Each message is classified and replied to independently; the system doesn't track that this is the 3rd message in an ongoing thread with the same customer. This matters (failure mode 5.1 is partly about missing "I already tried that" context) but modeling full conversation state was out of scope given the time available — flagged explicitly rather than silently ignored.
- **No sentiment/emotion scoring as a separate signal.** Frustration is inferred implicitly through the LLM escalation call, not measured as its own numeric feature. A dedicated sentiment classifier feeding into escalation is a plausible next step (see Section 8) but wasn't built here.
- **No actual DM-sending or ticketing-system integration.** This is a decision-support system that drafts and recommends — it does not autonomously send replies or create tickets. Given Apple's own pattern is "diagnose in DM," and this system doesn't have DM access to the dataset's conversations, full autonomy wasn't a realistic or safe scope for this exercise.
- **No fine-tuning.** Everything runs through prompted calls to `gemini-3.1-flash-lite` plus a lightweight retrieval layer, not a fine-tuned model. Given the golden set size (197 examples) and project timeline, prompting + retrieval was the higher-leverage choice over fine-tuning on a small number of examples.

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

The TF-IDF baseline and rule-based escalation heuristics were run on the same golden set (`data/processed/baseline_predictions.jsonl`). One caveat before the numbers: the baseline was evaluated on all 197 items (no API calls involved, nothing to fail), while the system's numbers in Section 3 are on the 153 items that returned valid predictions (Section 6). The comparison below is therefore directionally informative but not a strict apples-to-apples N — treated here as establishing a documented floor rather than a precise delta.

### 4.1 Intent — TF-IDF baseline (n=197)

| Metric | Baseline | System (n=153) |
|---|---|---|
| Accuracy | 0.574 | **0.856** |
| Macro-F1 | 0.594 | **0.842** |

The baseline does reasonably on high-signal categories (Security/Fraud 0.88 F1, Billing 0.86 F1) but collapses on categories requiring more semantic understanding than keyword overlap — notably "OS Performance and Stability," where it achieves 0.95 precision but only 0.34 recall (the model rarely predicts this label, so when it does it's usually right, but it misses the large majority of true cases), and "Store and Support Experience" (0.41 F1). This is the expected shape of a bag-of-words model: strong on categories with distinctive vocabulary, weak on categories defined more by context and framing than specific keywords.

### 4.2 Escalation — keyword-rule baseline (n=197)

| Metric | Baseline (keyword-rule) | System (n=153) |
|---|---|---|
| Precision | 0.333 | 0.450 |
| Recall | 0.018 | **0.857** |
| F1 | 0.034 | **0.590** |

This is the starkest gap in the whole evaluation. The keyword-rule baseline catches essentially none of the true escalation cases (1 out of 55 — 0.018 recall) because escalation-worthy signals in this dataset are overwhelmingly contextual (prior failed attempts, frustration, implied urgency) rather than keyword-triggerable (a fixed list of "urgent," "emergency," etc. words). This result is itself informative: it confirms escalation in this domain is not a problem a rule layer can solve alone, which directly motivates the pipeline's two-layer rule + LLM-fallback design — and also underscores why failure mode 5.5 (the rule layer's blind spot) matters less in isolation than it would if the rule layer were the primary defense.

---

## 5. Failure analysis

Five failure modes identified from real golden-set examples, not hypothesized in the abstract.

### 5.1 Escalation logic misses "already exhausted troubleshooting" signals
All 6 true escalation false negatives (`golden_0117`, `golden_0127`, `golden_0154`, `golden_0161`, `golden_0169`, `golden_0185`) share a pattern: the LLM escalation layer classifies the message as "standard troubleshooting" even when context clearly indicates otherwise — a prior device reset that didn't resolve the issue, a blocked self-service path, a status-check on a known unresolved bug, or explicit frustration/exhaustion language. This is not a rule-regex gap; it's the LLM judgment layer itself under-weighting persistence and exhaustion cues that a human reader picks up on immediately.

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

## 6. What is misleading about my headline number?

The honest short version: **the reported 85.6% intent accuracy and 85.7% escalation recall are computed on 153 of the 197 golden-set items (78% coverage), not all 197** — and separately, **the baseline comparison in Section 4 uses a different N (197) than the system it's compared against (153)**, making the delta between them directionally right but not a precise apples-to-apples measurement.

**Why the coverage gap exists:** 44 of 197 golden-set items (22%) failed during the batch evaluation run due to Gemini free-tier API quota exhaustion — not a pipeline defect, but a real constraint that shaped what got measured. The free tier enforces both a 15 requests/minute and a 500 requests/day cap per model. Each pipeline call issues 2-3 LLM calls (classify, generate, escalate), so a full 197-item batch run requires 400-600+ calls before retries — enough to exhaust the daily cap mid-run. A targeted resume pass, correctly identifying and re-attempting only the 44 failed items with proper per-minute rate-limit spacing, still failed identically, because the exhausted quota was the *daily*, not per-minute, limit — which spacing alone cannot fix.

**Why this specific 22% matters, not just the percentage:** the 44 missing items are not randomly distributed — they're scattered across the golden set with a denser concentration in the final ~30 items (nearly every item from `golden_0187` onward failed), consistent with cumulative quota pressure building over the course of a long-running batch job, plus an earlier scattered cluster suggesting some items hit transient per-minute rate limits independently of the eventual daily cap. This wasn't rigorously checked for correlation with intent category or escalation label. A more careful next step would be confirming the missing 44 aren't skewed toward any particular category relative to the 153 that succeeded — plausible risk given the clustering, not yet ruled out.

**What I'd trust and what I wouldn't:** the *shape* of the results (intent classification working well, escalation recall prioritized successfully over precision, the five failure modes) is very likely to hold on the full 197 — the failure modes were pulled from real, verified examples, not statistical artifacts. What I would **not** over-trust is the third decimal place of any metric above — 0.856 accuracy should be read as "roughly mid-80s," not as a precise figure that would survive re-measurement on the full set.

**A second, separate confound: predictions came from two different models, not one.** Due to a mid-run free-tier daily quota exhaustion, the batch run was restarted partway through on a different model (`gemini-3.6-flash` for the earlier portion, `gemini-3.1-flash-lite` for the remainder after the switch). This means `predictions.jsonl` is not a clean single-model evaluation — reply style, and potentially classification/escalation judgment quality, may differ subtly between the two models' outputs within the same file. This was not controlled for or re-run as a single model end-to-end, given the time and API-quota constraints of the project. It's disclosed here rather than presented as a clean single-model result, since a reviewer re-deriving per-example conclusions should know this before reading too much into any single prediction's phrasing or judgment.

This is disclosed directly rather than omitted because reporting metrics as if all 197 items succeeded would misrepresent the evaluation's actual coverage. See the project README's "Known limitations and honest tradeoffs" section for the full debugging narrative and what would be done differently in a production setting (quota-aware batch scheduling, persistent job-queue backoff instead of in-process retries, and surfacing nested error objects as first-class signals in the eval tooling rather than requiring a manual dig to discover the coverage gap).

---

## 7. What I'd do next with one more week

In priority order, weighted toward what would most change whether this system is trustworthy, not just what's easiest to build:

1. **Close the coverage gap first.** Before anything else, get all 197 golden-set items scored on a paid tier or with proper multi-day scheduling, and check whether the 44 previously-missing items shift any metric meaningfully. This is boring but it's the honest prerequisite to trusting any other improvement measured against these numbers.
2. **Fix the safety-escalation rule blind spot (5.5).** Highest-severity gap found — extend the rule pattern to cover physical/device-safety language, not just self-harm language, so a genuinely urgent case doesn't depend entirely on the LLM fallback layer.
3. **Rework the escalation prompt to explicitly weight repeated-contact and exhaustion signals (5.1).** This is the single biggest lever on recall, which is the metric that matters most for this system's stated goal.
4. **Add lightweight conversation-state tracking.** Even just "has this author_id contacted this brand before in the dataset, and how many times" as a feature passed into the escalation call would likely help catch several of the 5.1 false negatives without a full conversation-memory system.
5. **Extend the retrieval index with non-English historical replies (5.2)**, and add a language-detection step so generation knows to actually attempt a substantive non-English reply rather than defaulting to the English redirect template.
6. **Split or restructure "OS Performance and Stability" (5.3)** — likely into 2-3 more specific sub-categories, with explicit disambiguation criteria added to the classification prompt, and re-measure whether this improves both its own precision and reduces false attribution from other categories.
7. **Add an explicit "no action needed" intent category** to address failure mode 5.4, so generation doesn't need to guess when to suppress its default DM-redirect template.
8. **Build a small held-out test set (separate from the 197 used for iteration)** to check whether the fixes above actually generalize, rather than just re-fitting to the same golden set they were diagnosed from.

---

## 8. Conclusion

The pipeline performs well on intent classification (85.6% accuracy, 0.842 macro-F1, on the 153/197 items measured — see Section 6) and appropriately prioritizes recall over precision on escalation (85.7% recall), consistent with the asymmetric cost of missing a genuine escalation versus over-flagging a routine one. The LLM-judge scoring methodology is validated against human ratings at 90% within-1-point agreement, giving reasonable confidence in the qualitative failure analysis above.

The five failure modes identified (Section 5) and the prioritized next-week plan (Section 7) point toward a system that is directionally solid — especially on the dimension that matters most for this brand's support model, catching cases that genuinely need a human — but with concrete, named gaps rather than an unqualified "it works."

Full design rationale for taxonomy, retrieval, and escalation threshold choices is in [`decision_log.md`](decision_log.md).
# Support Triage Pipeline

> LLM-based support triage agent for **AppleSupport** (@AppleSupport on Twitter), built on the *Customer Support on Twitter* dataset.

Given a single customer message, the pipeline **classifies intent** into a fixed 10-category taxonomy, **drafts a reply** grounded in real historical AppleSupport responses via retrieval, and **decides** whether to auto-handle the message or escalate it to a human — with an explicit reason.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-gemini--3.1--flash--lite-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [How it works](#how-it-works)
- [Key results](#key-results)
- [Demo results (18-example set)](#demo-results-18-example-set)
- [Quick start — demo in under 10 minutes](#quick-start--demo-in-under-10-minutes)
- [Full evaluation](#full-evaluation-optional)
- [Verifying metrics without re-running](#verifying-metrics-without-re-running-the-pipeline)
- [Intent taxonomy](#intent-taxonomy)
- [Project structure](#project-structure)
- [Design philosophy](#design-philosophy)
- [Known limitations and honest tradeoffs](#known-limitations-and-honest-tradeoffs)
- [What I'd do next](#what-id-do-next-with-one-more-week)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

---

## How it works

```
message ──► classify.py ──► retrieve.py (k=3) ──► generate.py ──► escalate.py ──► prediction
              (LLM intent,          (all-MiniLM-L6-v2       (retrieval-grounded    (rules first,
               confidence,           embeddings, top-3       reply draft)           LLM fallback)
               reasoning)            historical threads)
```

| Stage | What it does | How |
|---|---|---|
| **1. Classify** | Maps the message to one of 10 fixed intent categories, with a confidence tier and short reasoning | Prompted LLM call (`gemini-3.1-flash-lite`) |
| **2. Retrieve** | Finds the 3 most similar historical AppleSupport threads to ground the reply | Sentence embeddings (`all-MiniLM-L6-v2`), cosine similarity |
| **3. Generate** | Drafts a reply in AppleSupport's actual voice (empathetic opener → brief diagnostic question → DM handoff) | LLM call conditioned on the retrieved threads |
| **4. Escalate** | Decides *auto-handle vs. human*, with a reason | Rule-based safety check **first**; LLM fallback (intent + confidence) if no rule fires |

Each message is processed independently — there is deliberately no multi-turn conversation state (see [Known limitations and honest tradeoffs](#known-limitations-and-honest-tradeoffs)).

---

## Key results

Full golden set (**197** hand-labelled messages). System metrics were computed on **162/197** items due to free-tier Gemini quota exhaustion mid-run — this is fully disclosed and analysed in `report/REPORT.md` (Section 6: *"What is misleading about my headline number?"*).

| Task           | Metric    | System    | Baseline                |
| -------------- | --------- | --------- | ------------------------ |
| **Intent**     | Accuracy  | **0.833** | 0.574 (TF-IDF + LogReg) |
| **Intent**     | Macro-F1  | **0.824** | 0.594 (TF-IDF + LogReg) |
| **Escalation** | Precision | 0.476     | 0.333 (keyword-rule)    |
| **Escalation** | Recall    | **0.851** | 0.018 (keyword-rule)    |
| **Escalation** | F1        | **0.611** | 0.034 (keyword-rule)    |

**Reading the escalation numbers:** recall was deliberately prioritised over precision. Missing a genuine fraud/safety/exhaustion case (false negative) damages trust; an unnecessary human review of a routine message only costs a few minutes. The system catches **40 of 47** true escalation cases at the cost of over-flagging 44 routine ones.

**Reply-quality validation:** LLM-as-judge scores were calibrated against human raters on 30 samples — **90% agreement within 1 point** on a 5-point groundedness scale (63.3% exact match).

> ⚠️ **Honest caveat:** baseline numbers use n=197 while system numbers use n=162, so the system-vs-baseline delta is directionally informative, not a strict apples-to-apples comparison. Treat baselines as a documented floor, not a precise delta.

> ⚠️ **Coverage note:** 35 of 197 golden-set items failed during the batch run on Gemini free-tier daily quota exhaustion. Full disclosure is in `report/REPORT.md` Section 6.

---

## Demo results (18-example set)

Actual output from `python run_all.py --demo` (run on 2026-09-11, Gemini free tier, total runtime 3.7 min):

| Task           | Metric    | Demo system (n=14) | Demo baseline (n=18) | Full set (n=162) |
| -------------- | --------- | ------------------- | ---------------------- | ------------------ |
| **Intent**     | Accuracy  | **0.929**           | 0.556 (TF-IDF)         | 0.833               |
| **Intent**     | Macro-F1  | **0.924**           | 0.477 (TF-IDF)         | 0.824               |
| **Escalation** | Precision | 0.636               | 0.000 (keyword-rule)   | 0.476               |
| **Escalation** | Recall    | **0.875**           | 0.000 (keyword-rule)   | 0.851               |
| **Escalation** | F1        | **0.737**           | 0.000 (keyword-rule)   | 0.611               |

**Notes on this run:**

- System metrics are on **14/18** items — 4 items hit the Gemini free-tier per-minute quota (15 req/min) and failed after retries. This is the same quota-exhaustion failure mode documented for the full run in `report/REPORT.md` Section 6.
- Escalation recall **0.875** on the demo (7/8 true escalation cases caught, 1 false negative) — consistent with the full-set recall of 0.851, confirming the deliberate recall-over-precision tradeoff.
- The keyword-rule escalation baseline scores **0.000** on the demo (misses all 9 true escalation cases), matching the full-set baseline recall of 0.018 — escalation signals in this domain are contextual, not keyword-triggerable.
- Demo intent accuracy (0.929) runs higher than the full set (0.833), as expected on a small curated subset — the full-set number is the more meaningful one.

<details>
<summary>Demo run — condensed terminal output</summary>

```
ALL STEPS COMPLETE in 3.7 minutes  (DEMO (18 examples))

4/6  Metrics: system intent (14 matched examples)
Accuracy:  0.929
Macro-F1:  0.924
macro avg  precision 0.95  recall 0.93  f1-score 0.92

5/6  Metrics: system escalation (14 matched examples)
Precision: 0.636
Recall:    0.875
F1:        0.737
Confusion matrix: [[7 1], [4 2]]  (1 false negative)

6a  Metrics: baseline intent (18 matched examples)
Accuracy:  0.556
Macro-F1:  0.477

6b  Metrics: baseline escalation (18 matched examples)
Precision: 0.000
Recall:    0.000
F1:        0.000
Confusion matrix: [[0 9], [0 9]]  (9 false negatives)
```

</details>

---

## Quick start — demo in under 10 minutes

The recommended path uses a curated **18-example demo set** that produces the same conclusions as the full evaluation, while staying comfortably inside Gemini free-tier rate limits.

### 1. Clone & set up

```bash
git clone https://github.com/numanmaldar/support-triage-pipeline.git
cd support-triage-pipeline

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set your API key

```bash
export GEMINI_API_KEY="your-key-here"        # PowerShell: $env:GEMINI_API_KEY = "your-key-here"
```

### 3. Run the demo evaluation

```bash
python run_all.py --demo
```

This runs the full pipeline (classify → retrieve + generate → escalate) plus the TF-IDF baseline on the 18-example demo set, and prints intent accuracy, macro-F1, and escalation precision / recall / F1.

**Expected runtime:** 4–10 minutes on the Gemini free tier (depending on rate limits).

**Note:** even the demo can lose a few items to the free-tier per-minute quota (15 req/min) on a heavily used key — the run still completes and scores the successful items, and the conclusions hold at slightly reduced N. Re-running picks up most failed items as the per-minute window resets.

---

## Full evaluation (optional)

The full golden set has **197** hand-labelled examples. Running it end-to-end takes significantly longer (often 30+ minutes) because of free-tier rate limits and daily quotas (15 requests/min, 500 requests/day per model — and each pipeline item issues 2–3 LLM calls).

```bash
python run_all.py --full
```

Full results, per-class breakdowns, a five-mode failure analysis, and the mandatory *"What is misleading about my headline number?"* section are documented in [`report/REPORT.md`](report/REPORT.md). The rationale behind 14 non-obvious design decisions (taxonomy, retrieval filtering, escalation thresholds, and more) is in [`report/decision_log.md`](report/decision_log.md).

---

## Verifying metrics without re-running the pipeline

If you only want to verify the already-computed numbers:

```bash
# System intent
python eval/metrics.py intent \
  --golden eval/golden_set.jsonl \
  --predictions data/processed/predictions.jsonl \
  --true_field true_intent \
  --pred_field prediction.intent \
  --id_field id

# System escalation
python eval/metrics.py escalation \
  --golden eval/golden_set.jsonl \
  --predictions data/processed/predictions.jsonl \
  --true_field escalate_human \
  --pred_field prediction.escalate \
  --id_field id
```

(Demo versions use `eval/demo_golden_set.jsonl` and the corresponding demo predictions file.)

---

## Intent taxonomy

| # | Category | Notes |
|---|---|---|
| 1 | Battery and Power Issues | |
| 2 | Billing and Subscriptions | |
| 3 | Connectivity and Call Failures | |
| 4 | Data Recovery and Syncing | |
| 5 | Hardware Damage and Physical Repair | Strongest class (F1 = 0.97) |
| 6 | OS Performance and Stability | Largest bucket — intentionally broad; ~28% of matched golden items |
| 7 | Other | Catch-all by design; small support, lower scores expected |
| 8 | Product Specs and How-To | |
| 9 | Security and Fraud Reporting | Strong class (F1 = 0.90) |
| 10 | Store and Support Experience | |

**Per-class performance** (system, n=162): strongest on Hardware Damage (0.97 F1), Security and Fraud Reporting (0.90), and Billing (0.90); weakest on *Other* (0.56) and *OS Performance and Stability* (0.82), the latter still the largest bucket at ~28% of matched golden items and still a structural overlap magnet — nearly any bug can be plausibly framed as "since the last update."

---

## Project structure

```
support-triage-pipeline/
├── src/
│   ├── classify.py          # Intent classification (LLM, 10-category taxonomy)
│   ├── retrieve.py          # Embedding retrieval (all-MiniLM-L6-v2, k=3)
│   ├── generate.py          # Reply drafting (retrieval-grounded)
│   ├── escalate.py          # Escalation decision (rules first, LLM fallback)
│   ├── pipeline.py          # End-to-end single + batch runner
│   └── ...
├── eval/
│   ├── golden_set.jsonl           # Full 197 hand-labelled examples
│   ├── demo_golden_set.jsonl      # Curated 18-example subset (for reviewers)
│   ├── metrics.py                 # Intent / escalation metric computation
│   ├── baselines.py               # TF-IDF + LogReg and keyword-rule baselines
│   └── judge.py                   # LLM-as-judge reply scoring
├── data/processed/                # Predictions, baselines, taxonomy, retrieval index
├── report/
│   ├── REPORT.md                  # Full evaluation report incl. failure analysis
│   └── decision_log.md            # 14 non-obvious design decisions
├── run_all.py                     # Single entrypoint (--demo / --full)
└── requirements.txt
```

Each golden-set item is labelled with `true_intent`, `escalate_human` (+ `escalation_reason_human`), and `ideal_reply_notes` — so both intent and escalation are evaluated against human ground truth, not proxy labels.

---

## Design philosophy

- **Decision-support, not full autonomy.** AppleSupport's real-world pattern is funneling customers to DM for device-specific diagnosis. This system drafts and recommends — it does not autonomously send replies or file tickets.
- **Escalation recall over precision.** Missing a genuine security/safety/exhaustion case is far worse than an unnecessary human review. This asymmetry drove threshold and prompt choices throughout.
- **No fine-tuning.** With only 197 golden examples and a tight timeline, prompting + retrieval was the higher-leverage choice over fine-tuning on a tiny dataset.
- **Honest limitations, documented.** Quota coverage gaps, baseline-N mismatches, and taxonomy overloading are disclosed in the report rather than hidden.
- **Validated judging.** LLM-as-judge reply scores were calibrated against human ratings before being used for failure analysis.

---

## Known limitations and honest tradeoffs

- **Free-tier Gemini limits** (15 req/min, 500 req/day) can interrupt batch runs — 35/197 items in the latest full run failed on quota, which is why headline metrics are on 162 items, and 4/18 items in the demo run hit the per-minute cap. The demo path is designed to minimise but not eliminate this.
- **Baseline vs. system N mismatch.** Baselines were scored on all 197 items (no API calls to fail); the system on 162. The comparison establishes a documented floor, not a precise delta.
- **Retrieval index is English-only** → non-English generation currently falls back to a template redirect instead of a substantive reply (classification still works across languages).
- **No multi-turn conversation state** — each message is handled independently, so "I already tried that" context is invisible to the escalation layer.
- **Rule-based safety trigger** currently matches self-harm language but not physical device-hazard language (e.g., electric-shock reports) — a known blind spot flagged as the highest-priority fix.
- **"OS Performance and Stability"** is an overloaded taxonomy bucket that absorbs confusion from adjacent categories.
- **Coverage-gap skew is unverified.** The 35 missing items cluster toward the end of the batch (`~188-197`), consistent with cumulative quota pressure rather than random dropout — though this hasn't been rigorously checked for correlation with intent category or escalation label.

See `report/REPORT.md` Sections 5–7 for the full failure analysis and discussion.

---

## What I'd do next with one more week

1. **Close the remaining coverage gap** — score the missing 35/197 items on a paid tier and check whether they shift any metric. See `report/REPORT.md` Section 6 for the full coverage-gap disclosure.
2. **Fix the safety-rule blind spot** — extend the pattern to physical/device-hazard language.
3. **Rework the escalation prompt** to explicitly weight repeated-contact and exhaustion signals (the biggest lever on recall).
4. **Add lightweight conversation-state tracking** (even just prior-contact counts per author) as an escalation feature.
5. **Extend the retrieval index with non-English historical replies** + language detection in generation.
6. **Split "OS Performance and Stability"** into 2–3 sub-categories with explicit disambiguation criteria.
7. **Add a "no action needed" intent category** so generation stops over-applying the DM-redirect template to non-actionable messages.
8. **Build a held-out test set** (separate from the 197 used for iteration) to verify fixes actually generalise.

---

## Requirements

- Python 3.10+
- A Gemini API key (`GEMINI_API_KEY`) — free tier works for the demo path
- See `requirements.txt` for packages (includes `sentence-transformers` for retrieval)

---

## Acknowledgements

This project was built with the assistance of AI coding tools:

- **Gemini** (`gemini-3.1-flash-lite`) — powers the runtime pipeline: intent classification, retrieval-grounded reply generation, and the LLM fallback layer in the escalation decision. It also served as the LLM-as-judge for reply-quality scoring during evaluation.
- **Claude (Anthropic)** — used as an AI coding assistant during development, for help with code structuring, debugging, documentation drafting, report review, and internal consistency checks on the evaluation write-up.

Data and models used:

- **Customer Support on Twitter** dataset (Kaggle, `thoughtvector/customer-support-on-twitter`) — source of all conversation data and historical AppleSupport replies.
- **`all-MiniLM-L6-v2`** (Sentence-Transformers) — sentence embeddings for the retrieval index.
- **scikit-learn** — TF-IDF vectoriser and Logistic Regression for the intent baseline.

Any other borrowed code or ideas are noted in the relevant source files.

---

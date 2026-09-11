# Support Triage Pipeline

An end-to-end system that reads an incoming customer support message and automatically:

1. **Classifies intent** into one of 10 support categories
2. **Retrieves relevant grounding context** and **generates a draft reply**
3. **Decides whether the case needs human escalation**, and why

Built and evaluated against a hand-labeled golden set of 197 real support messages, with results benchmarked against a TF-IDF baseline and an LLM-judge scoring pipeline calibrated against human ratings.

---

## Why this exists

Support teams triage a high volume of repetitive, low-context messages (password resets, "how do I..." questions, billing confusion) alongside a small number of genuinely urgent ones (fraud reports, safety issues, an already-frustrated customer on their third contact). The goal here wasn't just "can an LLM classify a support ticket" — it was to build something closer to what a real triage system needs to get right: **know what a message is about, draft something useful, and know when to get out of the way and hand it to a human.**

This project treats that as three separable, individually-gradable problems (classification, generation, escalation) glued together by one pipeline, evaluated against ground truth a human actually labeled — not against the LLM's own confidence in itself.

---

## Architecture

```
message
   │
   ▼
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  classify.py │────▶│     generate.py       │────▶│   escalate.py     │
│  intent +    │     │  retrieve.py (k=3) +   │     │  rule layer +     │
│  confidence  │     │  grounded draft reply  │     │  LLM fallback     │
└─────────────┘     └──────────────────────┘     └──────────────────┘
```

- **`classify.py`** — LLM call against a fixed taxonomy (`data/processed/taxonomy_final.json`), returns intent label, a confidence tier, and a short reasoning string.
- **`retrieve.py`** — embeds the incoming message (`all-MiniLM-L6-v2`) and pulls the top-k=3 most similar historical support threads from a pre-built local index, used as grounding context.
- **`generate.py`** — drafts a reply using the retrieved context, so responses are grounded in real prior resolutions rather than the model improvising from scratch.
- **`escalate.py`** — a two-layer decision: a fast rule-based check for known high-risk patterns, falling back to an LLM judgment call using intent + classifier confidence when no rule fires.
- **`pipeline.py`** — single entrypoint tying all four stages together, runnable per-message or in batch over a JSONL file.

All four stages call **`gemini-3.1-flash-lite`** via `llm_client.py`, which wraps the call with retry/backoff logic (see [Known limitations](#known-limitations-and-honest-tradeoffs) — this mattered more than expected).

---

## Repo structure

```
├── src/
│   ├── pipeline.py          # orchestrates the full flow, single + batch modes
│   ├── classify.py          # intent classification
│   ├── retrieve.py          # embedding-based retrieval over historical threads
│   ├── generate.py          # grounded reply generation
│   ├── escalate.py          # rule + LLM escalation decision
│   ├── llm_client.py        # Gemini API wrapper w/ retry logic
│   ├── taxonomy.py          # taxonomy loading/validation
│   ├── ingest.py            # raw data → processed threads
│   ├── sample_golden_set.py # sampling strategy for the eval set
│   └── label_golden_set.py  # human-labeling helper tool
├── eval/
│   ├── golden_set.jsonl             # 197 hand-labeled examples (ground truth)
│   ├── baselines.py                 # TF-IDF + majority-class baselines
│   ├── judge.py                     # LLM-as-judge scoring + human calibration
│   ├── metrics.py                   # intent + escalation metrics (accuracy, F1, precision/recall)
│   └── human_calibration_scores.jsonl
├── data/
│   ├── raw/                 # source data (gitignored — see below)
│   └── processed/
│       ├── predictions.jsonl          # system outputs on the golden set
│       ├── baseline_predictions.jsonl # baseline outputs on the golden set
│       ├── judge_scores.jsonl         # LLM-judge scores per reply
│       ├── taxonomy_final.json        # the 10-category intent taxonomy
│       └── brand_stats.csv
├── report/
│   ├── REPORT.md             # full write-up: methodology, results, failure analysis
│   └── decision_log.md       # running log of design decisions and why
└── requirements.txt
```

Large/regenerable artifacts (`data/raw/`, the retrieval embedding index, `threads.jsonl`, the trained baseline `.pkl`) are gitignored — they're either not redistributable or fully reproducible from the tracked source files. Everything needed to **verify the reported results without rerunning the pipeline** — predictions, baseline outputs, judge scores, taxonomy — is tracked and included.

---

## Setup and reproduction

```bash
git clone https://github.com/numanmaldar/support-triage-pipeline.git
cd support-triage-pipeline
pip install -r requirements.txt
```

You'll need a Gemini API key (free tier works, see [limitations](#known-limitations-and-honest-tradeoffs) below for why that matters):

```bash
export GEMINI_API_KEY="your-key-here"     # macOS/Linux
$env:GEMINI_API_KEY = "your-key-here"     # Windows PowerShell
```

**Run the pipeline on a single message:**
```bash
python src/pipeline.py single --message "my phone won't stop restarting"
```

**Run the full batch evaluation:**
```bash
python src/pipeline.py batch --input eval/golden_set.jsonl --output data/processed/predictions.jsonl
python eval/baselines.py train --threads data/processed/threads.jsonl --out data/processed/baseline_intent_model.pkl
python eval/baselines.py predict --model data/processed/baseline_intent_model.pkl --golden eval/golden_set.jsonl --out data/processed/baseline_predictions.jsonl
python eval/metrics.py intent --golden eval/golden_set.jsonl --predictions data/processed/predictions.jsonl --true_field true_intent --pred_field prediction.intent --id_field id
python eval/metrics.py escalation --golden eval/golden_set.jsonl --predictions data/processed/predictions.jsonl --true_field escalate_human --pred_field prediction.escalate --id_field id
```

Expect this to take 15-25+ minutes on the free API tier due to rate limits — see below.

---

## Results

### Intent classification (153/197 matched — see limitations)
| Metric | Score |
|---|---|
| Accuracy | 0.856 |
| Macro-F1 | 0.842 |

Strong across most categories (0.90+ F1 on Billing, Hardware Damage, Security/Fraud). Weakest on "Other" (F1 0.50, only 6 support examples) and confusion between "OS Performance and Stability" and adjacent categories — see failure mode #3 below.

### Escalation decision (153/197 matched)
| Metric | Score |
|---|---|
| Precision | 0.450 |
| Recall | **0.857** |
| F1 | 0.590 |

Recall was the metric to optimize for here, not precision — a missed escalation (a real fraud/safety case handled as routine) is a materially worse failure than an unnecessary human review. The system catches 36/42 true escalation cases; the 6 false negatives are analyzed in the failure modes below.

### LLM-judge calibration against human ratings
30 replies double-scored by both the LLM judge and a human rater on groundedness:
- Exact match: 63.3%
- Within 1 point (5-point scale): **90.0%**

Used to validate that the LLM-judge scores driving the qualitative failure analysis below are a reasonable proxy for human judgment, not just the model grading its own homework.

Full breakdown, confusion matrices, and per-category numbers are in [`report/REPORT.md`](report/REPORT.md).

---

## Failure analysis — 5 real failure modes

Pulled from actual golden-set failures, not hypothesized. Full detail and specific example IDs in the report; summary:

1. **Escalation logic misses "already exhausted troubleshooting" signals.** All 8 escalation false negatives share a pattern: prior failed troubleshooting, a blocked self-service path, or explicit frustration language that a human reader picks up on but the LLM layer classifies as routine.
2. **Non-English messages get a templated redirect instead of a real answer.** The two lowest-scored replies overall are non-English (French, Spanish) — intent classification correctly identifies them, but generation falls back to an English-support boilerplate because the retrieval index is English-only.
3. **"OS Performance and Stability" is an overloaded taxonomy bucket.** It's the largest category (56/197) and the single most-confused-with-everything-else label — a taxonomy design limitation ("since the update..." fits almost any bug), not purely a classifier failure.
4. **Generic "DM us for help" filler gets applied to non-actionable messages.** The two worst-scored replies were pure venting or simple feedback that needed no support action at all, but got the system's safest default template anyway — high on safety, near-zero on actionability.
5. **The rule-based safety-escalation trigger has a blind spot for physical hazards.** The regex layer only catches self-harm language, not device-safety complaints (e.g. a literal shock hazard report) — meaning a genuinely urgent case could rely entirely on the LLM fallback layer, which (per #1) has its own demonstrated gaps.

---

## Known limitations and honest tradeoffs

**44/197 golden-set items (22%) were not scored due to Gemini free-tier API quota exhaustion, not a code defect.** The reported metrics above are computed on the 153 items that did succeed. This is worth being direct about rather than glossing over:

- The free tier caps `gemini-3.1-flash-lite` at both 15 requests/minute *and* 500 requests/day per project. Each pipeline call fires 2-3 LLM calls (classify, generate, escalate), so a 197-item batch run needs 400-600+ calls — well past the daily cap even before accounting for retries.
- I built a resume script (`resume_fails.py` — kept out of the tracked repo since it was diagnostic tooling, not part of the core pipeline) that correctly identified all 44 failed items and reprocessed only those, with proper rate-limit spacing — but hit the *daily* cap, not the per-minute one, which no amount of spacing fixes. That requires either a quota reset, a second API key, or enabling billing.
- **What I'd do differently in production:** track spend/quota headroom before a batch run, not after; use a job queue with persistent backoff state instead of an in-process retry loop, so a day-boundary quota reset doesn't require restarting from scratch; and treat "prediction contains an error object" as a top-level, not nested, signal so eval tooling catches it immediately instead of silently under-counting (an issue I found the hard way — my first sanity check script only looked for `error` at the JSON record's top level, missing all 44 nested failures).

I'm including this section deliberately. A take-home that reports suspiciously clean 100%-coverage numbers is less convincing than one that shows the actual failure mode of the infrastructure it depends on, and how that failure was diagnosed and handled.

---

## Design decisions

The reasoning behind taxonomy choices, retrieval design, escalation thresholding, and other calls made along the way are logged as they happened in [`report/decision_log.md`](report/decision_log.md) — intended to show the process, not just the destination.

---

## Full report

[`report/REPORT.md`](report/REPORT.md) has the complete write-up: methodology, per-category metrics, confusion matrices, the human-calibration study, and the full failure analysis with specific example IDs from the golden set.

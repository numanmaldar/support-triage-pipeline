# Decision Log

1. **Brand: AppleSupport.** Chosen over AmazonHelp (higher volume but more
   generic/logistics-heavy) because AppleSupport's issues cluster into clean,
   distinguishable technical intents (iOS bugs, account, hardware, billing)
   well suited to a small taxonomy.

2. **"Resolved" heuristic is measuring the wrong thing.** Initial heuristic
   (`brand_had_last_word` OR thanks-like reply) returns 93.3% "resolved" on
   real AppleSupport threads. Manual inspection of sampled threads shows the
   dominant real pattern is AppleSupport asking a triage question or
   redirecting to DM as its last visible tweet -- not an actual fix. The
   heuristic conflates "brand responded" with "issue solved." Kept the
   heuristic (cheap, reproducible) but will report the true meaning
   explicitly in REPORT.md's "misleading headline number" section, and will
   NOT claim the agent's replies are grounded in "how issues were solved" --
   only in "how AppleSupport typically triages/responds to this type of
   issue on-platform," since the actual resolution mostly happens in DMs
   invisible to this dataset.

3. **Scope narrowing from (2):** the reply-drafter's job is to produce a
   realistic, on-brand *first response* (ask the right diagnostic question,
   or correctly redirect to DM/official support for account-specific
   issues) -- not to claim to solve the customer's problem outright. This is
   stated explicitly in problem framing so it isn't mistaken for a flaw
   discovered later.

4. **Merged "Keyboard and Autocorrect Glitches" into "OS Performance and
   Stability."** The proposed cluster was dominated by a single time-bound
   bug (iOS 11.1's "I" -> "?" autocorrect bug, Oct/Nov 2017). Keeping it as
   a permanent top-level intent would overfit the taxonomy to this dataset's
   specific time window rather than generalizing to future AppleSupport
   traffic. Folded into the broader "update-triggered bug" category instead.

5. **Added an explicit "Other" bucket** rather than forcing every message
   into one of 9 categories. Draft taxonomy sample flagged non-English
   messages and non-actionable venting as a real, non-trivial chunk of
   traffic -- forcing these into a wrong category would silently corrupt
   both the classifier's training signal and the golden set later.

6. **Non-English messages: classified directly via Gemini's multilingual
   ability, not routed to "Other" by default.** Adds a caveat for the
   report: classifier accuracy on non-English messages is not separately
   verified in this dataset (English speaker labeling the golden set), so
   this is a known blind spot to disclose, not a validated capability.

7. **Retrieval index restricted to resolved=true threads only**, despite
   knowing the "resolved" heuristic is weak (see decision #2). Chose the
   stricter filter anyway because it's a directional quality signal even
   if noisy -- better than indexing threads that visibly ended in
   frustration. Re-quantify in the report: what fraction of "resolved"
   threads in the index are actually just "brand asked a triage question
   last" vs. a real substantive reply, based on manual spot-checks.

8. **Retrieval quality validated by manual spot-check.** Query "my phone
   won't stop restarting" -> top-3 matches (similarity 0.87-0.90) were all
   semantically correct AND contained substantive troubleshooting content,
   not generic DM-redirects. This nuances decision #2/#7: the "resolved"
   heuristic is unreliable in aggregate, but a meaningful subset of
   resolved threads do carry real, groundable brand guidance -- retrieval
   surfaces those specifically because it's matching on content similarity,
   not just the resolved flag.

9. **Escalation dollar threshold set at $50.** Arbitrary but stated
   explicitly: below this, letting the AI draft a DM-redirect response is
   low-risk; above it, a human should be in the loop for refund/billing
   disputes. Worth sensitivity-testing against the golden set later --
   flag in the report if $50 turns out too low/high based on false
   escalation rate.

10. **Escalation logic defaults to escalate=True on any parse failure or
    LLM ambiguity**, consistent with the recall-over-precision design
    choice: an unnecessary escalation costs a human a few seconds of
    review; a missed one risks real customer harm going unhandled.

11. **PowerShell $-interpolation silently strips dollar amounts from
    double-quoted CLI args.** Discovered when the $120 escalation-rule
    test fell through to the LLM layer instead of the dollar-threshold
    rule. Root cause: PowerShell, not the code -- $120 in a double-quoted
    string is treated as a variable reference. Fixed by using single
    quotes for any test message containing a literal $ sign. Noted in
    README as a Windows-specific gotcha for anyone reproducing results.

12. **Fixed keyword bucket overlap in golden-set sampling.** Both "Battery"
    and "Billing" patterns matched charg* (charging vs. being charged
    money) -- a real ambiguity in English, not just a regex bug. Caught it
    via a synthetic smoke test before running on real data, where it would
    have silently mislabeled the sampling hint (not the final label, since
    a human assigns true_intent anyway, but it would have skewed which
    examples got sampled into each bucket).

13. **~50% of AppleSupport traffic didn't match any keyword bucket** during
    golden-set sampling. Not evidence the taxonomy is wrong (keyword
    matching is a coarse proxy, and the LLM classifier handles semantics
    far better), but real signal that a meaningful share of messages are
    short, vague, or emotionally-phrased in ways that resist simple
    keyword rules -- worth a mention in problem framing.
   
14. Predictions were generated across TWO different Gemini models
    (gemini-3.6-flash for the first ~N items, gemini-3.1-flash-lite for
    the rest) due to free-tier daily quota limits hit mid-run. This is a
    real inconsistency worth disclosing in the report -- reply quality/
    style may differ slightly between the two models, which is a
    legitimate confound when interpreting per-example results. Not
    ideal, but disclosed honestly rather than hidden.
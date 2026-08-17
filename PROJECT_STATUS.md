# Blunt project status — read before you present

Written for the team, not the judges. This is the "what's real, what's
fragile, what NOT to say without a caveat" document.

## What is solid (say this with confidence)

- **The pipeline runs end to end and is reproducible.** `python
  generate_submission.py` produces exactly 24,000 rows; all validation
  checks pass (probabilities sum to 1, `review_priority` in [0,1], every
  template (candidate_id, phase) pair covered). Numbers reproduce the
  README exactly. Nothing here is faked or hard-coded.
- **Entity linkage is real and deliberately conservative.** DOB+lastname
  where available, fuzzy first-name blocking otherwise, and anything
  ambiguous is left unmatched rather than guessed.
- **Probabilities are genuinely calibrated** (`CalibratedClassifierCV`,
  Platt scaling) with Brier scores reported — not raw tree-vote fractions.
- **Every feature is auditable** back to a specific source, state, and date
  via the evidence trail. The demo shows this live.
- **The demo works.** Renders with 0 exceptions, interactive, loads
  instantly from precomputed data (no live model run to fail on stage).

## What is fragile or oversold (caveat these, or a sharp judge will)

1. **The headline "45% accuracy" is statistically fragile.**
   On the single 75-case holdout the pipeline reports, McNemar's paired test
   vs. the majority-class baseline gives **p = 0.31 — NOT significant.** That
   specific number cannot support "our model beats baseline."
   **Say instead:** cross-validated over all 300 labels, the model beats a
   majority-class baseline **50% vs 35%, paired p ≈ 0.001**. That IS a real,
   consistent edge. (Caveat even there: the 5-fold paired t-test is mildly
   optimistic because folds share data — but +15 pts with ~3% std is solid.)

2. **~50% on 3 classes is triage, not accuracy.** It is meaningfully above
   the 33% chance / 35% baseline, but it is NOT an accurate classifier.
   Frame the whole product as **decision-support / queue prioritization**,
   never as automated determinations. (The README already does this — good.)

3. **The explanations describe the EVIDENCE, not the model's reasoning.**
   This is the most important caveat. Whether Gemini or the template writes
   it, the paragraph is a plain-English summary of the evidence trail. It is
   **not** a faithful account of why the RandomForest produced that
   probability. **Do NOT say** "the AI explains why the model decided X."
   **Do say** "the AI turns the evidence behind the flag into plain language
   a reviewer can act on."

4. **`insufficient_evidence` is the weakest class (recall ~0.23)** and the
   name oversells. In practice many "insufficient" cases have 5 linked
   sources that simply *conflict* — so it means "ambiguous/conflicting,"
   not "no data." The model is worst at exactly the "we're unsure" bucket.

5. **`review_priority` weights are hand-tuned, not learned.** 0.6 /
   0.25 / 0.15 and the 0.5 low-evidence discount are reasonable heuristics
   but arbitrary and unvalidated. Don't call them "optimized."

6. **"34.5% of cases changed class T0→T1" cuts both ways.** We frame it as
   "responsive to new evidence" (true), but a third of predictions flipping
   on an evidence refresh is also model instability on borderline cases. Be
   ready if a judge reads it the second way.

7. **Linkage precision is designed-for, not measured.** ~30% of source rows
   are unmatched, which is fine (some belong to people outside the pool),
   but there is no linkage ground truth, so we cannot actually quote a
   false-match rate. Don't claim high linkage precision as a proven number.

8. **Gemini is optional and may be running on the fallback.** If the AI
   Studio / GCP key isn't loaded on the demo laptop, explanations come from
   the deterministic template (clearly labeled in the UI). That's safe — but
   if you tell judges "it uses Gemini," verify the green banner is showing
   first, or say "LLM-generated, with a deterministic fallback."

Nothing in the codebase is fabricated — the sins here are **framing**, not
fake numbers.

## Fastest fixes, ranked

1. **(Free, do tonight) Fix the reporting, not the model.** Lead with the
   5-fold CV result (50% vs 35% baseline, p ≈ 0.001), not the underpowered
   75-case split. Update the README validation section to match. This alone
   turns "not significant" into "significant" honestly, using the same model
   and data. See `significance_test.py`.

2. **Do NOT swap classifiers — it's proven not to help.**
   `experiment_model.py` CV-tested RF variants, ExtraTrees, Hist/Gradient
   Boosting, and balanced LogReg: all within ±0.5% of the current 50%. The
   ceiling is set by the features and the 300 labels, not the model.

3. **(If you have hours, not minutes) Real accuracy gains must come from
   features or links**, per the README's own next-steps: temporal features
   (address-change rate, gaps between events), and recovering some of the
   ~30% unmatched source rows via partial address / vehicle_ref
   corroboration. Higher risk, higher payoff. Not a Wednesday-morning task.

## Current state of the 5 asks

1. Pipeline runs, all validations pass. ✅
2. `generate_case_explanation()` written + tested on 3 real cases (one per
   class). Runs live on Gemini when a key is present; verified on the
   deterministic fallback now. ✅ (live-Gemini pending a working key)
3. Streamlit demo built and verified (0 render exceptions, interactive). ✅
4. Significance test done; verdict + fastest fix above. ✅
5. This document. ✅

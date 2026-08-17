# Out-of-State Tag Holder Review — Methodology

## What this system does

For each of the 12,000 candidate cases in `Data_T0/candidate_records.csv`,
this system produces a review classification (`review_warranted`,
`review_not_warranted`, or `insufficient_evidence`), calibrated class
probabilities, and a `review_priority` score — first using only T0
evidence, then again after incorporating T1 evidence
(`Data_T1/evidence_update_stream.csv`).

This is a decision-support tool: it triages cases for staff attention.
It does not make a legal, residency, fee, or enforcement determination.

## Pipeline

```
src/link_entities.py   Step 1: entity linkage
src/features.py        Step 2: per-source evidence features
src/pipeline_t0.py      Step 3: model selection, calibration, T0 scoring
src/link_t1.py          Step 4: T1 evidence linkage and merge
src/generate_submission.py  Step 5: T1 scoring + case_predictions.csv
```

Run `python3 src/generate_submission.py` to reproduce everything end to
end, including validation checks on the output file.

### Step 1 — Entity linkage

Only `candidate_records.csv` carries a clean `candidate_record_id`. The
five other source tables (address history, license/credential events,
vehicle title events, work location signals, external context signals)
identify people only by name (+ date of birth where present), and those
names contain realistic data-quality noise: truncation (e.g. "N" instead
of "Nwzgpc"), case differences, and small typos.

- **Where DOB is available** (license_id_events): the join key is
  `(normalized last_name, date_of_birth)`, which is unique for
  11,998/12,000 candidates.
- **Where DOB is not available** (address_history,
  external_context_signals, work_location_signals,
  vehicle_title_events): we block on normalized last name, then
  disambiguate using fuzzy first-name matching (exact match, prefix/
  truncation match, or bounded edit distance ≤2).
- **Conservative by design**: any row that can't be resolved to exactly
  one candidate is left unmatched rather than guessed. A wrong link
  would poison every downstream feature, so precision is prioritized
  over recall in linkage.

Resulting match rates (T0):

| Source | Match rate |
|---|---|
| license_id_events | 70.9% |
| address_history | 68.1% |
| external_context_signals | 68.0% |
| work_location_signals | 88.6% |
| vehicle_title_events | 68.1% |

The remaining unmatched rows are expected, not a linkage failure: these
source files plausibly contain records for people outside the 12,000
candidate pool (a realistic feature of shared administrative data), and
a small number of rows are too corrupted to safely resolve.

### Step 2 — Feature construction

**Key design decision:** a candidate's "home state" for comparison
purposes is their OWN currently tagged/observed state
(`candidate_records.observed_state`) — NOT a fixed "DE". Only ~50%
(5,956/12,000) of candidates are currently DE-tagged; the rest carry
PA/MD/NJ/NY/VA/FL/NC/SC tags. The operational question ("Out-of-State
Tag Holder Review") is whether a person currently tagged in another
state shows evidence of actually residing in Delaware and therefore
owing DE registration — so evidence is compared against each
candidate's own tag state, and Delaware-pointing evidence is tracked
as its own explicit signal.

For each candidate, we find the MOST RECENT dated event in each of the
five sources and compute:

- whether that source's state matches the candidate's tag state
- whether that source's state is specifically DE
- a confidence weight (linkage confidence × source-specific quality,
  e.g. license status active/expired, external evidence_quality
  standard/limited)
- a recency weight (exponential decay, ~1-year half-life, relative to
  the phase's as-of date)

These produce a 30-feature vector per candidate (source presence,
state-match, DE-match, and weighted-recency per source, plus aggregate
counts) rather than a single hand-tuned "mismatch score" — this lets
the classifier learn which source combinations matter instead of
relying on a manually guessed formula.

Every feature is traceable back to a specific source record, state, and
date via the `evidence_trail` field for auditability.

### Step 3 — Model selection and calibration

Only 300 of 12,000 cases are labeled (`Development_Labels.csv`) — a
weakly-supervised setting. We:

1. 5-fold stratified cross-validation across logistic regression
   (two regularization strengths), random forest, and gradient
   boosting.
2. Select the best (RandomForest, ~50% CV accuracy vs. 33% random
   baseline for 3 balanced classes).
3. Wrap it in `CalibratedClassifierCV` (sigmoid/Platt scaling) so
   output probabilities are honest rather than raw tree-vote fractions.
4. Report held-out accuracy and per-class Brier scores for
   transparency (see below).

**Honest limitation:** held-out accuracy is ~45-50%. This is a hard
problem from sparse, weakly-linked, synthetic evidence with only 300
labels — meaningfully better than chance, but not highly precise.
`insufficient_evidence` recall in particular is the weakest class,
which is arguably appropriate: it's the hardest class to detect from
features alone, and the system leans toward flagging uncertainty rather
than forcing a confident guess.

### Step 4 — T1 evidence incorporation

`evidence_update_stream.csv` rows are linked the same conservative way
(67.0% match rate) and merged into the candidate's evidence lists by
source domain (`address`, `license`, `title`→vehicle, `external`; no
`work` domain appears in T1). All three record actions (`new_record`,
`status_update`, `record_correction`) are appended as new dated events;
because feature-building always uses the MOST RECENT dated event per
source, later evidence naturally supersedes earlier evidence as long as
its effective_date is more recent. This is a simplifying assumption —
we do not attempt field-level patching of specific prior records — and
is noted here for auditability.

### Step 5 — review_priority

`review_priority` is NOT simply a copy of `p_review_warranted`. It
combines:

- `p_review_warranted` (60% weight) — the primary driver
- normalized evidence strength / total corroborating weight (25%) — a
  60%-confident flag backed by 5 sources is more actionable than a
  60%-confident flag backed by 1
- number of contributing sources (15%)
- a 50% priority discount when `insufficient_evidence` is itself the
  top class, so low-evidence cases don't crowd the review queue ahead
  of well-evidenced ones

## Validation results (T0, held-out 25% split of the 300 dev labels)

```
                       precision    recall  f1-score   support
insufficient_evidence       0.30      0.23      0.26        26
 review_not_warranted       0.52      0.54      0.53        26
     review_warranted       0.50      0.61      0.55        23
              accuracy                           0.45        75
```

Brier scores (lower = better calibrated; 0.25 ≈ coin-flip):
insufficient_evidence 0.228, review_not_warranted 0.194,
review_warranted 0.181.

### Is the improvement statistically real?

The single 75-case split above is underpowered: on it, McNemar's paired
test against a majority-class baseline is **not** significant (p ≈ 0.31).
The honest, better-powered evidence is 5-fold cross-validation over all
300 labels (`significance_test.py`):

```
Model (RandomForest):     50.0% (+/- 3.5%)
Majority-class baseline:  35.0%
Paired t-test across folds: p ≈ 0.001
```

So the ~15-point improvement over baseline is real and consistent under
cross-validation, even though any single small hold-out estimate is noisy.
This is triage that meaningfully beats chance — not an accurate classifier.
A model search (`experiment_model.py`) confirms the ~50% ceiling is set by
the features and the 300 labels, not the choice of classifier (RF,
ExtraTrees, boosting, and balanced LogReg all land within ±0.5%).

T0 → T1: 34.5% of the 12,000 candidates changed predicted class after
incorporating T1 evidence, indicating the system is responsive to new
evidence rather than static (though on borderline cases this also reflects
prediction instability).

## Known limitations / next steps

- Entity linkage leaves ~30% of source rows unmatched; a probabilistic
  record-linkage approach (e.g. weighting partial address/vehicle_ref
  corroboration) could recover more without sacrificing precision.
- Held-out accuracy (~45-50%) leaves room for improvement; richer
  temporal features (e.g. rate of address changes, gaps between events)
  were not explored due to time constraints.
- `record_correction` events are treated identically to
  `status_update`/`new_record`; a more faithful implementation would
  overwrite/void the specific prior record it corrects.
- This tool is not a substitute for human judgment and does not access
  or infer any information beyond what is in the provided synthetic
  dataset.

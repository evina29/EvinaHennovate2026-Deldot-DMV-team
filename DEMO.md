# Demo runbook — DMV Out-of-State Tag Review

Live, judge-facing Streamlit app. Four tabs: a **Morning Worklist** of the
highest-priority cases, a **Case File** view (class, calibrated probabilities,
review priority, evidence timeline, plain-English explanation), a **New
Evidence (T0 → T1)** before/after view, and a **Model Honesty** panel with the
real cross-validated confusion matrix and accuracy.

## One-time setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the challenge data in place (unzip the challenge zip so this exists):
#    ./challenge_data/Data_T0/candidate_records.csv   (etc.)

# 3. Generate the submission + the demo's data files
python generate_submission.py     # writes case_predictions.csv (24,000 rows)
python export_case_data.py        # writes demo_data/case_data.json + sample_ids.json
python export_metrics.py          # writes demo_data/metrics.json (honesty panel)
```

`case_data.json` and `metrics.json` are what the app reads at runtime, so the
demo loads instantly and never re-runs the model live. `challenge_data/` is
NOT needed once those files exist.

## (Optional) Enable live Gemini explanations

Without a key the app still works — it generates grounded, plain-English
explanations from a built-in template and labels them "template fallback".
To use live Gemini instead:

```bash
cp .env.example .env
# edit .env and paste your AI Studio / GCP "Generative Language API" key:
#   GEMINI_API_KEY=AIza...
```

## Run the demo

```bash
streamlit run app.py
# opens http://localhost:8501
```

## Flagship view: the deployed console (offline HTML)

The most polished, judge-facing view is a single self-contained file styled like
a real internal DMV case-management console — sidebar navigation, a sortable
data table, muted enterprise palette, monospace data fields, an audit-log
evidence trail, and the AI explanation styled as an analyst note. It needs
**nothing installed** — just double-click it:

```bash
python export_html.py     # writes dmv_demo.html (needs the two data files above)
# then double-click dmv_demo.html — opens in any browser, no server, no network
```

Same four sections and the curated + worklist cases, with explanations
pre-generated and baked in. Because it has no dependencies, it is also the
**can't-fail backup** if Python/Streamlit misbehaves on the day — keep it open
in a browser tab.

The Streamlit app (`app.py`) shares the same restrained design system and adds
**live** AI note generation when a key is set; use it to show the note being
produced live, and the HTML console for the primary visual walkthrough.

## Suggested live flow (about 3 minutes)

1. **Morning Worklist** — "This is a reviewer's queue for the day, ranked by
   priority." Click **Open →** on the top case.
2. **Case File** — walk the evidence timeline, then click **Generate
   explanation** to show the plain-English writeup.
3. **New Evidence (T0 → T1)** — load the ⭐ flip case (`CAN-CKLXPC2OP3`): at T0
   there isn't enough to act; at T1 a Delaware title correction arrives and it
   becomes REVIEW WARRANTED. "The system responds to new evidence."
4. **Model Honesty** — show the confusion matrix and own the ~50%-on-3-classes
   number directly: "a triage aid that prioritizes human review, not an
   automated decision-maker."

Curated demo cases (sidebar buttons, all hand-picked from real output):

| Case | Scenario |
|---|---|
| `CAN-CKLXPC2OP3` ⭐ | Flips insufficient → warranted when T1 evidence arrives |
| `CAN-6QEV4RPCT3` | Strong review-warranted (three sources point to DE) |
| `CAN-8L913O7UT4` | Clear no-review-needed (all sources agree) |
| `CAN-3EA4YQEIXT` | Genuinely insufficient (sources conflict) |

## Files

| File | Purpose |
|---|---|
| `generate_submission.py` | End-to-end pipeline → `case_predictions.csv` |
| `link_entities.py`, `features.py`, `pipeline_t0.py`, `link_t1.py` | Pipeline stages |
| `explain.py` | `generate_case_explanation()` (Gemini + template fallback) |
| `export_case_data.py` | Runs pipeline once, dumps `demo_data/case_data.json` |
| `export_metrics.py` | Cross-validated confusion matrix → `demo_data/metrics.json` |
| `demo_cases.py` | Hand-picked candidate IDs for the sidebar quick-access buttons |
| `app.py` | The Streamlit demo (4 tabs) |
| `test_explanations.py` | Prints explanations for 3 real cases (one per class) |
| `significance_test.py` | Model-vs-baseline statistical tests |
| `experiment_model.py` | Fast CV comparison of alternative models |

## Talking points / honest caveats

See `PROJECT_STATUS.md` for the blunt version of what's solid, what's
fragile, and what NOT to say to a judge without a caveat.

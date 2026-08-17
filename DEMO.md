# Demo runbook — DMV Out-of-State Tag Review

Live, judge-facing Streamlit app. Enter a `candidate_record_id` and see its
predicted classification, calibrated class probabilities, review priority,
the per-source evidence trail, and a plain-English explanation for a
non-technical DMV reviewer.

## One-time setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the challenge data in place (unzip the challenge zip so this exists):
#    ./challenge_data/Data_T0/candidate_records.csv   (etc.)

# 3. Generate the submission + the demo's data file
python generate_submission.py     # writes case_predictions.csv (24,000 rows)
python export_case_data.py        # writes demo_data/case_data.json + sample_ids.json
```

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

The app shows a green "Gemini connected" banner if a key is loaded, or an
amber "template fallback" banner if not. Example case IDs (one per class)
are one click away under **Try an example**.

## Files

| File | Purpose |
|---|---|
| `generate_submission.py` | End-to-end pipeline → `case_predictions.csv` |
| `link_entities.py`, `features.py`, `pipeline_t0.py`, `link_t1.py` | Pipeline stages |
| `explain.py` | `generate_case_explanation()` (Gemini + template fallback) |
| `export_case_data.py` | Runs pipeline once, dumps `demo_data/case_data.json` for the app |
| `app.py` | The Streamlit demo |
| `test_explanations.py` | Prints explanations for 3 real cases (one per class) |
| `significance_test.py` | Model-vs-baseline statistical tests |
| `experiment_model.py` | Fast CV comparison of alternative models |

## Talking points / honest caveats

See `PROJECT_STATUS.md` for the blunt version of what's solid, what's
fragile, and what NOT to say to a judge without a caveat.

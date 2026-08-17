"""
Test generate_case_explanation() on 3 REAL cases pulled from our exported
pipeline output (one per predicted class, T1 phase). Prints the evidence
that was fed in and the generated explanation for each.

Runs with Gemini if GEMINI_API_KEY / .env is present, otherwise prints the
deterministic fallback (and says which path was used).
"""
import json
import sys

from explain import generate_case_explanation

PHASE = "T1"


def load():
    with open("demo_data/case_data.json") as fh:
        data = json.load(fh)
    with open("demo_data/sample_ids.json") as fh:
        samples = json.load(fh)
    return data, samples


def evidence_for(data, cid, phase=PHASE):
    rec = data[cid]
    p = rec[phase]
    return {
        "candidate_record_id": cid,
        "home_state": p.get("home_state") or rec.get("observed_state"),
        "predicted_class": p["predicted_class"],
        "probabilities": p["probabilities"],
        "review_priority": p["review_priority"],
        "evidence_trail": p.get("evidence_trail", []),
        "per_source": p.get("per_source", {}),
        "n_sources": p.get("n_sources", 0),
    }


def main():
    data, samples = load()
    # one real case per class
    picks = {}
    for cls in ["review_warranted", "review_not_warranted", "insufficient_evidence"]:
        picks[cls] = samples[cls][0]

    for cls, cid in picks.items():
        ev = evidence_for(data, cid)
        print("=" * 74)
        print(f"CLASS: {cls}   CASE: {cid}   (phase {PHASE})")
        print("-" * 74)
        print(f"Tag/observed state: {ev['home_state']}")
        pr = ev["probabilities"]
        print(f"Probabilities: warranted={pr['review_warranted']:.0%}  "
              f"not_warranted={pr['review_not_warranted']:.0%}  "
              f"insufficient={pr['insufficient_evidence']:.0%}")
        print(f"review_priority: {ev['review_priority']}")
        print("Evidence trail:")
        for t in ev["evidence_trail"]:
            print(f"   - {t}")
        if not ev["evidence_trail"]:
            print("   (none linked)")
        print("-" * 74)
        text, src = generate_case_explanation(ev)
        print(f"EXPLANATION  [source = {src}]:")
        print(text)
        print()


if __name__ == "__main__":
    main()

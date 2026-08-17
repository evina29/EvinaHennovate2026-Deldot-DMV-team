"""
Run the full pipeline once and export everything the demo needs
(per-candidate class, calibrated probabilities, review_priority, and the
per-source evidence trail for BOTH T0 and T1) to demo_data/case_data.json.

This decouples the Streamlit demo from a live model run: the app just loads
this JSON, so it starts instantly and can't fail mid-demo on a model call.
"""
import json
import os

from generate_submission import main as run_pipeline


def phase_payload(feats, scores, cid):
    f = feats[cid]
    pred_class, proba_dict, priority = scores[cid]
    return {
        "predicted_class": pred_class,
        "probabilities": {
            "review_warranted": round(proba_dict.get("review_warranted", 0.0), 6),
            "review_not_warranted": round(proba_dict.get("review_not_warranted", 0.0), 6),
            "insufficient_evidence": round(proba_dict.get("insufficient_evidence", 0.0), 6),
        },
        "review_priority": priority,
        "home_state": f.get("home_state"),
        "n_sources": f.get("n_sources"),
        "total_weight": f.get("total_weight"),
        "net_score": f.get("net_score"),
        "evidence_trail": f.get("evidence_trail", []),
        "per_source": f.get("per_source", {}),
    }


def main():
    out_path, t0_result, scores_t0, scores_t1, feats_t0, feats_t1 = run_pipeline()

    ids = list(feats_t0.keys())
    data = {}
    for cid in ids:
        data[cid] = {
            "candidate_record_id": cid,
            "observed_state": feats_t0[cid].get("candidate_observed_state"),
            "T0": phase_payload(feats_t0, scores_t0, cid),
            "T1": phase_payload(feats_t1, scores_t1, cid),
        }

    os.makedirs("demo_data", exist_ok=True)
    out = "demo_data/case_data.json"
    with open(out, "w") as fh:
        json.dump(data, fh)
    print(f"\nExported {len(data)} candidates to {out} "
          f"({os.path.getsize(out)/1e6:.1f} MB)")

    # also dump a small sample-index of a few IDs per class (T1) so the demo
    # can offer ready-made examples to judges without hunting for IDs.
    samples = {"review_warranted": [], "review_not_warranted": [], "insufficient_evidence": []}
    for cid, rec in data.items():
        cls = rec["T1"]["predicted_class"]
        if cls in samples and len(samples[cls]) < 8:
            # prefer cases that actually have evidence, for a better demo
            if rec["T1"]["n_sources"] and rec["T1"]["n_sources"] >= 2:
                samples[cls].append(cid)
    with open("demo_data/sample_ids.json", "w") as fh:
        json.dump(samples, fh, indent=2)
    print("Wrote demo_data/sample_ids.json (example IDs per class)")


if __name__ == "__main__":
    main()

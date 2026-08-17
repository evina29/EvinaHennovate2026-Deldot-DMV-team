"""
End-to-end driver: T0 pipeline -> T1 evidence incorporation -> T1 scoring
-> case_predictions.csv matching the required submission schema.

review_priority is computed separately from the class probabilities: it
combines (a) how confident/warranted the flag is, (b) how much evidence
backs that confidence (more sources = more trustworthy signal, not just
higher probability), and (c) recency, so stale evidence doesn't rank as
urgently as fresh evidence. This is intentionally NOT just a copy of
p_review_warranted, since a low-evidence 60%-confident flag and a
5-source 60%-confident flag should not queue at the same priority.
"""
import sys
import csv
import numpy as np

from link_entities import link_all_sources, load_csv
from features import build_features, SOURCES
from pipeline_t0 import (
    run_t0, featurize_all, to_vector, build_dev_matrix, FEATURE_NAMES
)
from link_t1 import link_t1_updates, apply_t1_updates, get_t1_asof_date

BASE = './challenge_data'  # <-- point this at your unzipped challenge data folder
OUT_DIR = '.'  # <-- writes case_predictions.csv into this same folder


def compute_review_priority(f, proba_row, class_order):
    """
    review_priority in [0,1]. Combines:
      - p(review_warranted): the primary driver
      - evidence strength (total_weight, capped/normalized): more
        corroborating evidence -> more actionable / trustworthy flag
      - a small penalty when insufficient_evidence itself is the
        top class (low-evidence cases shouldn't crowd the priority queue
        even if their warranted-probability isn't zero)
    """
    idx_warranted = class_order.index('review_warranted')
    p_warranted = proba_row[idx_warranted]

    evidence_strength = min(f['total_weight'] / 3.0, 1.0)  # normalize; 3.0+ = well-evidenced
    n_src_factor = min(f['n_sources'] / 5.0, 1.0)

    priority = 0.6 * p_warranted + 0.25 * evidence_strength + 0.15 * n_src_factor

    idx_insuff = class_order.index('insufficient_evidence')
    p_insuff = proba_row[idx_insuff]
    if p_insuff > 0.5:
        priority *= 0.5  # de-prioritize cases we can't confidently characterize

    return round(float(min(max(priority, 0.0), 1.0)), 4)


def score_phase(feats_dict, model, class_order):
    """Returns dict: candidate_record_id -> (pred_class, proba_dict, priority)."""
    ids = list(feats_dict.keys())
    X = np.array([to_vector(feats_dict[cid]) for cid in ids])
    proba = model.predict_proba(X)
    pred = model.predict(X)

    out = {}
    for i, cid in enumerate(ids):
        proba_row = proba[i]
        proba_dict = {cls: float(proba_row[j]) for j, cls in enumerate(class_order)}
        # normalize defensively so it sums to exactly 1.0
        total = sum(proba_dict.values())
        if total > 0:
            proba_dict = {k: v / total for k, v in proba_dict.items()}
        priority = compute_review_priority(feats_dict[cid], proba_row, class_order)
        out[cid] = (str(pred[i]), proba_dict, priority)
    return out


def main():
    print("=" * 70)
    print("PHASE 1: T0 pipeline (link -> features -> model -> calibrate)")
    print("=" * 70)
    t0_result = run_t0()

    linked_t0 = t0_result['linked']
    feats_t0 = t0_result['feats']
    model = t0_result['model']
    class_order = t0_result['class_order']

    print("\n" + "=" * 70)
    print("PHASE 2: T1 evidence incorporation")
    print("=" * 70)
    # need a linker object to match T1 updates by name; rebuild from candidates
    from link_entities import EntityLinker
    candidates = load_csv(f'{BASE}/Data_T0/candidate_records.csv')
    linker = EntityLinker(candidates)

    matched_updates, unmatched_updates = link_t1_updates(BASE, linker)
    print(f"T1 evidence_update_stream: matched {len(matched_updates)}/"
          f"{len(matched_updates) + len(unmatched_updates)} rows to candidates "
          f"({len(matched_updates)/(len(matched_updates)+len(unmatched_updates)):.1%})")

    linked_t1, n_applied = apply_t1_updates(linked_t0, matched_updates)
    print(f"Applied {n_applied} T1 events into candidate evidence records")

    t1_asof = get_t1_asof_date(BASE)
    print(f"T1 as-of date: {t1_asof}")

    print("\nBuilding T1 features (T0 evidence + T1 updates, home_state = candidate's own tag state)...")
    feats_t1 = featurize_all(linked_t1, t1_asof)

    print("\n" + "=" * 70)
    print("PHASE 3: Scoring both phases with the calibrated model")
    print("=" * 70)
    scores_t0 = score_phase(feats_t0, model, class_order)
    scores_t1 = score_phase(feats_t1, model, class_order)

    # quick check: how much movement happened T0->T1 (sanity check on
    # "response to later evidence")
    n_changed = sum(1 for cid in scores_t0 if scores_t0[cid][0] != scores_t1[cid][0])
    print(f"\nCases where predicted class changed T0 -> T1: {n_changed}/{len(scores_t0)} "
          f"({n_changed/len(scores_t0):.1%})")

    print("\n" + "=" * 70)
    print("PHASE 4: Writing case_predictions.csv")
    print("=" * 70)
    template = load_csv(f'{BASE}/Submission_Template.csv')
    # template has one row per (candidate_record_id, phase) -- follow it exactly
    out_rows = []
    for row in template:
        cid = row['candidate_record_id']
        phase = row['phase']
        scores = scores_t0 if phase == 'T0' else scores_t1
        if cid not in scores:
            # should not happen (all 12,000 candidates are scored) but guard anyway
            pred_class, proba_dict, priority = 'insufficient_evidence', {
                'review_warranted': 0.0, 'review_not_warranted': 0.0,
                'insufficient_evidence': 1.0}, 0.0
        else:
            pred_class, proba_dict, priority = scores[cid]
        out_rows.append({
            'candidate_record_id': cid,
            'phase': phase,
            'predicted_class': pred_class,
            'p_review_warranted': round(proba_dict.get('review_warranted', 0.0), 6),
            'p_review_not_warranted': round(proba_dict.get('review_not_warranted', 0.0), 6),
            'p_insufficient_evidence': round(proba_dict.get('insufficient_evidence', 0.0), 6),
            'review_priority': priority,
        })

    out_path = f'{OUT_DIR}/case_predictions.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'candidate_record_id', 'phase', 'predicted_class',
            'p_review_warranted', 'p_review_not_warranted',
            'p_insufficient_evidence', 'review_priority'])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out_path}")

    # --- validation checks ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    n_rows = len(out_rows)
    n_expected = 24000
    print(f"Row count: {n_rows} (expected {n_expected}) -> {'OK' if n_rows == n_expected else 'MISMATCH'}")

    bad_sums = 0
    for r in out_rows:
        s = r['p_review_warranted'] + r['p_review_not_warranted'] + r['p_insufficient_evidence']
        if abs(s - 1.0) > 1e-4:
            bad_sums += 1
    print(f"Rows where probabilities don't sum to 1: {bad_sums} -> {'OK' if bad_sums == 0 else 'FIX NEEDED'}")

    bad_priority = sum(1 for r in out_rows if not (0.0 <= r['review_priority'] <= 1.0))
    print(f"Rows with review_priority out of [0,1]: {bad_priority} -> {'OK' if bad_priority == 0 else 'FIX NEEDED'}")

    ids_seen = set((r['candidate_record_id'], r['phase']) for r in out_rows)
    template_ids = set((r['candidate_record_id'], r['phase']) for r in template)
    print(f"All template (candidate_id, phase) pairs covered: {'OK' if ids_seen == template_ids else 'MISMATCH'}")

    return out_path, t0_result, scores_t0, scores_t1, feats_t0, feats_t1


if __name__ == '__main__':
    main()

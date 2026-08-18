"""
Compute honest, cross-validated evaluation metrics for the honesty panel and
save them to demo_data/metrics.json so the demo app can display them instantly
without needing ./challenge_data present at demo time.

We report OUT-OF-FOLD (5-fold stratified CV) predictions over ALL 300 dev
labels -- every label is predicted by a model that never saw it in training.
This is the honest headline number (matches the ~50% figure in
PROJECT_STATUS.md), not the optimistic single-split report.

Run:  python export_metrics.py     (needs ./challenge_data unzipped)
"""
import json
import os
from collections import Counter

import numpy as np
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from link_entities import link_all_sources, load_csv
from pipeline_t0 import (BASE, CLASSES, featurize_all, build_dev_matrix,
                         get_t0_asof_date)

OUT = "demo_data/metrics.json"


def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=5,
                                  min_samples_leaf=5, random_state=42)


def triage_lift(y, proba_warranted):
    """Out-of-fold triage-lift / gains curve for the review_warranted class.

    Ranks every labeled case by the model's OUT-OF-FOLD confidence that it is
    review_warranted (exactly how the worklist orders cases), then measures how
    many true review_warranted cases a reviewer catches working top-down vs.
    random order. This proves the *triage* is useful even though raw 3-class
    accuracy is only ~50%.
    """
    order = np.argsort(-proba_warranted)          # highest confidence first
    y_sorted = np.array([1 if y[i] == "review_warranted" else 0 for i in order])
    n = len(y_sorted)
    total_pos = int(y_sorted.sum())
    base = total_pos / n
    cum = np.cumsum(y_sorted)

    # gains curve: fraction of caseload reviewed -> fraction of all true positives found
    xs = [round((k + 1) / n, 4) for k in range(n)]
    gains = [round(int(cum[k]) / total_pos, 4) for k in range(n)]

    pak = {}
    for k in (10, 25, 50, 75, 100):
        if k <= n:
            hits = int(cum[k - 1])
            pak[str(k)] = {"precision": round(hits / k, 4),
                           "hits": hits,
                           "lift": round((hits / k) / base, 3)}
    # effort saved: to catch the first `hits@75` true positives in ranked order
    # you open 75 files; at the base rate you'd expect to open hits/base files.
    kref = 75 if n >= 75 else n
    hits_ref = int(cum[kref - 1])
    files_random = hits_ref / base if base else 0
    effort_saved = round(1 - kref / files_random, 3) if files_random else 0
    return {
        "base_rate": round(base, 4),
        "total_positives": total_pos,
        "precision_at_k": pak,
        "effort_saved_at_75": effort_saved,   # e.g. 0.47 -> 47% fewer files opened
        "gains_x": xs,
        "gains_prioritized": gains,
    }


def eval_phase(X, y, label_col):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = CalibratedClassifierCV(rf(), method="sigmoid", cv=5)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=CLASSES).tolist()

    # out-of-fold P(review_warranted) for the triage-lift curve.
    # cross_val_predict(predict_proba) returns columns in sorted-class order.
    proba = cross_val_predict(
        CalibratedClassifierCV(rf(), method="sigmoid", cv=5),
        X, y, cv=cv, method="predict_proba")
    wi = sorted(set(y)).index("review_warranted")
    lift = triage_lift(y, proba[:, wi])

    base = cross_val_predict(DummyClassifier(strategy="most_frequent"),
                             X, y, cv=cv)
    base_acc = accuracy_score(y, base)
    return {
        "label_col": label_col,
        "n": int(len(y)),
        "classes": CLASSES,
        "confusion_matrix": cm,          # rows = true, cols = predicted
        "accuracy": float(acc),
        "baseline_majority_accuracy": float(base_acc),
        "chance_accuracy": round(1.0 / len(CLASSES), 4),
        "class_distribution": dict(Counter(y)),
        "triage": lift,
    }


def main():
    print("Linking sources + featurizing (needs ./challenge_data)...")
    linked, _, _ = link_all_sources(BASE)
    candidates = load_csv(f"{BASE}/Data_T0/candidate_records.csv")
    as_of = get_t0_asof_date(candidates)
    feats = featurize_all(linked, as_of)

    dev_labels = load_csv(f"{BASE}/Development_Labels/Development_Labels.csv")

    out = {}
    for col in ("label_t0", "label_t1"):
        if col not in dev_labels[0]:
            continue
        X, y, ids = build_dev_matrix(feats, dev_labels, col)
        print(f"  {col}: {len(X)} labeled cases, dist={Counter(y)}")
        out[col] = eval_phase(X, y, col)
        print(f"    5-fold CV accuracy = {out[col]['accuracy']:.1%} "
              f"(baseline {out[col]['baseline_majority_accuracy']:.1%})")

    out["_meta"] = {
        "method": "5-fold stratified cross_val_predict (out-of-fold); "
                  "CalibratedClassifierCV(RandomForest, sigmoid).",
        "note": "Every label predicted by a model that never trained on it.",
    }

    os.makedirs("demo_data", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

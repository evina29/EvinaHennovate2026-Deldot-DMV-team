"""
Quick, honest search for a fast accuracy win over the current model.

We reuse the exact dev matrix (300 labeled cases, 30 features) and compare
candidate configs under the SAME 5-fold stratified CV the pipeline uses, so
numbers are comparable to the reported 50%. We report mean accuracy vs the
majority baseline (35%). No test-set peeking; CV only.
"""
from collections import Counter
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from link_entities import link_all_sources, load_csv
from pipeline_t0 import featurize_all, build_dev_matrix, get_t0_asof_date, BASE


def main():
    linked, _, _ = link_all_sources(BASE)
    candidates = load_csv(f"{BASE}/Data_T0/candidate_records.csv")
    as_of = get_t0_asof_date(candidates)
    feats = featurize_all(linked, as_of)
    dev = load_csv(f"{BASE}/Development_Labels/Development_Labels.csv")
    X, y, _ = build_dev_matrix(feats, dev, "label_t0")
    print(f"{len(X)} labeled cases; classes: {dict(Counter(y))}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "CURRENT: RandomForest d5": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42),
        "RF balanced class weight": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=42),
        "RF deeper (d8)": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=3, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=3, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=42),
        "GradientBoosting d2": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
        "LogReg balanced + scale": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
    }

    print(f"{'config':<30} {'CV acc':>8} {'std':>6}")
    print("-" * 46)
    results = {}
    for name, clf in configs.items():
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        results[name] = acc.mean()
        print(f"{name:<30} {acc.mean():>7.1%} {acc.std():>6.1%}")

    best = max((k for k in results if "Baseline" not in k), key=results.get)
    print("\nBest non-baseline config:", best, f"({results[best]:.1%})")
    cur = results["CURRENT: RandomForest d5"]
    print(f"Current model: {cur:.1%}  |  Best: {results[best]:.1%}  |  "
          f"delta: {results[best]-cur:+.1%}")


if __name__ == "__main__":
    main()

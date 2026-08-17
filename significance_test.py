"""
Is our ~45% held-out accuracy actually better than a dumb majority-class
baseline, or is it within noise on 75 test cases?

We reproduce the EXACT held-out split the pipeline reports on
(train_test_split(..., test_size=0.25, random_state=1, stratify=y)), then run:

  1. Model accuracy vs. majority-class baseline accuracy (point estimates
     + 95% Wilson confidence intervals).
  2. Binomial test of the model's correct-count against the baseline
     accuracy AND against pure 1/3 chance (one-sided).
  3. McNemar's exact test -- the statistically correct PAIRED comparison of
     two classifiers evaluated on the SAME test cases. This is the headline
     number.
  4. A 5-fold cross-validated view (less variance than one 75-case split)
     for context.
"""
from collections import Counter
import numpy as np
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from link_entities import link_all_sources, load_csv
from pipeline_t0 import featurize_all, build_dev_matrix, get_t0_asof_date, BASE


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


def mcnemar_exact(model_correct, base_correct):
    """
    Exact McNemar test. b = model right & baseline wrong,
    c = model wrong & baseline right. Two-sided exact binomial on b vs b+c.
    """
    b = int(np.sum(model_correct & ~base_correct))
    c = int(np.sum(~model_correct & base_correct))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main():
    print("Building features + dev matrix (this links entities once)...")
    linked, _, _ = link_all_sources(BASE)
    candidates = load_csv(f"{BASE}/Data_T0/candidate_records.csv")
    as_of = get_t0_asof_date(candidates)
    feats = featurize_all(linked, as_of)
    dev_labels = load_csv(f"{BASE}/Development_Labels/Development_Labels.csv")
    X, y, _ = build_dev_matrix(feats, dev_labels, "label_t0")
    print(f"Labeled cases: {len(X)}, class distribution: {dict(Counter(y))}\n")

    # --- reproduce the exact held-out split the pipeline reports ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y)
    n = len(y_te)

    rf = RandomForestClassifier(n_estimators=300, max_depth=5,
                                min_samples_leaf=5, random_state=42)
    model = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    base = DummyClassifier(strategy="most_frequent")
    base.fit(X_tr, y_tr)
    y_base = base.predict(X_te)

    model_correct = (y_pred == y_te)
    base_correct = (y_base == y_te)
    k_model = int(model_correct.sum())
    k_base = int(base_correct.sum())
    acc_model = k_model / n
    acc_base = k_base / n

    print("=" * 70)
    print(f"HELD-OUT TEST SET: n = {n} cases, {len(set(y))} balanced classes")
    print("=" * 70)
    print(f"Majority baseline predicts: '{base.predict(X_te[:1])[0]}' for every case")
    lo_m, hi_m = wilson_ci(k_model, n)
    lo_b, hi_b = wilson_ci(k_base, n)
    print(f"Model accuracy   : {acc_model:.1%}  ({k_model}/{n})   "
          f"95% CI [{lo_m:.1%}, {hi_m:.1%}]")
    print(f"Baseline accuracy: {acc_base:.1%}  ({k_base}/{n})   "
          f"95% CI [{lo_b:.1%}, {hi_b:.1%}]")
    print(f"Random-chance (1/{len(set(y))}) accuracy: {1/len(set(y)):.1%}")
    print(f"Absolute improvement over baseline: {acc_model - acc_base:+.1%}")
    print()

    # --- 2. binomial tests (unpaired, baseline treated as fixed target) ---
    p_vs_base = stats.binomtest(k_model, n, acc_base, alternative="greater").pvalue
    p_vs_chance = stats.binomtest(k_model, n, 1/len(set(y)), alternative="greater").pvalue
    print("-" * 70)
    print("BINOMIAL TESTS (one-sided, H1: model better)")
    print("-" * 70)
    print(f"  Model vs. majority baseline ({acc_base:.1%}): p = {p_vs_base:.4f}")
    print(f"  Model vs. pure chance ({1/len(set(y)):.1%})     : p = {p_vs_chance:.4f}")
    print()

    # --- 3. McNemar's paired test (the correct headline test) ---
    b, c, p_mcnemar = mcnemar_exact(model_correct, base_correct)
    print("-" * 70)
    print("McNEMAR'S EXACT TEST (paired, model vs baseline on SAME cases)")
    print("-" * 70)
    print(f"  Model right, baseline wrong (b): {b}")
    print(f"  Model wrong, baseline right (c): {c}")
    print(f"  p-value (two-sided): {p_mcnemar:.4f}")
    print()

    # --- 4. cross-validated view (lower variance than one split) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    cv_base = cross_val_score(DummyClassifier(strategy="most_frequent"),
                              X, y, cv=cv, scoring="accuracy")
    print("-" * 70)
    print("5-FOLD CROSS-VALIDATION (context; uses all 300 labels)")
    print("-" * 70)
    print(f"  Model : {cv_model.mean():.1%} (+/- {cv_model.std():.1%})")
    print(f"  Baseline: {cv_base.mean():.1%} (+/- {cv_base.std():.1%})")
    # paired t-test across folds
    t, p_t = stats.ttest_rel(cv_model, cv_base)
    print(f"  Paired t-test across folds: p = {p_t:.4f}")
    print()

    # --- verdict ---
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    alpha = 0.05
    sig = p_mcnemar < alpha
    print(f"  Headline (McNemar) p = {p_mcnemar:.4f} -> "
          f"{'STATISTICALLY SIGNIFICANT' if sig else 'NOT significant'} at a=0.05")
    if not sig:
        print("  The single-split improvement could plausibly be noise.")
    print(f"  CV supporting view: model beats baseline by "
          f"{(cv_model.mean()-cv_base.mean()):+.1%}, paired p = {p_t:.4f}")


if __name__ == "__main__":
    main()

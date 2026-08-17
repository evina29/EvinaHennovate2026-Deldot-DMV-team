"""
T0 pipeline v2: link entities -> build rich per-source features ->
cross-validate multiple models -> pick the best -> calibrate probabilities
-> score all 12,000 candidates.
"""
import sys
from datetime import date
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, brier_score_loss, accuracy_score

from link_entities import link_all_sources, load_csv
from features import build_features, SOURCES

BASE = './challenge_data'  # <-- point this at your unzipped challenge data folder
CLASSES = ['review_warranted', 'review_not_warranted', 'insufficient_evidence']

FEATURE_NAMES = (
    ['n_sources', 'out_of_state_weight', 'in_state_weight', 'de_weight',
     'net_score', 'total_weight', 'n_distinct_states_seen',
     'n_sources_agree_home', 'n_sources_point_de', 'home_state_is_de']
    + [f'{src}_present' for src in SOURCES]
    + [f'{src}_matches_home' for src in SOURCES]
    + [f'{src}_is_de' for src in SOURCES]
    + [f'{src}_weight' for src in SOURCES]
)


def get_t0_asof_date(candidates):
    dates = []
    for c in candidates:
        try:
            dates.append(date.fromisoformat(c['candidate_observed_date'][:10]))
        except Exception:
            pass
    return max(dates) if dates else date.today()


def featurize_all(linked, as_of_date):
    """home_state = candidate's OWN tagged/observed state (see features.py docstring)."""
    feats = {}
    for cid, record in linked.items():
        cand_home_state = record['candidate'].get('observed_state') or 'DE'
        feats[cid] = build_features(record, home_state=cand_home_state, as_of_date=as_of_date)
    return feats


def to_vector(f):
    ps = f['per_source']
    vec = [
        f['n_sources'], f['out_of_state_weight'], f['in_state_weight'], f['de_weight'],
        f['net_score'], f['total_weight'], f['n_distinct_states_seen'],
        f['n_sources_agree_home'], f['n_sources_point_de'], f['home_state_is_de'],
    ]
    vec += [ps[src]['present'] for src in SOURCES]
    vec += [ps[src]['matches_home'] for src in SOURCES]
    vec += [ps[src]['is_de'] for src in SOURCES]
    vec += [ps[src]['weight'] for src in SOURCES]
    return vec


def build_dev_matrix(feats, dev_labels, label_col='label_t0'):
    X, y, ids = [], [], []
    for row in dev_labels:
        cid = row['candidate_record_id']
        if cid not in feats:
            continue
        X.append(to_vector(feats[cid]))
        y.append(row[label_col])
        ids.append(cid)
    return np.array(X), np.array(y), ids


def select_best_model(X, y, verbose=True):
    """5-fold stratified CV over a few candidate models; returns the best."""
    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=2000, C=1.0),
        'LogisticRegression_L2_strong': LogisticRegression(max_iter=2000, C=0.3),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, clf in candidates.items():
        acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        scores[name] = acc.mean()
        if verbose:
            print(f"  {name}: {acc.mean():.3f} (+/- {acc.std():.3f})")
    best_name = max(scores, key=scores.get)
    if verbose:
        print(f"  -> selected: {best_name}")
    return candidates[best_name], best_name, scores


def run_t0():
    print("Step 1: Linking entities across source tables...")
    linked, link_stats, _ = link_all_sources(BASE)
    for name, (m, tot) in link_stats.items():
        print(f"  {name}: {m}/{tot} ({m/tot:.1%})")

    candidates = load_csv(f'{BASE}/Data_T0/candidate_records.csv')
    as_of = get_t0_asof_date(candidates)
    print(f"\nT0 as-of date: {as_of}")

    print("\nStep 2: Building rich per-source features for all 12,000 candidates...")
    feats = featurize_all(linked, as_of)

    print("\nStep 3: Loading dev labels...")
    dev_labels = load_csv(f'{BASE}/Development_Labels/Development_Labels.csv')
    X, y, ids = build_dev_matrix(feats, dev_labels, 'label_t0')
    print(f"  {len(X)} labeled cases, distribution: {Counter(y)}")

    print("\nStep 4: 5-fold CV model selection...")
    best_clf, best_name, cv_scores = select_best_model(X, y)

    print(f"\nStep 5: Calibrating probabilities ({best_name} + sigmoid calibration)...")
    calibrated = CalibratedClassifierCV(best_clf, method='sigmoid', cv=5)
    calibrated.fit(X, y)

    # Held-out sanity report using a simple split for a human-readable report
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    check_model = CalibratedClassifierCV(best_clf, method='sigmoid', cv=5)
    check_model.fit(X_tr, y_tr)
    y_pred = check_model.predict(X_te)
    print("\n=== Held-out validation report (25% split) ===")
    print(classification_report(y_te, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_te, y_pred):.3f}")

    proba_te = check_model.predict_proba(X_te)
    class_order = list(check_model.classes_)
    print("Calibration (Brier score per class):")
    for i, cls in enumerate(class_order):
        y_bin = np.array([1 if v == cls else 0 for v in y_te])
        bs = brier_score_loss(y_bin, proba_te[:, i])
        print(f"  {cls}: {bs:.4f}")

    print("\nStep 6: Scoring all 12,000 candidates (T0)...")
    all_ids = list(feats.keys())
    X_all = np.array([to_vector(feats[cid]) for cid in all_ids])
    proba_all = calibrated.predict_proba(X_all)
    pred_all = calibrated.predict(X_all)
    class_order_final = list(calibrated.classes_)
    print(f"  Predicted label distribution: {Counter(pred_all)}")

    return {
        'linked': linked, 'feats': feats, 'model': calibrated,
        'class_order': class_order_final, 'all_ids': all_ids,
        'proba_all': proba_all, 'pred_all': pred_all, 'as_of_date': as_of,
        'cv_scores': cv_scores, 'best_model_name': best_name,
    }


if __name__ == '__main__':
    run_t0()

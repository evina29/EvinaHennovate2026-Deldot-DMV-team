"""
Feature engineering for the Out-of-State Tag Holder challenge.

Core idea: the candidate's vehicle is tagged in a state (usually DE).
We compare that against the MOST RECENT state seen in each independent
source (address history, license/credential, work location, vehicle
title events, external signals). Consistent out-of-state signals across
multiple independent sources = stronger evidence the tag may not belong
in Delaware. Sparse or single-source evidence = insufficient_evidence,
not a guess.

All features are auditable: every score traces back to a specific
source, state, date, and (where applicable) status/quality flag, so a
human reviewer can see exactly why a case scored the way it did.
"""
from datetime import datetime, date

SOURCES = ['address', 'license', 'work', 'vehicle', 'external']


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


def most_recent_state_per_source(record):
    """
    For each source, find the most recent dated event and its state.
    Returns dict: source -> {'state':.., 'date':.., 'weight': float, 'detail':..}
    weight reflects confidence: match confidence, credential status,
    evidence_quality, and source type all factor in.
    """
    out = {}

    # address_history: state, effective_start_date (fallback effective_end_date)
    best = None
    for r in record['address']:
        d = parse_date(r.get('effective_start_date')) or parse_date(r.get('effective_end_date'))
        if d and (best is None or d > best[0]):
            best = (d, r)
    if best:
        d, r = best
        w = 1.0 if r.get('_match_confidence') == 'high' else 0.75
        out['address'] = {'state': r['state'], 'date': d, 'weight': w,
                           'detail': f"address on file changed to {r['state']} ({d})"}

    # license_id_events: credential_state, event_date; active > superseded/expired > unknown
    best = None
    for r in record['license']:
        d = parse_date(r.get('event_date'))
        if d and (best is None or d > best[0]):
            best = (d, r)
    if best:
        d, r = best
        status = r.get('credential_status', 'unknown')
        status_w = {'active': 1.0, 'superseded': 0.8, 'expired': 0.6, 'unknown': 0.4}.get(status, 0.4)
        conf_w = 1.0 if r.get('_match_confidence') == 'high' else 0.75
        out['license'] = {'state': r['credential_state'], 'date': d, 'weight': status_w * conf_w,
                           'detail': f"license/credential state {r['credential_state']} ({status}, {d})"}

    # work_location_signals: work_state, observed_date
    best = None
    for r in record['work']:
        d = parse_date(r.get('observed_date'))
        if d and (best is None or d > best[0]):
            best = (d, r)
    if best:
        d, r = best
        w = 1.0 if r.get('_match_confidence') == 'high' else 0.75
        out['work'] = {'state': r['work_state'], 'date': d, 'weight': w,
                        'detail': f"employer location {r['work_state']} ({d})"}

    # vehicle_title_events: event_state, event_date
    best = None
    for r in record['vehicle']:
        d = parse_date(r.get('event_date'))
        if d and (best is None or d > best[0]):
            best = (d, r)
    if best:
        d, r = best
        w = 1.0 if r.get('_match_confidence') == 'high' else 0.75
        out['vehicle'] = {'state': r['event_state'], 'date': d, 'weight': w,
                           'detail': f"vehicle title event in {r['event_state']} ({r['event_type']}, {d})"}

    # external_context_signals: signal_state, effective_date; weight by evidence_quality
    best = None
    for r in record['external']:
        d = parse_date(r.get('effective_date'))
        if d and (best is None or d > best[0]):
            best = (d, r)
    if best:
        d, r = best
        q_w = 1.0 if r.get('evidence_quality') == 'standard' else 0.5
        conf_w = 1.0 if r.get('_match_confidence') == 'high' else 0.75
        out['external'] = {'state': r['signal_state'], 'date': d, 'weight': q_w * conf_w,
                            'detail': f"external signal ({r['signal_type']}) state {r['signal_state']} ({d})"}

    return out


def recency_decay(event_date, as_of_date, half_life_days=365):
    """Weight decays as evidence gets older. 1.0 = today, 0.5 at half_life_days old."""
    if event_date is None:
        return 0.0
    age = (as_of_date - event_date).days
    if age < 0:
        age = 0
    return 0.5 ** (age / half_life_days)


def build_features(record, home_state='DE', as_of_date=None):
    """
    Returns a feature dict for one candidate, including a human-readable
    evidence trail for auditability, AND a granular per-source feature
    breakdown (rather than one collapsed 'net score') so a downstream
    classifier can learn which source combinations actually matter
    instead of relying on a single hand-tuned formula.

    home_state = the state currently on the candidate's tag/observed
    record. The review question is whether OTHER evidence (address,
    license, work, vehicle title, external signals) is consistent with
    that tag state or points elsewhere -- notably toward Delaware, since
    the operational concern is out-of-state tag holders who may actually
    reside in DE and owe DE registration.
    """
    if as_of_date is None:
        as_of_date = date.today()

    cand = record['candidate']
    per_source = most_recent_state_per_source(record)

    n_sources = len(per_source)
    out_of_state_weight = 0.0
    in_state_weight = 0.0
    de_weight = 0.0  # specifically: weighted evidence pointing to DE
    evidence_trail = []
    states_seen = set()
    per_source_detail = {}

    for src in SOURCES:
        info = per_source.get(src)
        if info is None:
            per_source_detail[src] = {
                'present': 0, 'matches_home': 0, 'is_de': 0, 'weight': 0.0
            }
            continue
        recency_w = recency_decay(info['date'], as_of_date)
        total_w = info['weight'] * recency_w
        states_seen.add(info['state'])
        matches_home = 1 if info['state'] == home_state else 0
        is_de = 1 if info['state'] == 'DE' else 0

        per_source_detail[src] = {
            'present': 1, 'matches_home': matches_home, 'is_de': is_de,
            'weight': round(total_w, 4), 'state': info['state'], 'date': str(info['date']),
        }

        if info['state'] and info['state'] != home_state:
            out_of_state_weight += total_w
            evidence_trail.append(f"[differs from tag state {home_state}] {info['detail']}")
        elif info['state'] == home_state:
            in_state_weight += total_w
            evidence_trail.append(f"[matches tag state {home_state}] {info['detail']}")
        if is_de:
            de_weight += total_w

    net_score = out_of_state_weight - in_state_weight
    total_weight = out_of_state_weight + in_state_weight
    n_sources_agree_home = sum(1 for s in per_source_detail.values() if s['matches_home'])
    n_sources_point_de = sum(1 for s in per_source_detail.values() if s.get('is_de'))

    return {
        'candidate_record_id': cand['candidate_record_id'],
        'home_state': home_state,
        'home_state_is_de': 1 if home_state == 'DE' else 0,
        'n_sources': n_sources,
        'out_of_state_weight': round(out_of_state_weight, 4),
        'in_state_weight': round(in_state_weight, 4),
        'de_weight': round(de_weight, 4),
        'net_score': round(net_score, 4),
        'total_weight': round(total_weight, 4),
        'n_distinct_states_seen': len(states_seen),
        'n_sources_agree_home': n_sources_agree_home,
        'n_sources_point_de': n_sources_point_de,
        'candidate_observed_state': cand.get('observed_state'),
        'evidence_trail': evidence_trail,
        'per_source': per_source_detail,
    }

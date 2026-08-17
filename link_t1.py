"""
T1 evidence incorporation.

evidence_update_stream.csv rows carry: source_domain (address/license/
title/external), record_action (new_record/status_update/record_
correction), state, vehicle_ref, effective_date, observed_date.

They don't carry DOB, so we link them the same conservative way as the
other DOB-less T0 tables: block on normalized last_name, disambiguate
with fuzzy first-name matching, and leave unmatched anything ambiguous.

All three record_actions (new_record, status_update, record_correction)
are treated as new evidence events appended to the relevant source's
event list. Because downstream feature-building always takes the MOST
RECENT dated event per source, a correction or status update naturally
takes precedence over older T0 evidence as long as its effective_date
is more recent -- which is the realistic case for "later evidence
supersedes earlier evidence." This is a simplifying assumption, noted
here for auditability: we do not attempt field-level diffing/patching
of individual T0 records.
"""
import sys
from link_entities import load_csv, norm, names_match

DOMAIN_TO_SOURCE = {
    'address': 'address',
    'license': 'license',
    'title': 'vehicle',
    'external': 'external',
    # no 'work' domain present in T1 evidence_update_stream
}


def link_t1_updates(base_dir, linker):
    """
    Returns (n_matched, n_total, list_of_unmatched_rows).
    Mutates nothing; call apply_t1_updates() to actually merge.
    """
    updates = load_csv(f'{base_dir}/Data_T1/evidence_update_stream.csv')
    matched, unmatched = [], []
    for r in updates:
        cid, conf = linker.match_no_dob(r['first_name'], r['last_name'])
        if cid:
            r['_match_confidence'] = conf
            r['_candidate_record_id'] = cid
            matched.append(r)
        else:
            unmatched.append(r)
    return matched, unmatched


def apply_t1_updates(linked_t0, matched_updates):
    """
    Returns a NEW linked dict (deep-ish copy at the list level so T0 stays
    intact) with T1 update rows appended into the right source list, in a
    schema-compatible shape so features.py's most_recent_state_per_source
    can read them without changes.
    """
    import copy
    linked_t1 = {cid: {
        'candidate': rec['candidate'],
        'address': list(rec['address']),
        'license': list(rec['license']),
        'vehicle': list(rec['vehicle']),
        'work': list(rec['work']),
        'external': list(rec['external']),
    } for cid, rec in linked_t0.items()}

    n_applied = 0
    for r in matched_updates:
        cid = r['_candidate_record_id']
        if cid not in linked_t1:
            continue
        domain = r['source_domain']
        source_key = DOMAIN_TO_SOURCE.get(domain)
        if source_key is None:
            continue

        if source_key == 'address':
            event = {
                'state': r['state'],
                'effective_start_date': r['effective_date'],
                'effective_end_date': '',
                'source_type': f"t1_{r['record_action']}",
                '_match_confidence': r['_match_confidence'],
            }
        elif source_key == 'license':
            event = {
                'credential_state': r['state'],
                'event_date': r['effective_date'],
                'credential_status': 'active',  # T1 refresh assumed current
                'event_type': f"t1_{r['record_action']}",
                '_match_confidence': r['_match_confidence'],
            }
        elif source_key == 'vehicle':
            event = {
                'event_state': r['state'],
                'event_date': r['effective_date'],
                'event_type': f"t1_{r['record_action']}",
                '_match_confidence': r['_match_confidence'],
            }
        elif source_key == 'external':
            event = {
                'signal_state': r['state'],
                'effective_date': r['effective_date'],
                'evidence_quality': 'standard',
                'signal_type': f"t1_{r['record_action']}",
                '_match_confidence': r['_match_confidence'],
            }
        else:
            continue

        linked_t1[cid][source_key].append(event)
        n_applied += 1

    return linked_t1, n_applied


def get_t1_asof_date(base_dir):
    from datetime import date
    updates = load_csv(f'{base_dir}/Data_T1/evidence_update_stream.csv')
    dates = []
    for r in updates:
        try:
            dates.append(date.fromisoformat(r['observed_date'][:10]))
        except Exception:
            pass
    return max(dates) if dates else date.today()

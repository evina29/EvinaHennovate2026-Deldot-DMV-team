"""
Entity linkage for the Out-of-State Tag Holder challenge.

Problem: only candidate_records.csv has a clean candidate_record_id.
All other T0 source tables (address_history, license_id_events,
vehicle_title_events, work_location_signals, external_context_signals)
only carry name (+ sometimes date_of_birth), and those names contain
realistic data-quality noise: truncation ("N" for "Nwzgpc"), case
differences, and small typos.

Strategy:
  - Primary key where DOB is available: (last_name_normalized, date_of_birth)
    -> nearly unique across the 12,000 candidates (11,998/12,000).
  - For DOB-less tables: block on normalized last_name, then disambiguate
    with fuzzy first-name matching (prefix containment or bounded edit
    distance) within the block.
  - Conservative by design: if a row can't be resolved to exactly one
    candidate with confidence, it is left UNMATCHED rather than guessed.
    A wrong link would poison every downstream feature, so precision is
    prioritized over recall here.
"""
import csv
import re
from collections import defaultdict


def norm(s):
    """Uppercase, strip everything but letters (drops SYNGIV-/SYNFAM- prefixes etc)."""
    return re.sub(r'[^A-Za-z]', '', s or '').upper()


def edit_distance_le(a, b, max_d=2):
    """True if levenshtein distance between a and b is <= max_d."""
    if abs(len(a) - len(b)) > max_d:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] <= max_d


def names_match(a, b):
    """Fuzzy first-name match: exact, truncation (prefix), or small edit distance."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    return edit_distance_le(a, b, 2)


class EntityLinker:
    def __init__(self, candidates):
        """candidates: list of dicts from candidate_records.csv"""
        self.candidates = candidates

        self.by_lndob = defaultdict(list)
        for c in candidates:
            key = (norm(c['last_name']), c['date_of_birth'])
            self.by_lndob[key].append(c)

        self.by_lastname = defaultdict(list)
        for c in candidates:
            self.by_lastname[norm(c['last_name'])].append(c)

    def match_with_dob(self, first_name, last_name, dob):
        key = (norm(last_name), dob)
        cands = self.by_lndob.get(key, [])
        if len(cands) == 1:
            return cands[0]['candidate_record_id'], 'high'
        if len(cands) > 1:
            good = [c for c in cands if names_match(c['first_name'], first_name)]
            if len(good) == 1:
                return good[0]['candidate_record_id'], 'medium'
        return None, None

    def match_no_dob(self, first_name, last_name):
        block = self.by_lastname.get(norm(last_name), [])
        good = [c for c in block if names_match(c['first_name'], first_name)]
        if len(good) == 1:
            return good[0]['candidate_record_id'], 'medium'
        return None, None


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def link_all_sources(base_dir):
    """
    Returns dict: candidate_record_id -> {
        'address': [...], 'license': [...], 'vehicle': [...],
        'work': [...], 'external': [...]
    }
    Each event dict is the original row plus '_match_confidence'.
    """
    candidates = load_csv(f'{base_dir}/Data_T0/candidate_records.csv')
    linker = EntityLinker(candidates)

    linked = {c['candidate_record_id']: {
        'candidate': c, 'address': [], 'license': [], 'vehicle': [],
        'work': [], 'external': []
    } for c in candidates}

    stats = {}

    # license_id_events -- has DOB
    lic = load_csv(f'{base_dir}/Data_T0/license_id_events.csv')
    m = 0
    for r in lic:
        cid, conf = linker.match_with_dob(r['first_name'], r['last_name'], r['date_of_birth'])
        if cid:
            r['_match_confidence'] = conf
            linked[cid]['license'].append(r)
            m += 1
    stats['license_id_events'] = (m, len(lic))

    # address_history -- name only
    addr = load_csv(f'{base_dir}/Data_T0/address_history.csv')
    m = 0
    for r in addr:
        cid, conf = linker.match_no_dob(r['first_name'], r['last_name'])
        if cid:
            r['_match_confidence'] = conf
            linked[cid]['address'].append(r)
            m += 1
    stats['address_history'] = (m, len(addr))

    # external_context_signals -- name only
    ext = load_csv(f'{base_dir}/Data_T0/external_context_signals.csv')
    m = 0
    for r in ext:
        cid, conf = linker.match_no_dob(r['first_name'], r['last_name'])
        if cid:
            r['_match_confidence'] = conf
            linked[cid]['external'].append(r)
            m += 1
    stats['external_context_signals'] = (m, len(ext))

    # work_location_signals -- name only
    work = load_csv(f'{base_dir}/Data_T0/work_location_signals.csv')
    m = 0
    for r in work:
        cid, conf = linker.match_no_dob(r['first_name'], r['last_name'])
        if cid:
            r['_match_confidence'] = conf
            linked[cid]['work'].append(r)
            m += 1
    stats['work_location_signals'] = (m, len(work))

    # vehicle_title_events -- owner_first_name/owner_last_name, name only
    veh = load_csv(f'{base_dir}/Data_T0/vehicle_title_events.csv')
    m = 0
    for r in veh:
        cid, conf = linker.match_no_dob(r['owner_first_name'], r['owner_last_name'])
        if cid:
            r['_match_confidence'] = conf
            linked[cid]['vehicle'].append(r)
            m += 1
    stats['vehicle_title_events'] = (m, len(veh))

    return linked, stats, linker


if __name__ == '__main__':
    BASE = './challenge_data'  # <-- point this at your unzipped challenge data folder
    linked, stats, linker = link_all_sources(BASE)
    print("=== Entity Linkage Match Rates ===")
    for name, (m, tot) in stats.items():
        print(f"  {name}: {m}/{tot} ({m/tot:.1%})")
    n_with_any_evidence = sum(1 for v in linked.values()
                               if v['address'] or v['license'] or v['vehicle']
                               or v['work'] or v['external'])
    print(f"\nCandidates with >=1 piece of linked evidence: {n_with_any_evidence}/{len(linked)}")

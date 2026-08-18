"""
Build a single, self-contained dmv_demo.html backup of the live demo.

No Python, no server, no install: double-click the file and it opens in any
browser. It embeds a trimmed slice of the real pipeline output (the curated
cases + the top worklist + the honesty metrics) and PRE-GENERATED plain-English
explanations, so it works with zero network. This is the can't-fail fallback to
the Streamlit app; the Streamlit app remains the main demo for live Gemini.

Run:  python export_html.py     (after export_case_data.py + export_metrics.py)
"""
import json

from explain import generate_case_explanation
from demo_cases import CURATED, CURATED_IDS, FLIP_CID

data = json.load(open("demo_data/case_data.json"))
metrics = json.load(open("demo_data/metrics.json"))


def worklist(n=12):
    rows = []
    for cid, rec in data.items():
        p = rec["T1"]
        if p["predicted_class"] != "review_warranted":
            continue
        ps = p.get("per_source", {})
        de = sum(1 for s in ps.values() if s.get("present") and s.get("is_de"))
        differ = sum(1 for s in ps.values()
                     if s.get("present") and not s.get("matches_home"))
        rows.append({"cid": cid, "priority": p["review_priority"],
                     "tag": p.get("home_state") or rec.get("observed_state") or "?",
                     "de_signals": de, "differ": differ,
                     "n_sources": p.get("n_sources", 0)})
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows[:n]


def evidence_dict(cid, phase, tag):
    return {"candidate_record_id": cid, "home_state": tag,
            "predicted_class": phase["predicted_class"],
            "probabilities": phase["probabilities"],
            "review_priority": phase["review_priority"],
            "evidence_trail": phase.get("evidence_trail", []),
            "per_source": phase.get("per_source", {}),
            "n_sources": phase.get("n_sources", 0)}


wl = worklist(12)
include = list(dict.fromkeys(CURATED_IDS + [r["cid"] for r in wl]))

cases = {}
for cid in include:
    rec = data[cid]
    tag = rec.get("observed_state") or "DE"
    entry = {"observed_state": rec.get("observed_state"),
             "T0": rec["T0"], "T1": rec["T1"], "expl": {}}
    for ph in ("T0", "T1"):
        text, src = generate_case_explanation(
            evidence_dict(cid, rec[ph], rec[ph].get("home_state") or tag))
        entry["expl"][ph] = {"text": text, "source": src}
    cases[cid] = entry
    print(f"  baked {cid}")

bundle = {"curated": CURATED, "flip": FLIP_CID, "worklist": wl,
          "cases": cases, "metrics": metrics}

TEMPLATE = open("html_template.html", encoding="utf-8").read()
html = TEMPLATE.replace("/*__DATA__*/", json.dumps(bundle))
with open("dmv_demo.html", "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"Wrote dmv_demo.html ({len(html)//1024} KB), {len(cases)} cases embedded.")

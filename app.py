"""
DMV Out-of-State Tag Review -- judge-facing live demo.

A decision-support tool that triages vehicle registration cases: it flags
vehicles that may need staff review because the person is tagged in another
state but shows evidence of actually living in Delaware.

Four sections (tabs):
  1. Morning Worklist  -- the highest-priority review_warranted cases, ranked
     like a reviewer's queue for the day.
  2. Case File         -- one case as a visual file: class, calibrated
     probabilities, review_priority, an evidence timeline, and a plain-English
     Gemini explanation for a non-technical reviewer.
  3. New Evidence (T0 -> T1) -- the same case before/after new evidence arrives,
     side by side, to show the system responds to new information.
  4. Model Honesty     -- the real cross-validated confusion matrix and
     accuracy vs. baseline, so limitations are owned up front.

Run:  streamlit run app.py
Data comes from demo_data/case_data.json + demo_data/metrics.json (produced by
export_case_data.py and export_metrics.py), so the app loads instantly and
never re-runs the model live.
"""
import json

import streamlit as st

from explain import generate_case_explanation, _get_client
from demo_cases import CURATED, CURATED_IDS, FLIP_CID

st.set_page_config(page_title="OSTR · Out-of-State Tag Review — Delaware DMV",
                   page_icon="🏛", layout="wide")

CLASSES = ["review_warranted", "review_not_warranted", "insufficient_evidence"]
CLASS_LABEL = {
    "review_warranted": "REVIEW WARRANTED",
    "review_not_warranted": "REVIEW NOT WARRANTED",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE",
}
# Muted semantic system — color is reserved strictly for the three review
# classes (matches the deployed console styling in html_template.html).
SEM = {
    "review_warranted": {"fg": "#b42318", "bg": "#fef3f2", "bd": "#f7ccc6", "full": "Review warranted"},
    "review_not_warranted": {"fg": "#12683f", "bg": "#eefaf2", "bd": "#bfe6cf", "full": "Review not warranted"},
    "insufficient_evidence": {"fg": "#b25409", "bg": "#fdf6ec", "bd": "#f2ddb8", "full": "Insufficient evidence"},
}
CLASS_COLOR = {k: v["fg"] for k, v in SEM.items()}
SOURCE_LABELS = {
    "address": "Address history",
    "license": "License / credential",
    "work": "Work location",
    "vehicle": "Vehicle title event",
    "external": "External signal",
}
SOURCE_ORDER = ["address", "license", "work", "vehicle", "external"]

# --- enterprise restraint: strip Streamlit chrome, set typographic system --- #
st.markdown("""
<style>
  :root{ --g1:#f6f7f8; --g3:#e2e5e9; --g6:#6b7480; --g8:#2b3138; --g9:#171b20; --accent:#1f4d6b; }
  #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{visibility:hidden;height:0}
  header[data-testid="stHeader"]{background:transparent}
  html, body, [class*="css"], .stMarkdown, p, span, div{
    font-family:"Inter","Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif; }
  .block-container{padding-top:2.2rem;padding-bottom:2rem;max-width:1180px}
  code, .mono, [data-testid="stMetricValue"]{
    font-family:"SFMono-Regular","Cascadia Mono",ui-monospace,"Roboto Mono",Consolas,monospace !important;
    font-variant-numeric:tabular-nums; }
  h1,h2,h3,h4{letter-spacing:-.1px;color:var(--g9)}
  /* metrics as compact enterprise stat blocks */
  [data-testid="stMetric"]{background:#fff;border:1px solid var(--g3);border-radius:6px;padding:10px 14px}
  [data-testid="stMetricLabel"] p{font-size:10px !important;font-weight:600;letter-spacing:.5px;
    text-transform:uppercase;color:var(--g6) !important}
  [data-testid="stMetricValue"]{font-size:22px !important;color:var(--g9)}
  /* restrained tabs */
  .stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--g3)}
  .stTabs [data-baseweb="tab"]{font-size:13px;padding:8px 14px;color:var(--g6)}
  .stTabs [aria-selected="true"]{color:var(--g9);box-shadow:inset 0 -2px 0 var(--accent)}
  /* buttons: rectangular, restrained */
  .stButton>button{border-radius:5px;border:1px solid #d3d8dd;font-size:12.5px;font-weight:500;color:var(--g8)}
  .stButton>button:hover{border-color:var(--accent);color:var(--accent)}
  [data-testid="stSidebar"]{background:#171b20;border-right:1px solid #000}
  [data-testid="stSidebar"] *{color:#c2c9d1}
  [data-testid="stSidebar"] .stButton>button{background:#212933;border-color:#2b333d;color:#d6dbe1;text-align:left}
  /* priority bars: neutral gray, not accent (color is reserved for classes) */
  [data-testid="stProgress"] div[role="progressbar"]>div>div{background-color:#48505a !important}
  [data-testid="stProgress"] p{font-size:11px;color:#6b7480;font-family:"SFMono-Regular",ui-monospace,monospace}
  /* status banners: flat, not glossy cards */
  [data-testid="stAlert"]{border-radius:5px;font-size:12.5px}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data():
    with open("demo_data/case_data.json") as fh:
        data = json.load(fh)
    try:
        with open("demo_data/metrics.json") as fh:
            metrics = json.load(fh)
    except FileNotFoundError:
        metrics = None
    return data, metrics


data, metrics = load_data()


@st.cache_data
def worklist_top(n=12):
    """Highest-priority review_warranted cases at T1 -- the reviewer's queue."""
    rows = []
    for cid, rec in data.items():
        p = rec["T1"]
        if p["predicted_class"] != "review_warranted":
            continue
        ps = p.get("per_source", {})
        de = sum(1 for s in ps.values() if s.get("present") and s.get("is_de"))
        differ = sum(1 for s in ps.values()
                     if s.get("present") and not s.get("matches_home"))
        rows.append({
            "cid": cid,
            "priority": p["review_priority"],
            "tag": p.get("home_state") or rec.get("observed_state") or "?",
            "prob": p["probabilities"]["review_warranted"],
            "de_signals": de,
            "differ": differ,
            "n_sources": p.get("n_sources", 0),
        })
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows[:n]


# --------------------------------------------------------------------------- #
# Small render helpers
# --------------------------------------------------------------------------- #
def class_badge(pred, size=13):
    """Restrained status pill: tinted background, colored dot + label."""
    s = SEM.get(pred, {"fg": "#555", "bg": "#eee", "bd": "#ccc",
                       "full": CLASS_LABEL.get(pred, pred)})
    return (f"<span style='display:inline-flex;align-items:center;gap:6px;"
            f"font-size:{size}px;font-weight:600;padding:3px 10px;border-radius:5px;"
            f"color:{s['fg']};background:{s['bg']};border:1px solid {s['bd']};'>"
            f"<span style='width:7px;height:7px;border-radius:2px;"
            f"background:{s['fg']};'></span>{s['full']}</span>")


def prob_bars(probs, highlight=None):
    """Three labeled horizontal bars, colored per class."""
    for cls in CLASSES:
        pct = float(probs.get(cls, 0.0))
        color = CLASS_COLOR[cls]
        emph = "font-weight:700;" if cls == highlight else ""
        label = CLASS_LABEL[cls].title()
        st.markdown(
            f"<div style='margin:3px 0;{emph}'>"
            f"<span style='display:inline-block;width:210px;'>{label}</span>"
            f"<span style='display:inline-block;width:52px;text-align:right;"
            f"padding-right:10px;'>{pct:.0%}</span>"
            f"<span style='display:inline-block;width:calc(100% - 280px);"
            f"vertical-align:middle;background:#eee;border-radius:6px;'>"
            f"<span style='display:inline-block;height:14px;width:{pct*100:.0f}%;"
            f"background:{color};border-radius:6px;'></span></span></div>",
            unsafe_allow_html=True)


def verdict_of(info, tag):
    """Return (uppercase verdict label, color) — no emoji, color carries meaning."""
    state = info.get("state", "?")
    if state == tag:
        return f"MATCH · {tag}", SEM["review_not_warranted"]["fg"]
    if info.get("is_de"):
        return "DELAWARE SIGNAL", SEM["insufficient_evidence"]["fg"]
    return f"DIFFERS FROM {tag}", SEM["review_warranted"]["fg"]


def evidence_timeline(phase, tag):
    """Sources laid out as an audit log: mono timestamp, node, source, verdict."""
    per_source = phase.get("per_source", {})
    present = [(s, per_source[s]) for s in SOURCE_ORDER
               if per_source.get(s, {}).get("present")]
    if not present:
        st.caption("No external records linked to this candidate.")
        return
    present.sort(key=lambda si: si[1].get("date", ""))
    for src, info in present:
        label, color = verdict_of(info, tag)
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:10px;padding:7px 0;"
            f"border-bottom:1px solid #eceef1;font-size:13px;'>"
            f"<span class='mono' style='color:#6b7480;font-size:11.5px;width:78px;"
            f"flex:none'>{info.get('date','')}</span>"
            f"<span style='width:8px;height:8px;border-radius:2px;background:{color};"
            f"flex:none;transform:translateY(1px)'></span>"
            f"<span style='flex:1'><b style='color:#2b3138'>"
            f"{SOURCE_LABELS.get(src, src)}</b>&nbsp;&nbsp;"
            f"<span class='mono' style='background:#eceef1;border:1px solid #e2e5e9;"
            f"border-radius:3px;padding:0 5px;font-size:11px;color:#2b3138'>"
            f"{info.get('state','?')}</span> "
            f"<span style='color:{color};font-size:10.5px;font-weight:700;"
            f"letter-spacing:.4px'>{label}</span></span>"
            f"<span class='mono' style='color:#9aa2ac;font-size:11px'>w "
            f"{info.get('weight',0):.2f}</span></div>",
            unsafe_allow_html=True)


def evidence_dict(cid, phase, tag):
    return {
        "candidate_record_id": cid,
        "home_state": tag,
        "predicted_class": phase["predicted_class"],
        "probabilities": phase["probabilities"],
        "review_priority": phase["review_priority"],
        "evidence_trail": phase.get("evidence_trail", []),
        "per_source": phase.get("per_source", {}),
        "n_sources": phase.get("n_sources", 0),
    }


def gains_svg(tr):
    """Cumulative-gains curve: prioritized queue vs. random order (inline SVG)."""
    xs = tr.get("gains_x") or []
    ys = tr.get("gains_prioritized") or []
    if not xs:
        return ""
    W, H, pad = 340, 200, 30
    x0, y0, x1, y1 = pad, H - pad, W - 6, 8
    px = lambda f: x0 + (x1 - x0) * f
    py = lambda f: y0 + (y1 - y0) * f
    d = f"M {px(0):.1f} {py(0):.1f}" + "".join(
        f" L {px(x):.1f} {py(y):.1f}" for x, y in zip(xs, ys))
    grid = "".join(
        f"<line x1='{px(g):.1f}' y1='{y1}' x2='{px(g):.1f}' y2='{y0}' stroke='#eceef1'/>"
        f"<line x1='{x0}' y1='{py(g):.1f}' x2='{x1}' y2='{py(g):.1f}' stroke='#eceef1'/>"
        for g in (0, .25, .5, .75, 1))
    return (
        f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:360px'>{grid}"
        f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y1}' stroke='#9aa2ac' stroke-dasharray='4 3'/>"
        f"<path d='{d}' fill='none' stroke='#1f4d6b' stroke-width='2'/>"
        f"<line x1='{x0}' y1='{y1}' x2='{x0}' y2='{y0}' stroke='#cfd4d9'/>"
        f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#cfd4d9'/>"
        f"<text x='{(x0+x1)/2}' y='{H-4}' text-anchor='middle' font-size='10' fill='#6b7480'>share of caseload reviewed</text>"
        f"<text x='10' y='{(y0+y1)/2}' font-size='10' fill='#6b7480' transform='rotate(-90 10 {(y0+y1)/2})' text-anchor='middle'>true cases found</text>"
        f"<text x='{x1-4}' y='{y1+42}' text-anchor='end' font-size='10' fill='#1f4d6b' font-weight='600'>prioritized</text>"
        f"<text x='{x1-4}' y='{py(.62):.0f}' text-anchor='end' font-size='10' fill='#9aa2ac'>random</text></svg>")


def set_case(cid):
    st.session_state.sel_cid = cid
    st.session_state.search = ""


# --------------------------------------------------------------------------- #
# Sidebar: case picker (drives the Case File + T0->T1 tabs)
# --------------------------------------------------------------------------- #
st.session_state.setdefault("sel_cid", FLIP_CID)

with st.sidebar:
    st.markdown("<div style='font-size:12.5px;font-weight:600;color:#f2f5f8;"
                "letter-spacing:.3px;padding:2px 0 8px'>DELAWARE DMV · OSTR</div>",
                unsafe_allow_html=True)
    key_ok = _get_client() is not None
    if key_ok:
        st.success("AI note service connected — reviewer notes generated live.",
                   icon=":material/check_circle:")
    else:
        st.info("AI note service offline — notes use the built-in template model "
                "(fully readable).", icon=":material/info:")

    st.caption("PINNED CASES")
    for c in CURATED:
        flag = "  ·  changed" if c["cid"] == FLIP_CID else ""
        st.button(f"{c['scenario']}  ·  {c['tag']}{flag}",
                  key=f"pick_{c['cid']}", on_click=set_case, args=(c["cid"],),
                  width="stretch")

    search = st.text_input("…or search any candidate_record_id", key="search",
                           placeholder="CAN-XXXXXXXXXX")
    s = search.strip().upper()
    if s:
        if s in data:
            st.session_state.sel_cid = s
        else:
            st.error(f"No candidate '{s}'")

    phase_name = st.radio("Evidence phase (Case File tab)", ["T1", "T0"],
                          horizontal=True,
                          help="T0 = initial evidence; T1 = after the "
                               "evidence-update stream arrives.")

cid = st.session_state.sel_cid
rec = data.get(cid)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:12px;border-bottom:1px solid "
    "#e2e5e9;padding-bottom:10px;margin-bottom:6px'>"
    "<span style='font-size:18px;font-weight:700;color:#171b20;letter-spacing:-.2px'>"
    "Out-of-State Tag Review</span>"
    "<span style='font-size:12px;color:#6b7480'>Delaware DMV · Decision Support "
    "Console · OSTR</span></div>", unsafe_allow_html=True)
st.caption("Decision-support triage — it prioritizes cases for staff review. "
           "It does **not** make a legal, residency, fee, or enforcement "
           "determination.")

tab_work, tab_case, tab_evo, tab_honest = st.tabs([
    "Worklist",
    "Case detail",
    "Evidence comparison",
    "Model performance",
])


# --------------------------------------------------------------------------- #
# TAB 1 — Morning Worklist
# --------------------------------------------------------------------------- #
with tab_work:
    wl = worklist_top(12)
    st.subheader("Review worklist — highest priority first")
    st.caption(f"{len(wl)} of the top **review-warranted** cases at T1, ranked "
               "by review_priority. Open a row to load it into Case detail.")
    # column header
    h = st.columns([0.9, 3.0, 1.0, 1.3, 1.3, 1.1, 1.2])
    for col, lbl in zip(h, ["PRIORITY", "CASE ID", "TAG", "CONFLICTS",
                            "DE SIGNALS", "SOURCES", ""]):
        col.markdown(f"<div style='font-size:10px;font-weight:600;letter-spacing:.5px;"
                     f"color:#6b7480'>{lbl}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:2px 0 6px;border:0;border-top:1px solid #e2e5e9'>",
                unsafe_allow_html=True)
    for r in wl:
        c1, c2, c3, c4, c5, c6, c7 = st.columns([0.9, 3.0, 1.0, 1.3, 1.3, 1.1, 1.2])
        tint = max(0.0, min(1.0, (r["priority"] - 0.28) / 0.45)) * 0.11
        c1.markdown(f"<div style='font-family:monospace;font-weight:700;color:#171b20;"
                    f"background:rgba(23,27,32,{tint:.3f});padding:2px 6px;border-radius:3px;"
                    f"display:inline-block'>{r['priority']:.3f}</div>", unsafe_allow_html=True)
        c2.markdown(f"<span style='font-family:monospace;font-weight:600;color:#171b20'>"
                    f"{r['cid']}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='font-family:monospace;background:#eceef1;"
                    f"border:1px solid #e2e5e9;border-radius:3px;padding:0 5px;"
                    f"font-size:12px'>{r['tag']}</span>", unsafe_allow_html=True)
        c4.markdown(f"<span style='font-family:monospace;color:"
                    f"{'#2b3138' if r['differ'] else '#9aa2ac'}'>{r['differ']}</span>",
                    unsafe_allow_html=True)
        c5.markdown(f"<span style='font-family:monospace;color:"
                    f"{'#2b3138' if r['de_signals'] else '#9aa2ac'}'>{r['de_signals']}</span>",
                    unsafe_allow_html=True)
        c6.markdown(f"<span style='font-family:monospace;color:#2b3138'>{r['n_sources']}</span>",
                    unsafe_allow_html=True)
        c7.button("Open", key=f"wl_{r['cid']}", on_click=set_case,
                  args=(r["cid"],), width="stretch")
    st.caption("A reviewer works top-down and can stop when time runs out, having "
               "cleared the highest-priority cases first.")


# --------------------------------------------------------------------------- #
# TAB 2 — Case File
# --------------------------------------------------------------------------- #
with tab_case:
    if rec is None:
        st.error(f"No candidate found with ID '{cid}'.")
    else:
        p = rec[phase_name]
        tag = p.get("home_state") or rec.get("observed_state") or "unknown"
        pred = p["predicted_class"]
        probs = p["probabilities"]

        meta = next((c for c in CURATED if c["cid"] == cid), None)
        head_l, head_r = st.columns([2, 1])
        with head_l:
            st.markdown(f"#### Case file · `{cid}` · phase **{phase_name}**")
            st.markdown(class_badge(pred), unsafe_allow_html=True)
        with head_r:
            if meta:
                st.caption(f"**Demo case:** {meta['scenario']}")
        st.write("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current tag state", tag)
        m2.metric("Review priority", f"{p['review_priority']:.2f}")
        m3.metric("Linked sources", p.get("n_sources", 0))
        top_prob = max(probs, key=probs.get)
        m4.metric("Top class", CLASS_LABEL[top_prob].title(),
                  f"{probs[top_prob]:.0%}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### Calibrated probabilities")
            prob_bars(probs, highlight=pred)
            st.caption("Probabilities are calibrated (Platt scaling), so they "
                       "can be read as real confidences, not raw votes.")
        with right:
            st.markdown("##### Evidence timeline")
            st.caption(f"Each source's most recent record vs. the **{tag}** tag.")
            evidence_timeline(p, tag)

        with st.expander("Raw evidence-trail notes (verbatim from the pipeline)"):
            for t in p.get("evidence_trail", []) or ["(none)"]:
                st.write("• " + t)

        st.divider()
        st.markdown("##### Reviewer note")
        cache_key = f"expl::{cid}::{phase_name}"
        cta = st.button("Generate reviewer note", type="primary",
                        key="gen_expl")
        if cta:
            with st.spinner("Generating note…"):
                text, source = generate_case_explanation(
                    evidence_dict(cid, p, tag))
            st.session_state[cache_key] = (text, source)
        if cache_key in st.session_state:
            text, source = st.session_state[cache_key]
            st.markdown(
                f"<div style='border:1px solid #e2e5e9;border-left:3px solid "
                f"#1f4d6b;background:#fbfbf9;border-radius:0 6px 6px 0;"
                f"padding:12px 14px;font-size:13.5px;line-height:1.6;"
                f"color:#2b3138'>{text}</div>", unsafe_allow_html=True)
            badge = ("AI-generated · gemini-2.5-flash" if source == "gemini"
                     else "rule-based evidence summary")
            st.caption(f"{badge} · summarizes the linked evidence, not the "
                       "model's internal reasoning. Decision-support only — not a "
                       "legal, residency, or enforcement determination.")


# --------------------------------------------------------------------------- #
# TAB 3 — New Evidence (T0 -> T1)
# --------------------------------------------------------------------------- #
with tab_evo:
    if rec is None:
        st.error(f"No candidate found with ID '{cid}'.")
    else:
        if cid != FLIP_CID:
            st.info(f"The clearest before/after is the flagged flip case.",
                    icon=":material/info:")
            st.button("Load the flip case (CAN-CKLXPC2OP3) →",
                      on_click=set_case, args=(FLIP_CID,), key="load_flip")
        t0, t1 = rec["T0"], rec["T1"]
        tag = t1.get("home_state") or rec.get("observed_state") or "unknown"

        st.subheader(f"`{cid}` — before and after new evidence")
        changed = t0["predicted_class"] != t1["predicted_class"]
        if changed:
            st.markdown(
                f"When the T1 evidence stream arrives, this case moves from "
                f"{class_badge(t0['predicted_class'], 16)} to "
                f"{class_badge(t1['predicted_class'], 16)}.",
                unsafe_allow_html=True)
        else:
            st.caption("This case keeps the same classification across phases.")
        st.write("")

        col0, col1 = st.columns(2)
        for col, name, ph in ((col0, "T0 — initial evidence", t0),
                              (col1, "T1 — after new evidence", t1)):
            with col:
                st.markdown(f"**{name}**")
                st.markdown(class_badge(ph["predicted_class"], 18),
                            unsafe_allow_html=True)
                st.metric("Review priority", f"{ph['review_priority']:.2f}")
                prob_bars(ph["probabilities"], highlight=ph["predicted_class"])
                st.markdown("<br>", unsafe_allow_html=True)
                evidence_timeline(ph, tag)

        # What changed, per source
        ps0 = t0.get("per_source", {})
        ps1 = t1.get("per_source", {})
        diffs = []
        for src in SOURCE_ORDER:
            a, b = ps0.get(src, {}), ps1.get(src, {})
            if a.get("state") != b.get("state") and b.get("present"):
                diffs.append(f"**{SOURCE_LABELS.get(src, src)}**: "
                             f"{a.get('state','—') if a.get('present') else '—'} "
                             f"→ **{b.get('state','?')}**")
        if diffs:
            st.divider()
            st.markdown("##### What the new evidence changed")
            for d in diffs:
                st.markdown("• " + d)


# --------------------------------------------------------------------------- #
# TAB 4 — Model Honesty
# --------------------------------------------------------------------------- #
with tab_honest:
    st.subheader("How good is the model, really?")
    if not metrics:
        st.warning("metrics.json not found — run `python export_metrics.py`.")
    else:
        mkey = "label_t1" if phase_name == "T1" else "label_t0"
        m = metrics.get(mkey) or metrics.get("label_t0")
        acc = m["accuracy"]
        base = m["baseline_majority_accuracy"]
        chance = m["chance_accuracy"]
        tr = m.get("triage", {})
        pak = tr.get("precision_at_k", {})
        at = pak.get("75") or pak.get("50") or {"precision": 0, "lift": 0}
        k_used = 75 if pak.get("75") else 50
        effort = tr.get("effort_saved_at_75", 0)
        base_rate = tr.get("base_rate", 0)

        # ---- lead with usefulness: triage effectiveness ----
        st.markdown("##### Triage effectiveness — the number that matters")
        tcol, gcol = st.columns([1.3, 1])
        with tcol:
            t1c, t2c, t3c = st.columns(3)
            t1c.metric(f"Triage lift @ top {k_used}", f"{at['lift']:.1f}×",
                       f"{at['precision']:.0%} true vs {base_rate:.0%} base")
            t2c.metric("Review effort saved", f"{effort:.0%}",
                       "fewer files opened")
            t3c.metric("Base rate", f"{base_rate:.0%}", "share needing review")
            st.caption(
                "Ranked by the model's **out-of-fold** confidence — exactly how "
                "the worklist orders cases. A reviewer working top-down reaches "
                f"confirmed out-of-state cases roughly **{at['lift']:.1f}× faster** "
                "than working the queue in arbitrary order.")
        with gcol:
            st.markdown(gains_svg(tr), unsafe_allow_html=True)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("3-class CV accuracy", f"{acc:.0%}",
                  f"+{(acc-base)*100:.0f} pts vs baseline")
        c2.metric("Majority-class baseline", f"{base:.0%}")
        c3.metric("Random chance (3 classes)", f"{chance:.0%}")

        st.caption(f"5-fold cross-validation over all {m['n']} development "
                   f"labels ({m['label_col']}): every label is predicted by a "
                   "model that never trained on it. Honest, not the optimistic "
                   "single-split number.")

        st.markdown("##### Confusion matrix  \n<small>rows = true label, "
                    "columns = model prediction; diagonal = correct</small>",
                    unsafe_allow_html=True)
        classes = m["classes"]
        cm = m["confusion_matrix"]
        short = {"review_warranted": "Warranted",
                 "review_not_warranted": "Not warranted",
                 "insufficient_evidence": "Insufficient"}
        html = ["<table style='border-collapse:collapse;font-size:14px;'>"]
        html.append("<tr><th style='padding:6px 10px;'></th>"
                    + "".join(f"<th style='padding:6px 10px;color:#555;'>pred:<br>"
                              f"{short[c]}</th>" for c in classes)
                    + "<th style='padding:6px 10px;color:#555;'>recall</th></tr>")
        for i, ct in enumerate(classes):
            row = cm[i]
            rtot = sum(row) or 1
            html.append(f"<tr><th style='padding:6px 10px;text-align:right;"
                        f"color:#555;'>true:<br>{short[ct]}</th>")
            for j, v in enumerate(row):
                correct = i == j
                bg = f"rgba(30,132,73,{0.15 + 0.6*v/rtot:.2f})" if correct \
                    else f"rgba(192,57,43,{0.10 + 0.4*v/rtot:.2f})"
                html.append(f"<td style='padding:10px 16px;text-align:center;"
                            f"background:{bg};font-weight:{'700' if correct else '400'};"
                            f"border:1px solid #fff;'>{v}</td>")
            html.append(f"<td style='padding:6px 10px;text-align:center;"
                        f"color:#555;'>{row[i]/rtot:.0%}</td></tr>")
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)

        st.markdown(
            f"**Reading it honestly.** Raw 3-class accuracy is ~{acc:.0%} and the "
            "model is weak on *insufficient_evidence* — but accuracy is the wrong "
            "yardstick for a queue. As a **triage tool it delivers "
            f"{at['lift']:.1f}× lift and ~{effort:.0%} less review effort**, which "
            "is its actual job: put the right cases in front of staff first. It "
            "prioritizes human review; it is **not** an automated decision-maker.")
        with st.expander("Why we trust this is a real edge (significance & data)"):
            st.markdown(
                "- Across 5 folds the model beats a majority-class baseline "
                f"**{acc:.0%} vs {base:.0%}** (paired t-test p ≈ 0.001).\n"
                "- Every label **and** the ranking score are out-of-fold — "
                "predicted by a model that never trained on that case.\n"
                "- We deliberately keep the forest shallow (max_depth 5) because "
                "there are only 300 labels — deeper models overfit and do not "
                "improve cross-validated accuracy.\n"
                "- Data is the **DTI-provided synthetic** development set (300 "
                "labels), so the accuracy ceiling reflects synthetic label noise; "
                "the pipeline and the triage lift are real.\n"
                "- The single 75-case hold-out split is underpowered (p ≈ 0.31), "
                "so we report the cross-validated numbers instead.")

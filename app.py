"""
DMV Out-of-State Tag Review -- judge-facing demo.

Type or pick a candidate_record_id and see, for the chosen evidence phase:
  - predicted classification (color-coded)
  - the 3 calibrated class probabilities
  - review_priority
  - the evidence trail (which source said which state, when, and whether it
    matched or differed from the vehicle's current tag state)
  - a plain-English Gemini explanation generated live for a DMV reviewer

Run:  streamlit run app.py
Data comes from demo_data/case_data.json (produced by export_case_data.py),
so the app loads instantly and never re-runs the model live.
"""
import json
import streamlit as st

from explain import generate_case_explanation, _get_client

st.set_page_config(page_title="DMV Out-of-State Tag Review",
                   page_icon="🚗", layout="wide")

SOURCE_LABELS = {
    "address": "Address history",
    "license": "License / credential",
    "work": "Work location",
    "vehicle": "Vehicle title event",
    "external": "External signal",
}
CLASS_COLOR = {
    "review_warranted": "#c0392b",
    "review_not_warranted": "#1e8449",
    "insufficient_evidence": "#b9770e",
}
CLASS_LABEL = {
    "review_warranted": "REVIEW WARRANTED",
    "review_not_warranted": "REVIEW NOT WARRANTED",
    "insufficient_evidence": "INSUFFICIENT EVIDENCE",
}


@st.cache_data
def load_data():
    with open("demo_data/case_data.json") as fh:
        data = json.load(fh)
    try:
        with open("demo_data/sample_ids.json") as fh:
            samples = json.load(fh)
    except FileNotFoundError:
        samples = {}
    return data, samples


data, samples = load_data()
all_ids = data  # dict; membership test is O(1)

st.title("🚗 DMV Out-of-State Tag Holder Review")
st.caption("Decision-support triage. This tool prioritizes cases for staff "
           "review; it does not make a legal, residency, fee, or enforcement "
           "determination.")

# ---- key status indicator ----
key_ok = _get_client() is not None
if key_ok:
    st.success("Gemini API connected — explanations are AI-generated live.", icon="✅")
else:
    st.warning("No Gemini API key found (set GEMINI_API_KEY or .env). "
               "Explanations will use the built-in template fallback.", icon="⚠️")

# ---- input row ----
first_sample = None
for cls in ("review_warranted", "review_not_warranted", "insufficient_evidence"):
    if samples.get(cls):
        first_sample = samples[cls][0]
        break
if first_sample is None:
    first_sample = next(iter(data))

st.session_state.setdefault("cid", first_sample)


def pick(cid):
    st.session_state.cid = cid


with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.text_input("Candidate Record ID", key="cid",
                      help="Type a full candidate_record_id, e.g. CAN-XXXXXXXXXX")
    with c2:
        phase = st.radio("Evidence phase", ["T1", "T0"], horizontal=True,
                         help="T0 = initial evidence; T1 = after the evidence "
                              "update stream is incorporated")

if samples:
    st.write("**Try an example:**")
    ex_cols = st.columns(3)
    for col, cls in zip(ex_cols, ("review_warranted", "review_not_warranted",
                                  "insufficient_evidence")):
        with col:
            st.markdown(f"<small>{CLASS_LABEL.get(cls, cls)}</small>",
                        unsafe_allow_html=True)
            for cid in samples.get(cls, [])[:3]:
                st.button(cid, key=f"ex_{cid}", on_click=pick, args=(cid,),
                          width="stretch")

cid = st.session_state.cid.strip().upper()
st.divider()

if cid not in data:
    st.error(f"No candidate found with ID '{cid}'. "
             "Check the ID or click one of the example buttons above.")
    st.stop()

rec = data[cid]
p = rec[phase]
home = p.get("home_state") or rec.get("observed_state") or "unknown"
pred = p["predicted_class"]
probs = p["probabilities"]

# ---- headline ----
color = CLASS_COLOR.get(pred, "#555")
st.markdown(
    f"<div style='padding:14px 18px;border-radius:10px;background:{color};"
    f"color:white;font-size:26px;font-weight:700;'>{CLASS_LABEL.get(pred, pred)}"
    f"</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Case ID", cid)
m2.metric("Current tag state", home)
m3.metric("Review priority", f"{p['review_priority']:.2f}")
m4.metric("Linked sources", p.get("n_sources", 0))
st.progress(min(max(float(p["review_priority"]), 0.0), 1.0),
            text=f"review_priority = {p['review_priority']:.2f} (higher = look sooner)")

# ---- probabilities ----
st.subheader("Calibrated class probabilities")
pc1, pc2, pc3 = st.columns(3)
pc1.metric("Review warranted", f"{probs['review_warranted']:.0%}")
pc2.metric("Review NOT warranted", f"{probs['review_not_warranted']:.0%}")
pc3.metric("Insufficient evidence", f"{probs['insufficient_evidence']:.0%}")
st.bar_chart({
    "probability": {
        "review_warranted": probs["review_warranted"],
        "review_not_warranted": probs["review_not_warranted"],
        "insufficient_evidence": probs["insufficient_evidence"],
    }
}, horizontal=True)

# ---- evidence trail ----
st.subheader("Evidence trail — what each source said")
per_source = p.get("per_source", {})
rows = []
for src in ("address", "license", "work", "vehicle", "external"):
    info = per_source.get(src, {})
    if not info.get("present"):
        continue
    state = info.get("state", "?")
    if state == home:
        verdict = "✅ matches tag state"
    elif info.get("is_de"):
        verdict = "🔶 points to DE"
    else:
        verdict = "⚠️ differs from tag state"
    rows.append({
        "Source": SOURCE_LABELS.get(src, src),
        "State": state,
        "Most recent date": info.get("date", ""),
        "vs. tag state": verdict,
        "Weight": info.get("weight", 0),
    })
if rows:
    st.dataframe(rows, width="stretch", hide_index=True)
else:
    st.info("No source evidence could be linked to this candidate.")

with st.expander("Raw evidence-trail notes (verbatim from the pipeline)"):
    trail = p.get("evidence_trail", [])
    if trail:
        for t in trail:
            st.write("• " + t)
    else:
        st.write("(none)")

# ---- Gemini explanation ----
st.subheader("Plain-English explanation for the reviewer")
cache_key = f"expl::{cid}::{phase}"
if st.button("🧠 Generate explanation", type="primary"):
    evidence = {
        "candidate_record_id": cid,
        "home_state": home,
        "predicted_class": pred,
        "probabilities": probs,
        "review_priority": p["review_priority"],
        "evidence_trail": p.get("evidence_trail", []),
        "per_source": p.get("per_source", {}),
        "n_sources": p.get("n_sources", 0),
    }
    with st.spinner("Generating explanation..."):
        text, source = generate_case_explanation(evidence)
    st.session_state[cache_key] = (text, source)

if cache_key in st.session_state:
    text, source = st.session_state[cache_key]
    badge = "Gemini (live)" if source == "gemini" else "template fallback"
    st.info(text)
    st.caption(f"Generated by: {badge}")

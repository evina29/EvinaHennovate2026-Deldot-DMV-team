"""
Plain-English case explanations for non-technical DMV reviewers.

generate_case_explanation() takes the evidence we already computed for a
flagged case (predicted class, calibrated probabilities, review_priority,
and the per-source evidence trail) and asks Gemini to turn it into a short,
readable paragraph a front-desk DMV reviewer can act on.

Design rules baked into the prompt (so we don't oversell to judges):
  - The model may ONLY use the evidence passed in. It must not invent
    addresses, names, dates, or states that aren't in the trail.
  - It must frame the output as a triage suggestion, not a legal or
    residency determination.
  - It must be honest about uncertainty when the class is
    insufficient_evidence or probabilities are close.

If GEMINI_API_KEY is not set, generate_case_explanation() falls back to a
deterministic template so the demo never hard-crashes in front of judges;
the UI labels which path produced the text.
"""
import os

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

CLASS_PLAIN = {
    "review_warranted": "recommended for staff review",
    "review_not_warranted": "not recommended for review at this time",
    "insufficient_evidence": "not enough evidence to decide either way",
}

SYSTEM_INSTRUCTION = (
    "You are a plain-language assistant for Delaware DMV staff. You explain "
    "why an automated triage tool flagged (or did not flag) a vehicle "
    "registration case for human review. Your reader is a non-technical "
    "front-desk reviewer, not a data scientist.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY the evidence provided. Never invent an address, name, date, "
    "state, or source that is not in the evidence.\n"
    "2. This is decision-support triage, not a legal, residency, fee, or "
    "enforcement determination. Say so in plain terms.\n"
    "3. Be honest about uncertainty. If the classification is "
    "'insufficient_evidence' or the probabilities are close, say the signal "
    "is weak and explain that more information is needed.\n"
    "4. Write ONE short paragraph (3-5 sentences), no bullet points, no "
    "jargon, no probabilities as raw decimals -- describe confidence in "
    "words (e.g. 'a moderate signal', 'a weak signal').\n"
    "5. Explain the WHY: which sources pointed to which state, and whether "
    "they agreed with or differed from the state the vehicle is currently "
    "tagged in.\n"
    "6. End with a one-line suggested next step for the reviewer."
)


def _confidence_word(p):
    if p >= 0.60:
        return "a strong"
    if p >= 0.45:
        return "a moderate"
    if p >= 0.35:
        return "a weak"
    return "a very weak"


def format_evidence_prompt(evidence):
    """Turn the structured evidence dict into the user-facing prompt text."""
    cid = evidence["candidate_record_id"]
    home = evidence.get("home_state") or "unknown"
    pred = evidence["predicted_class"]
    probs = evidence["probabilities"]
    priority = evidence["review_priority"]
    trail = evidence.get("evidence_trail") or []

    lines = [
        f"Case ID: {cid}",
        f"Vehicle is currently tagged/observed in state: {home}",
        f"Automated classification: {pred} ({CLASS_PLAIN.get(pred, pred)})",
        "Model confidence by outcome (calibrated):",
        f"  - review warranted: {probs.get('review_warranted', 0):.0%}",
        f"  - review not warranted: {probs.get('review_not_warranted', 0):.0%}",
        f"  - insufficient evidence: {probs.get('insufficient_evidence', 0):.0%}",
        f"Review priority score (0-1, higher = look sooner): {priority:.2f}",
        "",
        "Evidence trail (each line is one independent source, its most "
        "recent dated record, and whether it matched or differed from the "
        "tag state):",
    ]
    if trail:
        for t in trail:
            lines.append(f"  - {t}")
    else:
        lines.append("  - (no source evidence could be linked to this case)")
    lines.append("")
    lines.append(
        "Write the plain-English explanation for the DMV reviewer now."
    )
    return "\n".join(lines)


SRC_PHRASE = {
    "address": "their address on file",
    "license": "their license/credential record",
    "work": "their employer location",
    "vehicle": "a vehicle title event",
    "external": "an external records signal",
}
_SRC_ORDER = ["address", "license", "work", "vehicle", "external"]


def _human_join(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _present_sources(evidence):
    """Return list of (src, state, is_de, matches_home) for linked sources.
    Prefers structured per_source; falls back to counting the trail."""
    per_source = evidence.get("per_source") or {}
    home = evidence.get("home_state")
    out = []
    for src in _SRC_ORDER:
        info = per_source.get(src)
        if info and info.get("present"):
            state = info.get("state")
            out.append((src, state, bool(info.get("is_de")),
                        state == home))
    return out


def _fallback_explanation(evidence):
    """Deterministic, no-API explanation. Grounded only in the evidence."""
    home = evidence.get("home_state") or "its current"
    pred = evidence["predicted_class"]
    probs = evidence["probabilities"]
    trail = evidence.get("evidence_trail") or []
    p_top = probs.get(pred, max(probs.values()) if probs else 0)
    conf = _confidence_word(p_top)

    present = _present_sources(evidence)
    if present:
        n = len(present)
        differ = [(s, st) for (s, st, de, m) in present if not m and st]
        agree = [(s, st, de, m) for (s, st, de, m) in present if m]
        de_srcs = [s for (s, st, de, m) in present if de and not m]
    else:  # no structured data -> fall back to trail-line counting
        n = len(trail)
        n_differ = sum(1 for t in trail if "differs from tag state" in t)
        differ = [("", "")] * n_differ
        agree = [("",)] * (n - n_differ)
        de_srcs = []

    differ_states = sorted({st for _, st in differ if st})
    differ_phrases = _human_join(
        [SRC_PHRASE.get(s, s) for s, _ in differ if s]) if present else ""

    if pred == "insufficient_evidence" or n == 0:
        if n == 0:
            reason = ("No outside records could be linked to this person, so "
                      "there is nothing to compare against the current tag.")
        elif n == 1:
            reason = ("Only a single source could be linked, which is not "
                      "enough on its own to say whether the vehicle should be "
                      "tagged elsewhere.")
        else:
            reason = (f"The {n} linked sources conflict — "
                      f"{len(differ)} point away from {home}"
                      + (f" (toward {_human_join(differ_states)})" if differ_states else "")
                      + f" while {len(agree)} agree with it — so no clear "
                        "conclusion can be drawn either way.")
        body = f"This case shows {conf}, inconclusive signal. {reason} " \
               "It is being surfaced as uncertain rather than as a confirmed issue."
    elif pred == "review_warranted":
        where = f" (toward {_human_join(differ_states)})" if differ_states else ""
        who = f" — specifically {differ_phrases}" if differ_phrases else ""
        body = (
            f"This case shows {conf} signal that the vehicle may actually "
            f"belong in a different state than its current {home} tag. Of {n} "
            f"linked source(s), {len(differ)} disagree with the {home} tag"
            f"{where}{who}."
            + (" Because independent records point to Delaware, this is worth "
               "a closer look." if de_srcs else
               " Because multiple independent records disagree with the tag on "
               "file, it is worth a closer look.")
        )
    else:  # review_not_warranted
        body = (
            f"This case shows {conf} signal, and the tool did not rate it as "
            f"needing review. Of {n} linked source(s), {len(agree)} agree with "
            f"the current {home} tag"
            + (f" and {len(differ)} differ"
               + (f" (toward {_human_join(differ_states)})" if differ_states else "")
               + "; on balance the differing records were not enough to move "
                 "this case above the review threshold, but they can be checked "
                 "if the reviewer wishes." if differ else
               ", so the records are consistent with the tag on file.")
        )

    next_step = {
        "review_warranted": "pull the full record and verify the differing "
                            "source(s) before any action.",
        "review_not_warranted": "no action needed now; revisit if new "
                                "evidence arrives.",
        "insufficient_evidence": "request or wait for additional records "
                                "before deciding.",
    }.get(pred, "review manually.")

    return (
        f"{body} This is an automated triage suggestion to help prioritize "
        "staff attention, not a legal or residency determination. "
        f"Suggested next step: {next_step}"
    ).strip()


def _load_dotenv():
    """Minimal .env loader (no dependency) so a pasted key persists locally."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai
    _client = genai.Client(api_key=api_key)
    return _client


def generate_case_explanation(evidence, model=None, force_fallback=False):
    """
    evidence: dict with keys candidate_record_id, home_state,
        predicted_class, probabilities (dict), review_priority,
        evidence_trail (list[str]).

    Returns (text, source) where source is 'gemini' or 'fallback'.
    """
    if force_fallback:
        return _fallback_explanation(evidence), "fallback"

    client = _get_client()  # loads .env as a side effect
    if client is None:
        return _fallback_explanation(evidence), "fallback"

    model = model or os.environ.get("GEMINI_MODEL") or MODEL
    from google.genai import types
    prompt = format_evidence_prompt(evidence)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=400,
            ),
        )
        text = (resp.text or "").strip()
        if not text:
            return _fallback_explanation(evidence), "fallback"
        return text, "gemini"
    except Exception as e:
        return (
            f"{_fallback_explanation(evidence)}\n\n[Note: live AI explanation "
            f"unavailable, showing template fallback. Error: {e}]",
            "fallback",
        )


if __name__ == "__main__":
    # Smoke test with a synthetic evidence dict (no data files needed).
    demo = {
        "candidate_record_id": "CAN-DEMO123",
        "home_state": "PA",
        "predicted_class": "review_warranted",
        "probabilities": {
            "review_warranted": 0.58,
            "review_not_warranted": 0.24,
            "insufficient_evidence": 0.18,
        },
        "review_priority": 0.71,
        "evidence_trail": [
            "[differs from tag state PA] address on file changed to DE (2026-03-02)",
            "[differs from tag state PA] employer location DE (2026-01-15)",
            "[matches tag state PA] license/credential state PA (active, 2025-11-20)",
        ],
    }
    text, src = generate_case_explanation(demo)
    print(f"[source={src}]\n{text}")

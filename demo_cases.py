"""
Curated candidate_record_ids for the live judge demo, chosen from the real
pipeline output so the presenter never has to search live and risk landing on
a boring case. Each was hand-verified against demo_data/case_data.json.

Keep this list short and strong. The FLIP case is the star of the demo.
"""

# The one to open first: at T0 there isn't enough to act; at T1 a Delaware
# title correction arrives and the same case becomes review_warranted.
FLIP_CID = "CAN-CKLXPC2OP3"

CURATED = [
    {
        "cid": "CAN-CKLXPC2OP3",
        "tag": "MD",
        "scenario": "New evidence flips it",
        "badge": "flip",
        "why": "MD-tagged. At T0 only the license + an external signal point to "
               "Delaware — not enough, so it sits as INSUFFICIENT. At T1 a "
               "Delaware vehicle-title correction lands; now three independent "
               "sources say DE and it becomes REVIEW WARRANTED. This is the "
               "system reacting to new evidence — open the T0 -> T1 tab for it.",
    },
    {
        "cid": "CAN-6QEV4RPCT3",
        "tag": "PA",
        "scenario": "Strong review-warranted",
        "badge": "warranted",
        "why": "PA-tagged, but the license, the vehicle title, and an external "
               "records signal all point to Delaware. Multiple independent "
               "sources disagree with the PA tag -> a clear candidate for staff "
               "review (possible DE resident on an out-of-state tag).",
    },
    {
        "cid": "CAN-8L913O7UT4",
        "tag": "MD",
        "scenario": "Clear no review needed",
        "badge": "not_warranted",
        "why": "MD-tagged and every linked source agrees on MD. A stray DE "
               "address at T0 is corrected to MD by T1. The system does NOT cry "
               "wolf when the records are consistent.",
    },
    {
        "cid": "CAN-3EA4YQEIXT",
        "tag": "MD",
        "scenario": "Genuinely insufficient",
        "badge": "insufficient",
        "why": "MD-tagged with a real conflict: address and an external signal "
               "point to DE, but the license, employer, and vehicle title stay "
               "MD (and the license is expired). Signals disagree, so the tool "
               "honestly withholds judgement rather than guessing.",
    },
]

CURATED_IDS = [c["cid"] for c in CURATED]

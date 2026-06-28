"""Foresight Gap synthesis — the second weekly Claude pass.

Crosses the week's computed research signals (divergences, velocity inflections,
eroding-confidence flags, quadrant emerging/white-space, citation movers, new
clusters, lab activity, and *what changed this week*) with a living, user-maintained
societal-context file (``config/context.md``) and a fixed STEEP/PESTLE-plus scanning
framework, to surface **novel, not-yet-in-the-news risk implications** that sit in the
SEAM between AI research and broader societal forces.

Same architecture as ``llm.py``: one Claude call via the shared ``llm.call_claude``,
result baked into ``analysis["foresight_gap"]`` in the snapshot (no page-load calls),
fully fail-soft (no key / no SDK / any error -> None).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import llm

log = logging.getLogger("signal_lag.foresight")


# The fixed scanning framework — a taxonomy of *domains* to reason across so the
# synthesis never tunnels on technology alone. It defines DIMENSIONS, never specific
# trends (those live in the user-maintained config/context.md).
SCANNING_FRAMEWORK: dict[str, str] = {
    "Social": "norms, behaviors, public trust, culture, labor and skills, education",
    "Technological": "adjacent non-AI tech: biotech, energy, compute supply, networks, robotics",
    "Economic": "markets, capital flows, business models, costs, employment, consolidation",
    "Environmental": "climate, resources, energy and water demand, supply chains, physical infrastructure",
    "Political": "governance, elections, institutions, state capacity, public administration",
    "Legal/Regulatory": "law, liability, IP, standards, enforcement gaps, jurisdiction",
    "Security/Geopolitical": "national security, conflict, cyber, dual-use diffusion, alliances, export controls",
    "Demographic": "population, aging, migration, inequality, generational shifts",
}


def load_context(path: str | Path) -> str:
    """Read the living societal-context file. Fail-soft: returns '' if missing/empty.

    HTML comments (the instructional header) are stripped so only the analyst's actual
    content is sent to the model.
    """
    p = Path(path)
    if not p.exists():
        log.info("No context file at %s; foresight runs without a societal layer", p)
        return ""
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("Could not read context file %s: %s", p, e)
        return ""
    # Strip HTML comment blocks (the how-to-use header).
    import re
    body = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
    return body


def _lbl(snap: dict, key):
    return (snap.get("label_map") or {}).get(key, key)


def build_signal_digest(snap: dict, diff: dict) -> dict:
    """Pull the strongest already-computed signals into a compact digest.

    Reads exactly the snapshot keys the dashboard uses (divergence, inflections,
    sentiment, quadrant, citations, new_clusters, lab_activity) plus this week's
    movement from ``diff_snapshots``. Labels are resolved via ``label_map`` so the
    model reasons over readable names and can cite them traceably.
    """
    def L(k):
        return _lbl(snap, k)

    # Capability outpacing safety (flagged divergences), widest gap first.
    flagged = [d for d in snap.get("divergence", []) if d.get("lagging")]
    divergences = [
        {
            "pairing": d["pairing"],
            "capability": L(d["capability_topic"]),
            "safety": L(d["safety_topic"]),
            "cap_growth_pct_per_qtr": round(d["cap_growth"] * 100, 1),
            "saf_growth_pct_per_qtr": round(d["saf_growth"] * 100, 1),
            "gap": d["gap"],
            "volume_ratio_cap_over_saf": d.get("volume_ratio"),
        }
        for d in sorted(flagged, key=lambda d: d["gap"], reverse=True)
    ]

    # Velocity inflections (momentum).
    infl = snap.get("inflections", []) or []
    accel = sorted([i for i in infl if i.get("change", 0) > 0],
                   key=lambda i: i["change"], reverse=True)[:5]
    decel = sorted([i for i in infl if i.get("change", 0) < 0],
                   key=lambda i: i["change"])[:5]

    def inf_row(i):
        return {"topic": L(i["topic_key"]), "change_pct": round(i["change"] * 100, 1),
                "recent_per_qtr": round(i["recent_mean"], 1)}

    velocity = {"accelerating": [inf_row(i) for i in accel],
                "decelerating": [inf_row(i) for i in decel]}

    # Eroding confidence (rising critical share) — the weak/early-warning signal.
    sent = snap.get("sentiment", {}) or {}
    rising = [
        {"topic": L(k), "recent_critical_share_pct": round(v.get("recent_share", 0) * 100, 1),
         "trend_pts": round(v.get("trend", 0) * 100, 1), "n_recent": v.get("n_recent", 0)}
        for k, v in sorted(sent.items(), key=lambda kv: kv[1].get("trend", 0), reverse=True)
        if v.get("rising")
    ]

    # Quadrant emerging + white-space (where the field is thin or just igniting).
    quad = snap.get("quadrant", []) or []
    emerging = [L(q["topic_key"]) for q in quad if q.get("quadrant") == "emerging"]
    white_space = [L(q["topic_key"]) for q in quad if q.get("quadrant") == "white-space"]

    # Citation movers.
    cites = snap.get("citations", {}) or {}

    def cite_rows(bucket):
        return [
            {"arxiv_id": r.get("arxiv_id"), "title": r.get("title"),
             "cited_by_count": r.get("cited_by_count"), "url": r.get("url")}
            for r in (cites.get(bucket) or [])[:5]
        ]

    citation_movers = {"sleepers": cite_rows("sleepers"), "rapid_growth": cite_rows("rapid_growth")}

    # Emergent (unsupervised) clusters.
    new_clusters = snap.get("new_clusters", []) or []

    # Recent lab announcements (capability-leading signal).
    lab = snap.get("lab_activity", []) or []
    lab_rows = [
        {"source": p.get("source"), "title": p.get("title"),
         "topic": L(p["topic"]) if p.get("topic") else None, "published": p.get("published")}
        for p in lab[:10]
    ]

    # What changed THIS week (weight movement, not just static state).
    changes = {
        "first_run": diff.get("first_run", False),
        "prev_date": diff.get("prev_date"),
        "new_safety_lag_alerts": [
            {"capability": L(a["capability_topic"]), "safety": L(a["safety_topic"]),
             "cap_growth_pct": round(a["cap_growth"] * 100, 1),
             "saf_growth_pct": round(a["saf_growth"] * 100, 1)}
            for a in diff.get("new_alerts", [])
        ],
        "new_accelerations": [
            {"topic": L(a["topic_key"]), "change_pct": round(a["change"] * 100, 1)}
            for a in diff.get("new_accelerations", [])
        ],
        "new_citation_sleepers": [
            {"arxiv_id": s.get("arxiv_id"), "title": s.get("title")}
            for s in diff.get("new_sleepers", [])[:5]
        ],
    }

    return {
        "divergences_safety_lagging": divergences,
        "velocity": velocity,
        "eroding_confidence_rising_critical_share": rising,
        "quadrant": {"emerging": emerging, "white_space": white_space},
        "citation_movers": citation_movers,
        "new_emergent_clusters": new_clusters,
        "recent_lab_activity": lab_rows,
        "what_changed_this_week": changes,
    }


SYSTEM = (
    "You are a strategic foresight analyst. Your job is to surface RISKS THAT ARE NOT "
    "YET VISIBLE in the news or the AI research literature — early, structural, "
    "between-the-cracks risks that emerge when AI research trajectories collide with "
    "developments in the wider world. You are not a doom-sayer and not a hype machine: "
    "you generate candidate hypotheses for a human analyst to pressure-test. You reason "
    "rigorously across many societal domains, you ground every claim in the specific "
    "signals you are given, and you state plainly when you are extrapolating beyond them. "
    "Reasoning and traceability matter more than confidence."
)


def _instructions(max_risks: int) -> str:
    framework = "\n".join(f"  - {k}: {v}" for k, v in SCANNING_FRAMEWORK.items())
    return f"""\
You are given four inputs (as JSON / text below):
1. SIGNAL_DIGEST — the strongest signals signal-lag computed from real arXiv/OpenAlex
   data this week (capability-vs-safety divergences, velocity inflections, eroding-
   confidence flags, quadrant emerging/white-space topics, citation movers, emergent
   clusters, recent lab activity).
2. WHAT_CHANGED_THIS_WEEK — the week-over-week movement (new alerts, new accelerations,
   new sleepers). WEIGHT THIS: prioritize risks driven by this week's *movement*, not
   just static state.
3. SCANNING_FRAMEWORK — the domains to reason across (so you never tunnel on technology
   alone):
{framework}
4. SOCIETAL_CONTEXT — the current real-world state across those domains, maintained by
   the analyst.

CRITICAL INSTRUCTIONS:
- The VALUE IS IN THE SEAM. Generate risks that emerge ONLY when you CROSS the research
  signals with developments across the societal domains — the kind of risk no single
  community is tracking because it sits BETWEEN domains (e.g. a capability trend × a
  regulatory gap × a demographic shift). A risk that lives entirely inside AI research,
  or entirely inside one societal domain, is NOT what we want.
- The SOCIETAL_CONTEXT file and any examples in it are ILLUSTRATIVE AND NON-EXHAUSTIVE.
  Reason across the FULL scanning framework and your own knowledge of the current
  real-world state. NEVER treat the listed items as the only societal factors that
  matter; if the context is sparse, use the framework plus what you know to fill the gaps
  (and say so).
- EXPLICITLY FORBIDDEN: restating well-known or already-reported AI risks (e.g. "models
  may hallucinate", "AI could cause job loss", "deepfakes threaten elections", "alignment
  is hard"). If a risk would not surprise an informed reader, discard it.
- GROUND every risk in the provided signals. In "derived_from", cite the ACTUAL topics,
  papers (by title/arxiv_id), or divergences from the digest — so each risk is traceable,
  not hallucinated. When you reason beyond the signals, set "extrapolation" to a short
  honest note of what you assumed that the data does NOT show.

Generate {max_risks} novel risk implications (fewer is fine if you cannot find that many
genuinely novel, well-grounded ones). Return ONLY a JSON object (no markdown, no
preamble) with exactly this shape:

{{
  "risks": [
    {{
      "risk": "one-line risk statement",
      "derived_from": "the specific signals this is built on — cite actual topics / paper titles / divergences from the digest",
      "source_arxiv_ids": ["<id>", "..."],   // ids from the digest this draws on (may be empty)
      "source_topics": ["<topic label>", "..."],  // topic labels from the digest this draws on
      "domains_crossed": ["<framework domain>", "<framework domain>"],  // the seam
      "why_underdiscussed": "why this isn't in the news/literature yet",
      "mechanism": "the causal chain — why it is plausible",
      "leading_indicator": "the concrete observable that would tell us it is materializing",
      "calibration": "honest read: genuine low-probability vs. emerging, and your confidence",
      "extrapolation": "what you assumed beyond the provided signals (or 'none — grounded in the digest')"
    }}
  ]
}}

Output must be valid JSON and nothing else."""


def synthesize_foresight_gap(
    digest: dict, context: str, api_key: str | None,
    model: str = "claude-opus-4-8", max_risks: int = 4,
) -> dict | None:
    """Run the foresight pass. Returns the analysis["foresight_gap"] block or None.

    Fail-soft (no key / no SDK / any API or parse error -> None). The returned wrapper
    bakes in the digest, framework, and context that were used, so the dashboard can
    show the full reasoning trail without recomputation.
    """
    payload = {
        "SIGNAL_DIGEST": {k: v for k, v in digest.items() if k != "what_changed_this_week"},
        "WHAT_CHANGED_THIS_WEEK": digest.get("what_changed_this_week", {}),
        "SOCIETAL_CONTEXT": context or "(none provided — reason across the full scanning "
        "framework and current real-world state you know of)",
    }
    text = llm.call_claude(
        SYSTEM,
        _instructions(max_risks) + "\n\nINPUTS:\n" + json.dumps(payload, ensure_ascii=False),
        api_key, model,
    )
    if text is None:
        return None
    result = llm.extract_json(text)
    if result is None or "risks" not in result:
        log.warning("Could not parse foresight JSON")
        return None
    risks = result.get("risks") or []
    log.info("Foresight synthesis complete (%d risks)", len(risks))
    return {
        "risks": risks,
        "digest": digest,
        "framework": SCANNING_FRAMEWORK,
        "context": context,
        "n_context_chars": len(context or ""),
    }

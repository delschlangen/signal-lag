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
- ANCHOR ON THE TOOL'S PROPRIETARY SIGNAL, THEN CROSS IT. This tool's distinctive edge
  is its own research-trend data — a safety subfield whose velocity is DECELERATING or
  whose CRITICAL share is RISING. That is something no outside commentator has. Weight
  risks that are anchored in such a research-trend signal MORE heavily: the research
  signal is what makes a risk genuinely novel. THEN cross it with the societal context to
  make it a real-world risk. (The research signal makes it novel; the societal cross makes
  it matter.) Do NOT downweight the societal layer — you need both; just lead with the
  research-trend anchor.
- THE VALUE IS IN THE CROSS-SILO SEAM. The highest-value risks are ones where TWO DISTINCT
  EXPERT COMMUNITIES each track one half of the problem and NOBODY is connecting them
  (e.g. medical-AI deployment teams who don't follow safety-research trends; faithfulness
  researchers who don't think about compliance incentives). For every risk, name in
  "communities" who already sees each half and why they are not connecting them. A risk
  that lives entirely inside one community is probably already covered there — discard it.
- REWARD FRAMING INVERSIONS. Where possible, look for risks that INVERT or COMPLICATE the
  conventional framing of a trend everyone treats as straightforwardly good or bad (e.g.
  "transparency regulation is a solution" → "it can freeze an unsound standard into law").
  These inversions tend to be the genuinely unsurfaced ones.
- PENALIZE CONFIDENT CLAIMS ON CONTESTED GROUND. If a risk depends on a claim that is
  itself disputed, unverified, or thinly evidenced — INCLUDING when you lean on this
  tool's own research-trend metric as a CAUSAL claim ("less research attention ⇒ less
  deployed safety scaffolding") rather than as a mere attention signal — the "calibration"
  MUST state that uncertainty explicitly and lower confidence. NEVER launder a contested
  or inferential claim into a confident risk.
- The SOCIETAL_CONTEXT file and any examples in it are ILLUSTRATIVE AND NON-EXHAUSTIVE.
  Reason across the FULL scanning framework and your own knowledge of the current
  real-world state. NEVER treat the listed items as the only societal factors that matter.
- EXPLICITLY FORBIDDEN: restating well-known or already-reported AI risks (e.g. "models
  may hallucinate", "AI could cause job loss", "deepfakes threaten elections", "alignment
  is hard"). If a risk would not surprise an informed reader, discard it.
- GROUND every risk in the provided signals. In "derived_from", cite the ACTUAL topics,
  papers (by title/arxiv_id), or divergences from the digest. In "extrapolation", give an
  honest note of what you assumed that the data does NOT show.

Generate {max_risks} novel risk implications (fewer is fine if you cannot find that many
genuinely novel, well-grounded ones). Return ONLY a JSON object (no markdown, no
preamble) with exactly this shape:

{{
  "risks": [
    {{
      "risk": "one-line risk statement",
      "research_anchor": "the specific research-trend signal this is anchored on (a decelerating velocity or rising critical-share topic from the digest) — or 'none' if it is anchored mainly in societal context",
      "derived_from": "the specific signals this is built on — cite actual topics / paper titles / divergences from the digest",
      "source_arxiv_ids": ["<id>", "..."],   // ids from the digest this draws on (may be empty)
      "source_topics": ["<topic label>", "..."],  // topic labels from the digest this draws on
      "domains_crossed": ["<framework domain>", "<framework domain>"],  // the seam
      "communities": "which two (or more) expert communities each see one half, and why they are not connecting them",
      "framing_inversion": "if this inverts a conventional 'this trend is good/bad' framing, state the inversion — else 'n/a'",
      "why_underdiscussed": "why this isn't in the news/literature yet",
      "mechanism": "the causal chain — why it is plausible",
      "leading_indicator": "the concrete observable that would tell us it is materializing",
      "calibration": "honest read: genuine low-probability vs. emerging, and your confidence — explicitly lowered where the risk leans on a contested/inferential claim",
      "extrapolation": "what you assumed beyond the provided signals (or 'none — grounded in the digest')"
    }}
  ]
}}

Output must be valid JSON and nothing else."""


VERIFY_SYSTEM = (
    "You are a skeptical fact-checking research analyst. Given a candidate foresight "
    "risk, you SEARCH THE WEB to determine two things: (1) whether this risk — or a "
    "close version of it — is already being publicly discussed (news, research, "
    "commentary), and (2) whether any claim the risk depends on is itself disputed, "
    "unverified, or contested. You deliberately search for BOTH confirming AND "
    "disputing coverage. Your job is to stop an over-claimed 'novel' risk from being "
    "briefed as novel when it is already front-page, and to surface disputes that "
    "undercut a confident-sounding claim. Be honest and calibrated, not generous."
)

VERIFY_INSTRUCTIONS = """\
Verify the candidate risk below. Run a few targeted web searches:
- Is this specific risk / the specific cross-domain seam already being written about?
  Search for the risk's core claim and its named components.
- Does the risk rest on any claim that is contested, disputed, or thinly evidenced?
  Actively search for sources that DISPUTE or complicate it (e.g. "<claim> disputed",
  "<claim> skeptics", "<claim> debunked"), not only sources that confirm it.

Then return ONLY a JSON object (no markdown, no preamble) with exactly this shape:

{
  "prior_coverage": "1-3 sentences: what already exists publicly on this risk/seam, honestly stated",
  "sources": [{"title": "...", "url": "..."}],   // real URLs you found (may be empty)
  "disputed_claims": "any claim the risk depends on that is contested/unverified, and WHO disputes it — or 'none found'",
  "novelty_rating": "genuinely_unsurfaced | partially_anticipated | already_widely_discussed",
  "recommended_action": "surface | flag | drop",
  "recalibrated_calibration": "a revised, honest calibration that reflects the prior coverage AND any dispute you found"
}

Rules:
- "genuinely_unsurfaced": you found little/nothing publicly connecting these elements.
- "partially_anticipated": the components exist in public discussion but the specific seam is fresh.
- "already_widely_discussed": this risk (or a close version) is already prominent — recommend "drop" or "flag".
- If the risk depends on a contested claim, the recalibrated_calibration MUST say so explicitly and lower confidence; never launder a disputed claim into a confident risk.
Output must be valid JSON and nothing else."""


def verify_novelty(
    risk: dict, api_key: str | None, model: str = "claude-opus-4-8",
    tool_version: str = "web_search_20260209",
) -> dict | None:
    """Web-search a single candidate risk for prior coverage + disputes.

    Returns a dict (prior_coverage, sources, disputed_claims, novelty_rating,
    recommended_action, recalibrated_calibration) or None on any failure (fail-soft).
    Uses Claude's server-side web search via the shared client; tries the current tool
    version and falls back to the older one if the account/platform lacks it.
    """
    risk_blob = json.dumps({
        "risk": risk.get("risk"),
        "derived_from": risk.get("derived_from"),
        "mechanism": risk.get("mechanism"),
        "domains_crossed": risk.get("domains_crossed"),
        "calibration": risk.get("calibration"),
    }, ensure_ascii=False)
    user = VERIFY_INSTRUCTIONS + "\n\nCANDIDATE RISK:\n" + risk_blob
    for tv in (tool_version, "web_search_20250305"):
        text = llm.call_claude(
            VERIFY_SYSTEM, user, api_key, model,
            tools=[{"type": tv, "name": "web_search"}],
        )
        if text:
            result = llm.extract_json(text)
            if result:
                return result
        # else try the fallback tool version
    log.warning("Novelty verification failed for risk: %s", (risk.get("risk") or "")[:60])
    return None


# Genuinely-novel first; already-discussed demoted to the bottom (unverified in between).
NOVELTY_RANK = {
    "genuinely_unsurfaced": 0,
    "partially_anticipated": 1,
    None: 2,                      # verification failed / not run
    "already_widely_discussed": 3,
}


def verify_and_rank_risks(
    risks: list, api_key: str | None, model: str = "claude-opus-4-8",
    tool_version: str = "web_search_20260209", max_workers: int = 4,
) -> list:
    """Verify each risk's novelty in parallel, attach ``verification``, and sort.

    Genuinely-unsurfaced risks float to the top; already-widely-discussed ones are
    demoted to the bottom (flagged, not dropped). Fail-soft per risk: a failed
    verification leaves ``verification: None`` and the risk keeps its place in the middle.
    """
    if not risks:
        return risks
    from concurrent.futures import ThreadPoolExecutor

    def _one(r):
        return verify_novelty(r, api_key, model, tool_version)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        verifications = list(ex.map(_one, risks))
    for r, v in zip(risks, verifications):
        r["verification"] = v
    risks.sort(key=lambda r: NOVELTY_RANK.get(
        (r.get("verification") or {}).get("novelty_rating"), 2))
    return risks


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

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

    # Citation-VERIFIED cross-domain borrowing (capability papers that actually cite
    # safety work, via OpenAlex referenced_works). Positive-only: absence is inconclusive.
    cflow = snap.get("citation_flow") or {}
    citation_verified = [
        {"arxiv_id": b.get("arxiv_id"), "title": b.get("title"),
         "capability_topics": b.get("capability_topics"),
         "cites_safety": [c.get("title") for c in (b.get("cited_safety") or [])][:5],
         "n_cited_safety": b.get("n_cited_safety"),
         "cited_by_count": b.get("cited_by_count")}
        for b in (cflow.get("verified_borrowers") or [])[:12]
    ]

    # Experimental: capability→safety author migration (a leading indicator).
    amig = snap.get("author_migration") or {}
    author_migration = {
        "available": bool(amig.get("available")),
        "n_migrants": amig.get("n_migrants", 0),
        "examples": [
            {"author": m.get("author"), "entered_safety_topics": m.get("entered_safety_topics"),
             "prior_papers": m.get("prior_papers")}
            for m in (amig.get("migrants") or [])[:8]
        ],
    }

    # Harm/misuse dual-use lens: which real-world MISUSE the accelerating research could
    # enable, with the momentum of each harm vector and a few enabling papers. Lets the
    # synthesis frame risks as capability→harm enablement over 0-24 months.
    harm = snap.get("harm") or {}
    harm_vectors = [
        {"harm_vector": v.get("label"), "trend_pct_per_qtr": v.get("change_pct"),
         "recent_per_qtr": v.get("recent_per_qtr"), "n_tagged": v.get("n_tagged"),
         "direction": v.get("direction"),
         "enabling_papers": [rp.get("title") for rp in (v.get("rep_papers") or [])[:3]]}
        for v in (harm.get("vectors") or [])
        if v.get("n_tagged", 0) >= 3
    ][:10]

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
        "citation_verified_borrowing": citation_verified,
        "harm_vectors_dual_use": harm_vectors,
        "author_migration_experimental": author_migration,
        "new_emergent_clusters": new_clusters,
        "recent_lab_activity": lab_rows,
        "what_changed_this_week": changes,
    }


def build_weekly_digest(window_days: int, counts_by_topic: dict, notable_papers: list,
                        snapshot: dict, diff: dict) -> dict:
    """Digest for the 'this week only' foresight pass.

    Focuses on the specific papers that landed in the last ``window_days``, with a compact
    quarterly backdrop (top lagging divergences + rising critical share) so the synthesis
    can still anchor on the research-trend signal before crossing it with societal context.
    """
    lm = snapshot.get("label_map", {})

    def L(k):
        return lm.get(k, k)

    flagged = [d for d in snapshot.get("divergence", []) if d.get("lagging")]
    backdrop_div = [
        {"pairing": d["pairing"], "capability": L(d["capability_topic"]),
         "safety": L(d["safety_topic"]),
         "cap_growth_pct_per_qtr": round(d["cap_growth"] * 100, 1),
         "saf_growth_pct_per_qtr": round(d["saf_growth"] * 100, 1)}
        for d in sorted(flagged, key=lambda d: d["gap"], reverse=True)[:3]
    ]
    sent = snapshot.get("sentiment", {}) or {}
    backdrop_rising = [
        {"topic": L(k), "recent_critical_share_pct": round(v.get("recent_share", 0) * 100, 1),
         "trend_pts": round(v.get("trend", 0) * 100, 1)}
        for k, v in sent.items() if v.get("rising")
    ][:5]
    changes = {
        "first_run": diff.get("first_run", False),
        "prev_date": diff.get("prev_date"),
        "new_safety_lag_alerts": [
            {"capability": L(a["capability_topic"]), "safety": L(a["safety_topic"])}
            for a in diff.get("new_alerts", [])
        ],
        "new_accelerations": [
            {"topic": L(a["topic_key"]), "change_pct": round(a["change"] * 100, 1)}
            for a in diff.get("new_accelerations", [])
        ],
    }
    return {
        "window": f"last {window_days} days",
        "this_week_paper_counts_by_topic": counts_by_topic,
        "notable_this_week_papers": notable_papers,
        "quarterly_backdrop": {
            "top_divergences_safety_lagging": backdrop_div,
            "rising_critical_share": backdrop_rising,
        },
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
- USE CITATION-VERIFIED BORROWING AS EVIDENCE, NOT VOCABULARY. SIGNAL_DIGEST may include
  "citation_verified_borrowing": capability/applied papers that ACTUALLY CITE core safety
  work (verified via real citation references, not shared keywords). Treat a listed borrowing as
  STRONG evidence that the cross-silo link is real — prefer anchoring cross-domain risks on
  these. CRUCIAL: absence from this list is INCONCLUSIVE, never proof of "no borrowing"
  (the cited work may sit outside our sample). Never claim a community "does not cite" or
  "ignores" another based on absence here.
- "author_migration_experimental" (capability→safety author movement) is an EXPERIMENTAL,
  NOISY signal off a sampled corpus. You may use it as soft corroboration of where talent
  is flowing, but NEVER let a risk rest on it alone, and say so in "calibration" if you
  lean on it.
- FRAME HARMS AS 0–24 MONTH ENABLEMENT. SIGNAL_DIGEST includes "harm_vectors_dual_use":
  real-world MISUSE categories (cyber-offense, bio/chem uplift, influence ops, scams, agentic
  misuse, etc.) with the momentum of the research that could enable them and a few enabling
  papers. Where a harm vector is accelerating, prefer surfacing risks framed as "accelerating
  capability/technique X plausibly enables misuse Y on a 0–24 month horizon" — name a CONCRETE
  LEADING INDICATOR that would show it materializing, name which defender/community is NOT
  watching that seam, and keep calibration honest about the enablement inference (a research
  trend is an *enabling* signal, not proof of imminent abuse).
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
      "extrapolation": "what you assumed beyond the provided signals (or 'none — grounded in the digest')",
      "severity": 1-5,            // how BAD if it materializes (5 = catastrophic / systemic / irreversible)
      "likelihood": 1-5,         // probability it MATERIALIZES within ~24 months (5 = likely)
      "exposure": 1-5,           // BREADTH if it does — how many users / systems / surfaces / sectors (5 = broad)
      "trajectory": "accelerating | steady | decelerating",  // is the ENABLING signal getting worse, flat, or fading
      "score_rationale": "1-2 sentences justifying the four scores, grounded in the digest"
    }}
  ]
}}

SCORING RUBRIC (fill severity/likelihood/exposure as integers 1-5; be calibrated, not
alarmist — a genuinely low-probability risk should score low on likelihood even if severe):
- severity: 1 trivial · 2 minor · 3 serious · 4 severe · 5 catastrophic/systemic.
- likelihood: 1 remote · 2 unlikely · 3 plausible · 4 probable · 5 likely (over ~24 months).
  LOWER this when the risk leans on a contested/inferential claim (consistent with calibration).
- exposure: 1 niche · 3 a significant population/sector · 5 broad cross-sector/cross-product.
- trajectory: read it off the digest's velocity/harm-vector momentum for the anchoring signal.
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


SURFACED_RATINGS = ("genuinely_unsurfaced", "partially_anticipated")


def _surviving_count(risks: list) -> int:
    """How many risks verified as genuinely-novel or partially-anticipated."""
    return sum(
        1 for r in risks
        if (r.get("verification") or {}).get("novelty_rating") in SURFACED_RATINGS
    )


def _verify_attach(risks, api_key, model, tool_version, max_workers=4) -> list:
    """Verify each risk's novelty in parallel and attach ``verification`` (no sort)."""
    if not risks:
        return risks
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        vs = list(ex.map(lambda r: verify_novelty(r, api_key, model, tool_version), risks))
    for r, v in zip(risks, vs):
        r["verification"] = v
    return risks


def _sort_by_novelty(risks: list) -> list:
    """Genuinely-novel first; already-widely-discussed demoted to the bottom."""
    risks.sort(key=lambda r: NOVELTY_RANK.get(
        (r.get("verification") or {}).get("novelty_rating"), 2))
    return risks


def verify_and_rank_risks(
    risks: list, api_key: str | None, model: str = "claude-opus-4-8",
    tool_version: str = "web_search_20260209", max_workers: int = 4,
) -> list:
    """Verify each risk's novelty in parallel, attach ``verification``, and sort."""
    return _sort_by_novelty(_verify_attach(risks, api_key, model, tool_version, max_workers))


LIVE_CONTEXT_SYSTEM = (
    "You are a research analyst preparing a SHORT, DATED real-world briefing to ground a "
    "foresight synthesis. The synthesis crosses fast-moving AI research trends with the "
    "current state of the world, but its hand-maintained context file can be months stale. "
    "You SEARCH THE WEB for the CURRENT status of the specific societal/regulatory/economic "
    "developments most relevant to the flagged research topics, and report only verifiable, "
    "dated facts — correct any item that has since changed (e.g. a regulation's real current "
    "stage and dates). You do not speculate or generate risks; you supply ground truth."
)

LIVE_CONTEXT_INSTRUCTIONS = """\
Given the analyst's standing SOCIETAL_CONTEXT and the topics flagged this period, run a few \
targeted web searches and return a SHORT current-state brief (no more than ~250 words). For \
the handful of policy/market/geopolitical developments most relevant to the flagged topics, \
state the CURRENT status WITH DATES as of now, and explicitly note where this differs from \
or updates the standing context (e.g. a law's actual current stage, an enforcement date, a \
funding/market shift). Plain prose, lead each item with its date. Only verifiable facts; if \
you cannot verify something, omit it. Do NOT propose risks — this is ground truth only."""


def fetch_live_context(
    context: str, digest: dict, api_key: str | None, model: str = "claude-opus-4-8",
    tool_version: str = "web_search_20260209",
) -> str | None:
    """One web-search pass that returns a short, dated 'current real-world brief'.

    Crossing fast arXiv data with a static ``context.md`` is the staleness risk (#3): the
    synthesis can anchor on out-of-date policy/market facts. This pre-synthesis brief pulls
    the CURRENT status (with dates) of the developments most relevant to the flagged topics,
    so the synthesis verifies any date/policy claim against live ground truth. Complements,
    never replaces, the hand-curated context. Fail-soft (-> None); tries the current tool
    version then the older one.
    """
    # The flagged topics that most need current real-world grounding.
    flagged_topics = sorted({
        d.get("safety") for d in digest.get("divergences_safety_lagging", [])
    } | {
        r.get("topic") for r in digest.get("eroding_confidence_rising_critical_share", [])
    } | set(digest.get("quadrant", {}).get("emerging", [])))
    flagged_topics = [t for t in flagged_topics if t]
    payload = {
        "FLAGGED_TOPICS": flagged_topics,
        "SOCIETAL_CONTEXT": context or "(none provided)",
    }
    user = LIVE_CONTEXT_INSTRUCTIONS + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
    for tv in (tool_version, "web_search_20250305"):
        text = llm.call_claude(
            LIVE_CONTEXT_SYSTEM, user, api_key, model,
            tools=[{"type": tv, "name": "web_search"}],
        )
        if text and text.strip():
            log.info("Live web context brief fetched (%d chars)", len(text))
            return text.strip()
    log.warning("Live web context fetch failed (fail-soft)")
    return None


INCIDENTS_SYSTEM = (
    "You are an all-source intelligence analyst compiling a register of REAL, already-occurred "
    "AI misuse/harm incidents. You SEARCH THE WEB (AI Incident Database, OECD AI Incidents "
    "Monitor, reputable news) and return only VERIFIABLE, DATED incidents with a real source "
    "URL — never invented or hypothetical ones. Each incident is categorized into one of the "
    "provided harm-vector keys. You are precise and conservative: if you cannot verify an "
    "incident with a real source and date, you omit it."
)


def _incidents_instructions(harm_vectors: list, max_incidents: int, today: str = "") -> str:
    keys = "\n".join(f"  - {v.get('key')}: {v.get('label')}" for v in harm_vectors)
    now_line = (f"TODAY IS {today}. " if today else "")
    year = (today[:4] if today else "")
    recency = (
        f"{now_line}PRIORITIZE THE MOST RECENT incidents — especially ones from {year} and the "
        f"last ~6 months — and SEARCH SPECIFICALLY for them (e.g. '{year} AI incident', "
        f"'{year} AI misuse', recent AI Incident Database entries). Do NOT just return the "
        f"well-known landmark cases from prior years; lead with the freshest verifiable ones, "
        f"and include older landmark incidents only to fill out the list. Note that incident "
        f"databases lag, so very recent items may be sparse — that's fine, return what is real."
        if year else
        "PRIORITIZE the most recent incidents (last ~6 months) and search specifically for them."
    )
    return f"""\
Search the web and compile up to {max_incidents} REAL AI misuse/harm INCIDENTS THAT ACTUALLY \
HAPPENED — drawn from the AI Incident Database, the OECD AI Incidents Monitor, and reputable \
news. {recency}

Categorize each into exactly one HARM_VECTOR key from this list (pick the closest; skip \
incidents that fit none):
{keys}

Return ONLY a JSON object (no markdown) of this exact shape:

{{
  "incidents": [
    {{
      "title": "short incident title",
      "date": "YYYY-MM or YYYY-MM-DD (the incident's date)",
      "harm_key": "<one key from the list>",
      "summary": "1-2 sentences: what happened",
      "deployer": "the product/actor involved, if known (else '')",
      "source_url": "a real URL to a report/article"
    }}
  ]
}}

Rules: ONLY real incidents you can source with a real URL and date — NEVER hypothetical, \
predicted, or unverifiable ones. If unsure, omit. Output valid JSON only."""


def fetch_incidents(
    harm_vectors: list, api_key: str | None, model: str = "claude-opus-4-8",
    tool_version: str = "web_search_20260209", max_incidents: int = 20, today: str = "",
) -> list | None:
    """Web-search pass that returns REAL recent AI-misuse incidents, tagged to harm vectors.

    The "lagging" / all-source half of the tool: actual incidents that have occurred, to
    cross against the upstream research-enablement signal. Gathered via Claude's server-side
    web search (the one external-data path that works in CI), constrained to verifiable,
    dated, sourced incidents categorized into our harm-vector keys. ``today`` anchors the
    search on the current date and forces recency (else the model defaults to prior-year
    landmark cases). Returns incidents sorted newest-first. Fail-soft (-> None).
    """
    if not harm_vectors:
        return None
    valid_keys = {v.get("key") for v in harm_vectors}
    user = _incidents_instructions(harm_vectors, max_incidents, today)
    for tv in (tool_version, "web_search_20250305"):
        text = llm.call_claude(
            INCIDENTS_SYSTEM, user, api_key, model,
            tools=[{"type": tv, "name": "web_search"}],
        )
        if not text:
            continue
        result = llm.extract_json(text)
        if result and "incidents" in result:
            incidents = [
                r for r in (result.get("incidents") or [])
                if r.get("harm_key") in valid_keys and r.get("source_url") and r.get("date")
            ]
            incidents.sort(key=lambda r: r.get("date") or "", reverse=True)  # newest first
            log.info("Incident fetch: %d verifiable incidents tagged to harm vectors",
                     len(incidents))
            return incidents
    log.warning("Incident fetch failed (fail-soft)")
    return None


def _synthesize_risks(
    digest: dict, context: str, api_key: str | None, model: str, max_risks: int,
    avoid_seams: list | None = None, lens: str = "", live_context: str | None = None,
) -> list | None:
    """One synthesis round -> list of candidate risks (or None on failure)."""
    payload = {
        "SIGNAL_DIGEST": {k: v for k, v in digest.items() if k != "what_changed_this_week"},
        "WHAT_CHANGED_THIS_WEEK": digest.get("what_changed_this_week", {}),
        "SOCIETAL_CONTEXT": context or "(none provided — reason across the full scanning "
        "framework and current real-world state you know of)",
    }
    if live_context:
        payload["LIVE_WEB_BRIEF"] = (
            "Current, web-verified real-world status (as of this refresh) — VERIFY any "
            "date/policy/market claim against THIS, and prefer it over the standing "
            "SOCIETAL_CONTEXT wherever they conflict:\n" + live_context
        )
    user = _instructions(max_risks)
    if lens:
        user += "\n\nLENS FOR THIS PASS: " + lens
    if avoid_seams:
        avoid = "\n".join(f"- {s}" for s in avoid_seams)
        user += ("\n\nALREADY CONSIDERED THIS WEEK — these seams have already been "
                 "generated (some were found to be already widely discussed). Produce "
                 "DIFFERENT, fresh seams; do NOT repeat or lightly reword any of these:\n"
                 + avoid)
    user += "\n\nINPUTS:\n" + json.dumps(payload, ensure_ascii=False)
    text = llm.call_claude(SYSTEM, user, api_key, model)
    if text is None:
        return None
    result = llm.extract_json(text)
    if result is None or "risks" not in result:
        log.warning("Could not parse foresight JSON")
        return None
    return _attach_scores(result.get("risks") or [])


def _coerce_score(v, default: int = 3) -> int:
    """Clamp a model-provided score to an integer in [1, 5] (default on garbage)."""
    try:
        return max(1, min(5, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def _attach_scores(risks: list) -> list:
    """Normalize severity/likelihood/exposure (1-5) + trajectory, and compute priority.

    priority = severity × likelihood (1-25) — the standard risk-matrix product, used to
    rank the evergreen register. Defaults (3) keep older/garbled outputs scoreable.
    """
    for r in risks or []:
        sev = _coerce_score(r.get("severity"))
        lik = _coerce_score(r.get("likelihood"))
        exp = _coerce_score(r.get("exposure"))
        r["severity"], r["likelihood"], r["exposure"] = sev, lik, exp
        traj = str(r.get("trajectory") or "steady").lower()
        r["trajectory"] = traj if traj in ("accelerating", "steady", "decelerating") else "steady"
        r["priority"] = sev * lik
    return risks


def synthesize_foresight_gap(
    digest: dict, context: str, api_key: str | None,
    model: str = "claude-opus-4-8", max_risks: int = 4, live_context: str | None = None,
) -> dict | None:
    """Single-round synthesis (no verification). Returns the foresight_gap block or None.

    Kept for the preview script; the production path uses ``run_foresight`` which adds
    verification and quality-driven backfill.
    """
    risks = _synthesize_risks(digest, context, api_key, model, max_risks,
                              live_context=live_context)
    if risks is None:
        return None
    log.info("Foresight synthesis complete (%d risks)", len(risks))
    return {
        "risks": risks,
        "digest": digest,
        "framework": SCANNING_FRAMEWORK,
        "context": context,
        "n_context_chars": len(context or ""),
        "live_context": live_context,
    }


def run_foresight(
    digest: dict, context: str, api_key: str | None, model: str = "claude-opus-4-8",
    max_risks: int = 4, verify: bool = True,
    tool_version: str = "web_search_20260209",
    min_surfaced: int = 3, max_rounds: int = 3, lens: str = "",
    live_context: str | None = None,
) -> dict | None:
    """Full foresight pass: synthesize -> verify -> backfill until enough survive.

    Quality over quantity: if too few risks survive verification as genuinely-novel /
    partially-anticipated (too many came back already-widely-discussed), it runs another
    synthesis round asking for DIFFERENT seams and verifies those too — up to
    ``max_rounds``. Stops early once ``min_surfaced`` survive or a round adds nothing.
    Without verification it's a single round. Fail-soft (-> None on first-round failure).
    ``lens`` optionally focuses the synthesis (e.g. on just this week's papers).
    """
    risks = _synthesize_risks(digest, context, api_key, model, max_risks, lens=lens,
                              live_context=live_context)
    if risks is None:
        return None

    rounds = 1
    if verify:
        _verify_attach(risks, api_key, model, tool_version)
        while _surviving_count(risks) < min_surfaced and rounds < max_rounds:
            need = max(2, min_surfaced - _surviving_count(risks))
            avoid = [r.get("risk", "") for r in risks]
            more = _synthesize_risks(digest, context, api_key, model, need,
                                     avoid_seams=avoid, lens=lens,
                                     live_context=live_context)
            if not more:
                break
            _verify_attach(more, api_key, model, tool_version)
            risks.extend(more)
            rounds += 1
            log.info("Foresight backfill round %d: %d surviving of %d",
                     rounds, _surviving_count(risks), len(risks))
        _sort_by_novelty(risks)

    return {
        "risks": risks,
        "digest": digest,
        "framework": SCANNING_FRAMEWORK,
        "context": context,
        "n_context_chars": len(context or ""),
        "verified": bool(verify),
        "rounds": rounds,
        "n_surfaced": _surviving_count(risks) if verify else None,
        "live_context": live_context,
    }


SCENARIO_SYSTEM = (
    "You are a strategic-foresight analyst running SCENARIO ANALYSIS for an AI emerging-risks "
    "team. Given a small set of already-surfaced, scored risks, you develop a few concrete, "
    "plausible scenarios for how the situation could evolve over the next 6-24 months. You are "
    "not predicting — you are mapping the possibility space so an analyst can pre-position. "
    "Each scenario is specific and decision-relevant: it names the drivers, the early "
    "observable indicators, the branch points where the future forks, and candidate "
    "mitigations. You stay grounded in the provided risks and are honest about uncertainty."
)


def _scenario_instructions(max_scenarios: int) -> str:
    return f"""\
You are given TOP_RISKS (already surfaced and scored by severity/likelihood/exposure) and the \
current real-world CONTEXT. Develop {max_scenarios} distinct, plausible SCENARIOS for how these \
risks could evolve over the next 6-24 months. Prefer scenarios that cut across more than one \
risk, and vary the horizon/severity across the set (not all worst-case). Return ONLY a JSON \
object (no markdown) of this exact shape:

{{
  "scenarios": [
    {{
      "title": "short scenario name",
      "horizon": "6 months | 12 months | 24 months",
      "estimative_likelihood": "very unlikely | unlikely | roughly even | likely | very likely",
      "narrative": "2-4 sentences: how it plausibly unfolds",
      "drivers": ["the forces pushing it", "..."],
      "leading_indicators": ["concrete, observable early-warning signs to watch for", "..."],
      "branch_points": ["the decisions/uncertainties where the outcome forks", "..."],
      "candidate_mitigations": ["actions that would reduce severity or likelihood", "..."],
      "linked_risks": ["the TOP_RISKS statement(s) this builds on"]
    }}
  ]
}}

Use estimative-probability language (ICD-203 style) for "estimative_likelihood". Ground every \
scenario in the provided risks/context; do not invent unrelated threats. Output valid JSON only."""


def generate_scenarios(
    risks: list, context: str, api_key: str | None, model: str = "claude-opus-4-8",
    max_scenarios: int = 3, top_k: int = 6, live_context: str | None = None,
) -> list | None:
    """One Claude pass: develop 6-24 month scenarios from the top-priority risks.

    Takes the highest-priority surfaced risks (already scored) and returns structured
    scenarios (narrative, drivers, leading indicators, branch points, mitigations,
    estimative likelihood). Fail-soft (-> None). One call; cheap relative to synthesis.
    """
    if not risks:
        return None
    top = sorted(risks, key=lambda r: r.get("priority") or 0, reverse=True)[:top_k]
    payload = {
        "TOP_RISKS": [
            {"risk": r.get("risk"), "severity": r.get("severity"),
             "likelihood": r.get("likelihood"), "exposure": r.get("exposure"),
             "trajectory": r.get("trajectory"), "mechanism": r.get("mechanism"),
             "leading_indicator": r.get("leading_indicator"),
             "domains_crossed": r.get("domains_crossed")}
            for r in top
        ],
        "CONTEXT": (live_context or context or "(none)")[:4000],
    }
    user = _scenario_instructions(max_scenarios) + "\n\nINPUTS:\n" + json.dumps(
        payload, ensure_ascii=False)
    text = llm.call_claude(SCENARIO_SYSTEM, user, api_key, model)
    if text is None:
        return None
    result = llm.extract_json(text)
    if not result or "scenarios" not in result:
        log.warning("Could not parse scenarios JSON")
        return None
    log.info("Scenario analysis complete (%d scenarios)", len(result.get("scenarios") or []))
    return result.get("scenarios") or []


EXPLAINER_SYSTEM = (
    "You translate a technical AI foresight risk into a PLAIN-LANGUAGE walkthrough for a "
    "smart non-specialist or executive. You explain the evidence chain in concrete terms: "
    "the technical evidence (the ACTUAL papers and the trend metric), the real-world context "
    "it was crossed with, the synthesis (why the gap creates the risk), and — crucially — "
    "the tool's OWN skepticism (where the evidence is contested, brittle, or a projection). "
    "You end with a plain bottom line that separates what is REAL/observed from what is "
    "PROJECTED. You are concrete, honest, and never overstate; you name papers by title and "
    "arXiv id where given, and you never invent evidence."
)

EXPLAINER_INSTRUCTIONS = """\
Translate the RISK below into a plain-language explanation a non-specialist could follow, \
using the SOURCE_PAPERS and CONTEXT provided. Return ONLY a JSON object of this exact shape:

{
  "technical_evidence": "What the actual papers/metrics show, in plain terms — reference the source papers by title and arXiv id, and cite the trend/safety-lag metric if present.",
  "societal_evidence": "The real-world context, standards, or developments this was crossed with.",
  "the_gap": "The synthesis: why the friction between the technical evidence and the real-world context creates the risk.",
  "skepticism": "The tool's OWN counter-evidence — where the capability is contested/brittle, or where it's a projection rather than an imminent fact.",
  "bottom_line": "1-2 plain sentences separating what is REAL/observed this period from what is PROJECTED."
}

Ground everything in the provided RISK fields and SOURCE_PAPERS — do not invent papers or \
facts. Output valid JSON only."""


def explain_risk(
    risk: dict, paper_lookup: dict, context: str, api_key: str | None,
    model: str = "claude-opus-4-8",
) -> dict | None:
    """One Claude call: a plain-language, 5-part walkthrough of a single risk.

    Sections: technical_evidence / societal_evidence / the_gap / skepticism / bottom_line —
    the legibility layer that explains HOW the tool reasoned (and where it doubts itself),
    for the in-app expander and the downloadable estimate. Fail-soft (-> None).
    """
    src = []
    for aid in (risk.get("source_arxiv_ids") or [])[:6]:
        p = paper_lookup.get(aid)
        if p:
            src.append({"arxiv_id": aid, "title": p.get("title"),
                        "abstract": (p.get("abstract") or "")[:400]})
    payload = {
        "RISK": {k: risk.get(k) for k in (
            "risk", "research_anchor", "derived_from", "source_topics", "domains_crossed",
            "communities", "mechanism", "leading_indicator", "calibration", "extrapolation",
            "severity", "likelihood", "exposure", "trajectory")},
        "VERIFICATION": risk.get("verification") or {},
        "SOURCE_PAPERS": src,
        "CONTEXT": (context or "(none)")[:2500],
    }
    user = EXPLAINER_INSTRUCTIONS + "\n\nINPUTS:\n" + json.dumps(payload, ensure_ascii=False)
    text = llm.call_claude(EXPLAINER_SYSTEM, user, api_key, model)
    if text is None:
        return None
    return llm.extract_json(text)


def attach_explanations(
    risks: list, paper_lookup: dict, context: str, api_key: str | None,
    model: str = "claude-opus-4-8", max_explainers: int = 4, max_workers: int = 4,
) -> list:
    """Attach a plain-language ``plain_explanation`` to the top-N priority risks (in parallel).

    Bounded to ``max_explainers`` (cost guard) and only the highest-priority risks — the ones
    a reader actually drills into. Fail-soft: a risk with no explanation just lacks the field.
    """
    if not risks or not api_key:
        return risks
    top = sorted(risks, key=lambda r: r.get("priority") or 0, reverse=True)[:max_explainers]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        outs = list(ex.map(
            lambda r: explain_risk(r, paper_lookup, context, api_key, model), top))
    for r, o in zip(top, outs):
        if o:
            r["plain_explanation"] = o
    log.info("Attached plain-language explanations to %d/%d risks",
             sum(1 for r in top if r.get("plain_explanation")), len(top))
    return risks

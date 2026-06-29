"""Author/institution flow: which labs are growing activity in which subfields.

A talent-flow leading indicator: rising institutional output in a topic often
precedes a wave of work there.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict

import pandas as pd

log = logging.getLogger("signal_lag.authors")


def institution_topic_trends(
    papers, tags: dict[str, list[tuple[str, float]]], window_periods: int = 2
) -> pd.DataFrame:
    """Per (institution, topic) recent vs prior paper counts and growth."""
    by_id = {p.arxiv_id: p for p in papers}
    rows = []
    for aid, taglist in tags.items():
        p = by_id.get(aid)
        if p is None or not p.institutions:
            continue
        per = pd.Period(p.published, freq="Q")
        for topic_key, _ in taglist:
            for inst in p.institutions:
                rows.append((inst, topic_key, per))
    if not rows:
        return pd.DataFrame(
            columns=["institution", "topic_key", "recent", "prior", "growth"]
        )
    df = pd.DataFrame(rows, columns=["institution", "topic_key", "period"])
    periods = pd.period_range(df["period"].min(), df["period"].max(), freq="Q")
    recent_cut = periods[-window_periods] if len(periods) >= window_periods else periods[0]
    prior_cut = (
        periods[-2 * window_periods] if len(periods) >= 2 * window_periods else periods[0]
    )

    agg = []
    for (inst, topic), g in df.groupby(["institution", "topic_key"]):
        recent = int((g["period"] >= recent_cut).sum())
        prior = int(((g["period"] >= prior_cut) & (g["period"] < recent_cut)).sum())
        growth = (recent - prior) / (prior if prior > 0 else 1.0)
        agg.append(
            {
                "institution": inst,
                "topic_key": topic,
                "recent": recent,
                "prior": prior,
                "growth": round(growth, 3),
                "total": len(g),
            }
        )
    out = pd.DataFrame(agg)
    return out.sort_values(["growth", "recent"], ascending=False).reset_index(drop=True)


def author_migration(
    papers, tax_tags, taxonomy, min_history: int = 2,
    recent_window_periods: int = 2, max_examples: int = 25,
) -> dict:
    """EXPERIMENTAL leading indicator: capability→safety author movement.

    Using the OpenAlex author ids captured during enrichment, build each author's
    quarterly trajectory of capability vs safety topic membership and flag authors who
    were capability-dominant historically (>= ``min_history`` prior papers, none of them
    safety) and whose *recent* papers (last ``recent_window_periods`` quarters) enter a
    safety/oversight topic. Such migration can precede a wave of safety work.

    NOISY BY CONSTRUCTION: the corpus is a temporally-stratified sample (150/cat/quarter),
    so per-author history is sparse and author ids are imperfect. This INFORMS the digest,
    is clearly labeled experimental, and never gates an alert. Returns a dict with
    ``migrants`` (examples) + counts; empty when author ids are unavailable. Fail-soft.
    """
    safety_keys = {t.key for t in taxonomy.safety_topics}
    cap_keys = {t.key for t in taxonomy.capability_topics}
    lm = {t.key: t.label for t in taxonomy.all_topics}

    # author_id -> list of (period, has_safety, has_capability, arxiv_id, safety_labels)
    hist: dict[str, list] = defaultdict(list)
    name_of: dict[str, str] = {}
    all_periods: list = []
    for p in papers:
        tags = tax_tags.get(p.arxiv_id, [])
        saf_labels = [lm.get(k, k) for k, _ in tags if k in safety_keys]
        has_cap = any(k in cap_keys for k, _ in tags)
        if not (saf_labels or has_cap):
            continue
        per = pd.Period(p.published, freq="Q")
        all_periods.append(per)
        for a in p.authors:
            if a.openalex_id:
                hist[a.openalex_id].append((per, bool(saf_labels), has_cap, p.arxiv_id, saf_labels))
                name_of.setdefault(a.openalex_id, a.name)

    if not hist or not all_periods:
        return {"migrants": [], "n_migrants": 0, "n_authors_tracked": 0, "available": False}

    periods = pd.period_range(min(all_periods), max(all_periods), freq="Q")
    recent_cut = periods[-recent_window_periods] if len(periods) >= recent_window_periods else periods[0]

    migrants = []
    tracked = 0
    for aid, recs in hist.items():
        if len(recs) < min_history + 1:
            continue
        tracked += 1
        prior = [r for r in recs if r[0] < recent_cut]
        recent = [r for r in recs if r[0] >= recent_cut]
        if len(prior) < min_history or not recent:
            continue
        prior_has_safety = any(r[1] for r in prior)
        prior_has_cap = any(r[2] for r in prior)
        recent_safety = [r for r in recent if r[1]]
        # Migration: historically capability, never safety; now entering safety.
        if prior_has_cap and not prior_has_safety and recent_safety:
            entered = sorted({lbl for r in recent_safety for lbl in r[4]})
            migrants.append({
                "author": name_of.get(aid),
                "author_openalex_id": aid,
                "prior_papers": len(prior),
                "recent_papers": len(recent),
                "entered_safety_topics": entered,
                "recent_arxiv_ids": [r[3] for r in recent_safety][:5],
            })

    migrants.sort(key=lambda m: (len(m["entered_safety_topics"]), m["prior_papers"]), reverse=True)
    log.info("Author migration: %d capability→safety migrants of %d tracked authors",
             len(migrants), tracked)
    return {
        "migrants": migrants[:max_examples],
        "n_migrants": len(migrants),
        "n_authors_tracked": tracked,
        "available": True,
    }

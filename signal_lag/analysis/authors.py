"""Author/institution flow: which labs are growing activity in which subfields.

A talent-flow leading indicator: rising institutional output in a topic often
precedes a wave of work there.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd


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

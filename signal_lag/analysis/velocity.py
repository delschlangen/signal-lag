"""Topic/cluster velocity: submission rate over time and inflection detection."""
from __future__ import annotations

import datetime as dt

import pandas as pd


def _quarter(d: dt.date) -> pd.Period:
    return pd.Period(d, freq="Q")


def topic_timeseries(
    papers, tags: dict[str, list[tuple[str, float]]]
) -> pd.DataFrame:
    """Long-form counts per (topic_key, period).

    `tags` maps arxiv_id -> [(topic_key, score), ...]. Returns columns:
    period, topic_key, count.
    """
    rows = []
    by_id = {p.arxiv_id: p for p in papers}
    for aid, taglist in tags.items():
        p = by_id.get(aid)
        if p is None:
            continue
        per = _quarter(p.published)
        for topic_key, _score in taglist:
            rows.append((per, topic_key))
    if not rows:
        return pd.DataFrame(columns=["period", "topic_key", "count"])
    df = pd.DataFrame(rows, columns=["period", "topic_key"])
    out = df.groupby(["topic_key", "period"]).size().reset_index(name="count")
    return out.sort_values(["topic_key", "period"]).reset_index(drop=True)


def drop_incomplete_tail(ts: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Remove the current (incomplete) quarter so it can't fake a deceleration.

    A refresh almost always runs mid-quarter; that partial quarter has fewer
    papers, which would otherwise read as every topic decelerating. Trend math
    (inflection/divergence) should only see complete quarters.
    """
    if ts.empty:
        return ts
    current = pd.Period(today, freq="Q")
    return ts[ts["period"] != current].reset_index(drop=True)


def _full_period_index(ts: pd.DataFrame) -> pd.PeriodIndex:
    pmin, pmax = ts["period"].min(), ts["period"].max()
    return pd.period_range(pmin, pmax, freq="Q")


def topic_series(ts: pd.DataFrame, topic_key: str) -> pd.Series:
    """Dense per-quarter count series (zero-filled) for one topic."""
    if ts.empty:
        return pd.Series(dtype=float)
    idx = _full_period_index(ts)
    s = (
        ts[ts["topic_key"] == topic_key]
        .set_index("period")["count"]
        .reindex(idx, fill_value=0)
        .astype(float)
    )
    s.index.name = "period"
    return s


def compute_inflections(
    ts: pd.DataFrame, window: int, threshold: float
) -> list[dict]:
    """Flag acceleration/deceleration per topic.

    Compares the mean count of the last `window` periods to the prior `window`.
    """
    out = []
    for topic_key in sorted(ts["topic_key"].unique()):
        s = topic_series(ts, topic_key)
        if len(s) < 2 * window:
            continue
        recent = s.iloc[-window:].mean()
        prior = s.iloc[-2 * window : -window].mean()
        if prior <= 0 and recent <= 0:
            continue
        change = (recent - prior) / (prior if prior > 0 else 1.0)
        direction = (
            "acceleration"
            if change >= threshold
            else "deceleration"
            if change <= -threshold
            else "steady"
        )
        out.append(
            {
                "topic_key": topic_key,
                "recent_mean": round(float(recent), 2),
                "prior_mean": round(float(prior), 2),
                "change": round(float(change), 3),
                "direction": direction,
            }
        )
    return sorted(out, key=lambda d: d["change"], reverse=True)


def newly_forming(ts: pd.DataFrame, max_age_periods: int) -> list[str]:
    """Topics whose first nonzero quarter is within the last `max_age_periods`."""
    if ts.empty:
        return []
    idx = _full_period_index(ts)
    cutoff = idx[-min(max_age_periods, len(idx))]
    out = []
    for topic_key in ts["topic_key"].unique():
        first = ts[(ts["topic_key"] == topic_key) & (ts["count"] > 0)]["period"].min()
        if first is not None and first >= cutoff:
            out.append(topic_key)
    return out

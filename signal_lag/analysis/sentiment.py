"""Negative / critical-signal layer.

Volume tells you how much a field is being worked on; it does NOT tell you whether
the field is succeeding. A topic can spike because of a breakthrough OR because
everyone is hitting a wall and piling in. This module detects the second case.

Using the same embedding approach as topic tagging, each paper gets a "critical"
score = cosine similarity between its abstract embedding and a centroid built from
negativity/limitation seed phrases. Per topic we then track the *share* of critical
papers and its trend over quarters. A rising critical share — especially when
volume is flat — is an early warning that confidence in an approach may be eroding.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Taxonomy
from .embeddings import Embedder

log = logging.getLogger("signal_lag.sentiment")


def build_negativity_centroid(taxonomy: Taxonomy, embedder: Embedder) -> np.ndarray | None:
    seeds = taxonomy.negativity_seeds
    if not seeds:
        return None
    vecs = embedder.embed(seeds)
    c = np.mean(vecs, axis=0)
    norm = np.linalg.norm(c)
    return (c / norm).astype(np.float32) if norm else c.astype(np.float32)


def critical_scores(paper_vecs: np.ndarray, centroid: np.ndarray | None) -> np.ndarray:
    """Cosine similarity of each paper to the negativity centroid (0 if none)."""
    if centroid is None or len(paper_vecs) == 0:
        return np.zeros(len(paper_vecs), dtype=np.float32)
    return paper_vecs @ centroid


def topic_sentiment(
    paper_ids: list[str],
    scores: np.ndarray,
    published_periods: dict[str, pd.Period],
    tax_tags: dict[str, list[tuple[str, float]]],
    taxonomy: Taxonomy,
    cfg: dict,
) -> dict:
    """Per-topic critical share + recent-vs-prior trend.

    Returns {topic_key: {critical_share, recent_share, prior_share, trend,
    n_recent, rising}}.
    """
    thr = float(cfg.get("critical_threshold", 0.22))
    window = int(cfg.get("window", 2))
    rise_thr = float(cfg.get("rising_share_threshold", 0.08))
    min_recent = int(cfg.get("min_recent_papers", 8))

    score_by_id = {pid: float(s) for pid, s in zip(paper_ids, scores)}

    # Build per-topic rows: (period, is_critical)
    rows = []
    for pid, tags in tax_tags.items():
        per = published_periods.get(pid)
        s = score_by_id.get(pid)
        if per is None or s is None:
            continue
        crit = s >= thr
        for topic_key, _ in tags:
            rows.append((topic_key, per, crit))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["topic_key", "period", "critical"])
    periods = pd.period_range(df["period"].min(), df["period"].max(), freq="Q")
    recent_cut = periods[-window] if len(periods) >= window else periods[0]
    prior_cut = periods[-2 * window] if len(periods) >= 2 * window else periods[0]

    out: dict = {}
    for topic_key, g in df.groupby("topic_key"):
        overall = float(g["critical"].mean())
        recent = g[g["period"] >= recent_cut]
        prior = g[(g["period"] >= prior_cut) & (g["period"] < recent_cut)]
        recent_share = float(recent["critical"].mean()) if len(recent) else 0.0
        prior_share = float(prior["critical"].mean()) if len(prior) else 0.0
        trend = recent_share - prior_share
        rising = (
            trend >= rise_thr
            and len(recent) >= min_recent
            and recent_share > overall * 0.0  # guard; recent positive
        )
        out[topic_key] = {
            "critical_share": round(overall, 3),
            "recent_share": round(recent_share, 3),
            "prior_share": round(prior_share, 3),
            "trend": round(trend, 3),
            "n_recent": int(len(recent)),
            "rising": bool(rising),
        }
    return out


def sentiment_timeseries(
    paper_ids: list[str],
    scores: np.ndarray,
    published_periods: dict[str, pd.Period],
    tax_tags: dict[str, list[tuple[str, float]]],
    threshold: float,
) -> pd.DataFrame:
    """Long-form critical share per (topic_key, period) for charting."""
    score_by_id = {pid: float(s) for pid, s in zip(paper_ids, scores)}
    rows = []
    for pid, tags in tax_tags.items():
        per = published_periods.get(pid)
        s = score_by_id.get(pid)
        if per is None or s is None:
            continue
        for topic_key, _ in tags:
            rows.append((topic_key, per, 1 if s >= threshold else 0))
    if not rows:
        return pd.DataFrame(columns=["topic_key", "period", "critical_share", "n"])
    df = pd.DataFrame(rows, columns=["topic_key", "period", "crit"])
    agg = df.groupby(["topic_key", "period"]).agg(
        critical_share=("crit", "mean"), n=("crit", "size")
    ).reset_index()
    return agg.sort_values(["topic_key", "period"]).reset_index(drop=True)

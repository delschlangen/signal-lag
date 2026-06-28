"""The headline layer: capability-vs-safety velocity divergence per pairing.

For each configured pairing we compute a normalized recent velocity for the
capability topic and its paired safety topic, then the gap between them. A large
positive gap (capability accelerating, safety flat) is the signal of interest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Taxonomy
from .velocity import topic_series


def _recent_velocity(s: pd.Series, window: int) -> tuple[float, float]:
    """Return (recent_mean, growth) where growth = (recent-prior)/prior."""
    if len(s) < 2 * window:
        recent = float(s.iloc[-window:].mean()) if len(s) >= window else float(s.mean() or 0)
        return recent, 0.0
    recent = float(s.iloc[-window:].mean())
    prior = float(s.iloc[-2 * window : -window].mean())
    growth = (recent - prior) / (prior if prior > 0 else 1.0)
    return recent, growth


def compute_divergence(
    ts: pd.DataFrame, taxonomy: Taxonomy, window: int, gap_threshold: float,
    min_recent_volume: float = 3.0,
) -> list[dict]:
    out = []
    for pair in taxonomy.pairings:
        cap_s = topic_series(ts, pair.capability)
        saf_s = topic_series(ts, pair.safety)
        cap_recent, cap_growth = _recent_velocity(cap_s, window)
        saf_recent, saf_growth = _recent_velocity(saf_s, window)

        # Growth-rate gap is the primary divergence signal.
        gap = cap_growth - saf_growth
        # Ratio of recent absolute volumes (how lopsided the field is right now).
        ratio = cap_recent / saf_recent if saf_recent > 0 else float("inf")

        # Flag only when capability is meaningfully active and outpacing safety,
        # so tiny-count topics can't produce noisy alerts.
        flag = (
            gap >= gap_threshold
            and cap_growth > 0
            and cap_recent >= min_recent_volume
        )
        out.append(
            {
                "pairing": pair.name,
                "capability_topic": pair.capability,
                "safety_topic": pair.safety,
                "cap_recent": round(cap_recent, 2),
                "saf_recent": round(saf_recent, 2),
                "cap_growth": round(cap_growth, 3),
                "saf_growth": round(saf_growth, 3),
                "gap": round(gap, 3),
                "volume_ratio": round(ratio, 2) if np.isfinite(ratio) else None,
                "lagging": bool(flag),
            }
        )
    return sorted(out, key=lambda d: d["gap"], reverse=True)


def quadrant_view(inflections: list[dict], ts: pd.DataFrame) -> list[dict]:
    """Classify topics into emerging / hot / cooling / white-space.

    Axes: recent volume (x) and growth (y). White space = low volume + low growth.
    """
    out = []
    for inf in inflections:
        vol = inf["recent_mean"]
        growth = inf["change"]
        if growth >= 0.3 and vol < 5:
            quad = "emerging"
        elif growth >= 0.3:
            quad = "hot"
        elif growth <= -0.3:
            quad = "cooling"
        elif vol < 2:
            quad = "white-space"
        else:
            quad = "established"
        out.append({**inf, "quadrant": quad})
    return out

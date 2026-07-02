"""Derived signals computed over an already-built snapshot (no network, no LLM).

These are pure functions over the snapshot dict the dashboard already loads, so they
render on the current snapshot immediately (no refresh needed) and are trivially
unit-testable. Three signals:

- ``monitoring_debt``      (#3)  — cumulative capability-minus-safety backlog per pairing.
- ``weekly_momentum``      (#14) — this week's per-topic volume vs. the quarterly-baseline
                                   expectation (a Poisson-style deviation), to tell a normal
                                   week from an anomalous spike.
- ``false_confidence_alerts`` (#13) — capability rising + self-criticism falling + paired
                                   safety flat: possible premature-deployment overconfidence.
"""
from __future__ import annotations

import math

QUARTER_DAYS = 91.31  # 365.25 / 4


def _counts_by_topic(snapshot: dict) -> tuple[dict, list]:
    """{topic_key: {period: count}} and the sorted list of periods, from the timeseries."""
    ts = snapshot.get("timeseries") or []
    by_topic: dict[str, dict[str, float]] = {}
    periods = sorted({r.get("period") for r in ts if r.get("period") is not None})
    for r in ts:
        by_topic.setdefault(r.get("topic_key"), {})[r.get("period")] = r.get("count", 0)
    return by_topic, periods


def monitoring_debt(snapshot: dict) -> list[dict]:
    """Per pairing, the cumulative Σ(capability − safety) paper count over quarters (#3).

    A one-quarter gap is noise; a *rising* cumulative debt curve is persistent structural
    imbalance (capability consistently out-producing its paired safety topic). Returns one
    entry per configured pairing with aligned ``periods`` / ``debt`` arrays. Uses raw
    per-quarter counts (clearly a first-order proxy: topics with different baseline sizes
    start offset — the *slope*, not the level, is the signal).
    """
    by_topic, periods = _counts_by_topic(snapshot)
    out = []
    for d in snapshot.get("divergence") or []:
        ck, sk = d.get("capability_topic"), d.get("safety_topic")
        cser, sser = by_topic.get(ck, {}), by_topic.get(sk, {})
        cum, debt = 0.0, []
        for p in periods:
            cum += (cser.get(p, 0) or 0) - (sser.get(p, 0) or 0)
            debt.append(round(cum, 1))
        out.append({
            "pairing": d.get("pairing"), "capability_topic": ck, "safety_topic": sk,
            "periods": periods, "debt": debt,
            "rising": len(debt) >= 2 and debt[-1] > debt[-2],
            "latest": debt[-1] if debt else 0.0,
        })
    return out


def weekly_momentum(snapshot: dict, window_days: int = 7, recent_periods: int = 2) -> list[dict]:
    """This week's per-topic count vs. the quarterly-baseline expectation (#14).

    Expected weekly count = (mean papers/quarter over the last ``recent_periods`` quarters)
    scaled to the window; deviation is reported as a percentage and a Poisson z-score
    (z = (actual − expected)/√expected), so a genuine spike is distinguished from ordinary
    weekly volume. Returns rows sorted by z descending. Empty if there is no weekly block.
    """
    weekly = (snapshot.get("weekly") or {}).get("counts_by_key") or {}
    by_topic, periods = _counts_by_topic(snapshot)
    recent = periods[-recent_periods:] if periods else []
    rows = []
    for k, actual in weekly.items():
        ser = by_topic.get(k)
        if not ser or not recent:
            continue
        qmean = sum(ser.get(p, 0) or 0 for p in recent) / len(recent)
        expected = qmean * (window_days / QUARTER_DAYS)
        if expected <= 0:
            continue
        z = (actual - expected) / math.sqrt(expected)
        rows.append({
            "topic_key": k, "actual": actual, "expected": round(expected, 1),
            "pct": round((actual - expected) / expected * 100, 0), "z": round(z, 1),
        })
    rows.sort(key=lambda r: r["z"], reverse=True)
    return rows


def false_confidence_alerts(
    snapshot: dict, min_critical_drop: float = 0.02, min_recent_papers: int = 8,
) -> list[dict]:
    """Possible premature-deployment overconfidence per capability→safety pairing (#13).

    Fires when, for a pairing: capability growth is positive (volume rising), the capability
    topic's *critical* share is FALLING (less visible self-critique), and the paired safety
    topic is flat or shrinking. Falling criticism in a fast-growing field can look positive
    but may signal deployment optimism outrunning scrutiny — a distinct warning class from
    safety-lag or sentiment-erosion. ``lab_active`` (recent lab announcements on the topic)
    is reported as corroboration, not required (lab-topic tagging is sparse). Cautious by
    construction: it says *investigate*, not *confirmed*.
    """
    sent = snapshot.get("sentiment") or {}
    lab_topics = {p.get("topic") for p in (snapshot.get("lab_activity") or []) if p.get("topic")}
    out = []
    for d in snapshot.get("divergence") or []:
        ck, sk = d.get("capability_topic"), d.get("safety_topic")
        cap_growth = d.get("cap_growth") or 0
        saf_growth = d.get("saf_growth") or 0
        cs = sent.get(ck) or {}
        crit_trend = cs.get("trend") or 0
        cap_rising = cap_growth > 0
        crit_falling = crit_trend <= -min_critical_drop and cs.get("n_recent", 0) >= min_recent_papers
        safety_flat = saf_growth <= 0
        if cap_rising and crit_falling and safety_flat:
            out.append({
                "pairing": d.get("pairing"), "capability_topic": ck, "safety_topic": sk,
                "cap_growth": cap_growth, "saf_growth": saf_growth,
                "critical_trend": crit_trend, "lab_active": ck in lab_topics,
            })
    out.sort(key=lambda a: (a["critical_trend"], -a["cap_growth"]))
    return out

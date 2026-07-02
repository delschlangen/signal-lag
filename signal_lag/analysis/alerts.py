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


def register_calibration(register: list) -> dict:
    """Forecast-validation scaffold (#9): what the register's own history shows so far.

    True hit-rate / Brier calibration needs materialized-vs-invalidated outcomes over
    several quarters; the register only recently started accumulating. This computes the
    honest precursors available NOW from score history — persistence, score movement,
    dispute exposure — and reports how much history exists so the display can say plainly
    what is and isn't yet measurable.
    """
    n = len(register or [])
    if not n:
        return {"n": 0}
    reseen = [e for e in register if (e.get("n_appearances") or 1) >= 2]
    upgraded = downgraded = 0
    for e in reseen:
        h = e.get("history") or []
        if len(h) >= 2 and h[-1].get("priority") is not None and h[0].get("priority") is not None:
            if h[-1]["priority"] > h[0]["priority"]:
                upgraded += 1
            elif h[-1]["priority"] < h[0]["priority"]:
                downgraded += 1
    ever_disputed = sum(
        1 for e in register
        if e.get("counterevidence") or any((p.get("disputed") for p in e.get("history") or []))
    )
    dates = sorted({p.get("date") for e in register for p in (e.get("history") or [])
                    if p.get("date")})
    return {
        "n": n,
        "n_reseen": len(reseen),
        "n_upgraded": upgraded,
        "n_downgraded": downgraded,
        "n_ever_disputed": ever_disputed,
        "n_refreshes": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
    }


def benchmark_transitions(history_rows: list) -> dict:
    """Harm-vector early-warning calibration scaffold (#28).

    Given the per-refresh benchmark history rows ({date, key, label, quadrant,
    n_incidents}), find each vector's 'foresight lead' episodes and whether incidents
    appeared at a LATER date (a lead that materialized) — the raw material for
    time-to-incident and false-positive rates once several refreshes accumulate.
    """
    by_key: dict[str, list] = {}
    for r in history_rows or []:
        if r.get("key"):
            by_key.setdefault(r["key"], []).append(r)
    materialized, open_leads = [], []
    for key, rows in by_key.items():
        rows.sort(key=lambda r: r.get("date") or "")
        label = rows[-1].get("label") or key
        lead_date = None
        for r in rows:
            if r.get("quadrant") == "foresight lead" and lead_date is None:
                lead_date = r.get("date")
            elif lead_date and (r.get("n_incidents") or 0) > 0:
                materialized.append({"key": key, "label": label, "lead_date": lead_date,
                                     "incident_date": r.get("date")})
                lead_date = None
        if lead_date:
            open_leads.append({"key": key, "label": label, "lead_date": lead_date})
    dates = sorted({r.get("date") for r in history_rows or [] if r.get("date")})
    return {"n_refreshes": len(dates), "materialized": materialized,
            "open_leads": open_leads}


def wilson_interval(share: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (#22) — safe at small n and share≈0/1."""
    if n <= 0:
        return (0.0, 1.0)
    share = min(1.0, max(0.0, share))
    denom = 1 + z * z / n
    center = (share + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def growth_uncertainty(recent_mean: float, prior_mean: float, window: int = 2) -> float | None:
    """±1σ uncertainty (in growth points) on growth = recent/prior − 1 (#22).

    Treats per-quarter counts as Poisson: with w quarters per side, the relative error of
    the ratio is √(1/(recent·w) + 1/(prior·w)). Returns None when a side is ~0 (growth is
    undefined-noisy there anyway).
    """
    if recent_mean <= 0 or prior_mean <= 0:
        return None
    rel = math.sqrt(1 / (recent_mean * window) + 1 / (prior_mean * window))
    return (recent_mean / prior_mean) * rel


def sentiment_quadrants(snapshot: dict) -> list[dict]:
    """Volume-change × critical-share-trend quadrants per topic (#11).

    The same critical-share number means different things depending on momentum:
    growing+more-critical = a field straining against problems; growing+less-critical =
    confidence (or premature certainty); shrinking+more-critical = post-mortem;
    shrinking+less-critical = fading/stabilizing. Reads inflections (volume change) and
    sentiment (critical-share trend) already in the snapshot.
    """
    infl = {i.get("topic_key"): i for i in snapshot.get("inflections") or []}
    out = []
    for k, v in (snapshot.get("sentiment") or {}).items():
        i = infl.get(k)
        if not i:
            continue
        vol, crit = i.get("change") or 0, v.get("trend") or 0
        if vol >= 0 and crit >= 0:
            quad = "growing & straining"
        elif vol >= 0:
            quad = "growing & confident"
        elif crit >= 0:
            quad = "contracting & critical"
        else:
            quad = "fading / stabilizing"
        out.append({
            "topic_key": k, "vol_change": round(vol, 3), "crit_trend": round(crit, 3),
            "quadrant": quad, "n_recent": v.get("n_recent", 0),
            "recent_share": v.get("recent_share", 0),
            "recent_per_qtr": i.get("recent_mean"),
        })
    return out


def confidence_adjusted_divergence(snapshot: dict) -> list[dict]:
    """Divergence gap weighted by each side's confidence posture (#12).

    Confidence posture = 1 − recent critical share: growth in a field publishing little
    self-critique reads as deployment-grade momentum (stronger warning); safety growth that
    is itself highly critical is weaker reassurance. adjusted_gap =
    cap_growth·cap_conf − saf_growth·saf_conf, shown NEXT TO the raw gap, never replacing
    it. Pairs missing sentiment data fall back to confidence 0.85 (the corpus-typical
    posture) so one thin topic doesn't zero the adjustment.
    """
    sent = snapshot.get("sentiment") or {}

    def conf(topic_key):
        v = sent.get(topic_key)
        if not v or (v.get("n_recent") or 0) < 5:
            return 0.85, False
        return 1 - (v.get("recent_share") or 0), True

    out = []
    for d in snapshot.get("divergence") or []:
        cc, c_real = conf(d.get("capability_topic"))
        sc, s_real = conf(d.get("safety_topic"))
        cap_g, saf_g = d.get("cap_growth") or 0, d.get("saf_growth") or 0
        adjusted = cap_g * cc - saf_g * sc
        reason_bits = []
        if c_real and cc >= 0.9:
            reason_bits.append("capability shows little self-critique (stronger warning)")
        if c_real and cc < 0.75:
            reason_bits.append("capability is unusually self-critical (softens the gap)")
        if s_real and sc < 0.75:
            reason_bits.append("safety growth is itself highly critical (weaker reassurance)")
        out.append({
            "pairing": d.get("pairing"), "raw_gap": d.get("gap"),
            "adjusted_gap": round(adjusted, 3),
            "cap_confidence": round(cc, 2), "saf_confidence": round(sc, 2),
            "reason": "; ".join(reason_bits) or "both sides near corpus-typical self-critique",
        })
    return out


def tab_deltas(snapshot: dict, previous: dict | None) -> dict:
    """Per-tab week-over-week deltas (#39): what changed since the previous snapshot.

    Extends the summary-level ``diff_snapshots`` to every tab, so a returning reader can
    scan movement without re-reading static state. Returns {} on a first run (no previous).
    Keys: divergence / velocity / sentiment / foresight / incidents / sources — each a dict
    of small lists ready to render.
    """
    if not previous:
        return {}
    prev_date = (previous.get("meta") or {}).get("refreshed_at")

    def _lagging(s):
        return {d["pairing"] for d in s.get("divergence") or [] if d.get("lagging")}

    div_now, div_prev = _lagging(snapshot), _lagging(previous)

    def _by_dir(s, direction):
        return {i["topic_key"] for i in s.get("inflections") or []
                if i.get("direction") == direction}

    accel_now, accel_prev = _by_dir(snapshot, "acceleration"), _by_dir(previous, "acceleration")
    decel_now, decel_prev = _by_dir(snapshot, "deceleration"), _by_dir(previous, "deceleration")

    sent_now, sent_prev = snapshot.get("sentiment") or {}, previous.get("sentiment") or {}
    rising_now = {k for k, v in sent_now.items() if v.get("rising")}
    rising_prev = {k for k, v in sent_prev.items() if v.get("rising")}
    shifts = sorted(
        ({"topic_key": k,
          "shift_pts": round((sent_now[k].get("trend") or 0) * 100
                             - (sent_prev.get(k, {}).get("trend") or 0) * 100, 1)}
         for k in sent_now if k in sent_prev),
        key=lambda r: abs(r["shift_pts"]), reverse=True)
    shifts = [s for s in shifts[:3] if abs(s["shift_pts"]) >= 2]

    def _risks(s):
        fg = (s.get("analysis") or {}).get("foresight_gap") or {}
        return {(r.get("risk") or "") for r in fg.get("risks") or [] if r.get("risk")}

    risks_now, risks_prev = _risks(snapshot), _risks(previous)

    def _incidents(s):
        return {(r.get("title"), r.get("date"))
                for r in (s.get("incidents") or {}).get("records") or []}

    inc_new = [{"title": t, "date": d}
               for (t, d) in sorted(_incidents(snapshot) - _incidents(previous),
                                    key=lambda x: x[1] or "", reverse=True)]

    return {
        "prev_date": prev_date,
        "divergence": {"new_lagging": sorted(div_now - div_prev),
                       "resolved": sorted(div_prev - div_now)},
        "velocity": {"new_accelerating": sorted(accel_now - accel_prev),
                     "new_decelerating": sorted(decel_now - decel_prev)},
        "sentiment": {"new_rising": sorted(rising_now - rising_prev),
                      "cleared": sorted(rising_prev - rising_now),
                      "biggest_shifts": shifts},
        "foresight": {"new_risks": sorted(risks_now - risks_prev),
                      "dropped_risks": sorted(risks_prev - risks_now)},
        "incidents": {"new": inc_new[:6]},
    }


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

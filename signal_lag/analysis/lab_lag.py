"""Lab-announcement → safety-response lag (#2).

Operationalizes the tool's name: for each lab/blog announcement tagged to a *capability*
topic that sits in a capability↔safety pairing, measure how long the paired **safety**
research takes to respond in the arXiv literature — the first safety paper afterwards, and
(more meaningfully) the first week the safety topic's volume rises measurably above its
pre-announcement baseline. Aggregates into a median response lag and unanswered-after-N-weeks
counts.

Pure/offline: it reads the topic-tagged lab posts, the paper corpus, and the taxonomy —
computed at snapshot-build time (where the full papers + tags are available), then baked into
the snapshot. Honest about sparsity: recent announcements whose response window hasn't elapsed
are reported as *pending*, never as *unanswered*.
"""
from __future__ import annotations

import datetime as dt
import logging
import statistics

log = logging.getLogger("signal_lag.lab_lag")


def _parse_date(s) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def lab_response_lag(
    posts: list, papers: list, tax_tags: dict, taxonomy, today: dt.date,
    baseline_weeks: int = 8, horizon_weeks: int = 12, uptick_factor: float = 1.5,
) -> dict:
    """Measure paired-safety research response lag to each capability lab announcement.

    Returns a dict with per-announcement rows, per-capability medians, and unanswered-after
    4/8/12-week counts. ``uptick_factor`` sets how far above the pre-announcement baseline the
    safety volume must rise to count as a *measurable* response. Fail-soft: no posts / no
    pairings / no dates -> ``{"available": False}``.
    """
    pairings = getattr(taxonomy, "pairings", None) or []
    if not posts or not pairings:
        return {"available": False}

    label_map = {t.key: t.label for t in taxonomy.all_topics}
    # capability topic -> list of paired safety topics.
    cap_to_safety: dict[str, list[str]] = {}
    for pr in pairings:
        cap_to_safety.setdefault(pr.capability, []).append(pr.safety)

    # Per-topic sorted list of paper publication dates.
    by_id = {p.arxiv_id: p.published for p in papers if getattr(p, "published", None)}
    safety_dates: dict[str, list[dt.date]] = {}
    for aid, tags in tax_tags.items():
        d = by_id.get(aid)
        if d is None:
            continue
        for tk, _ in tags:
            safety_dates.setdefault(tk, []).append(d)
    for k in safety_dates:
        safety_dates[k].sort()

    def _count_between(dates: list[dt.date], lo: dt.date, hi: dt.date) -> int:
        return sum(1 for d in dates if lo < d <= hi)

    rows = []
    for post in posts:
        cap = post.get("topic")
        if cap not in cap_to_safety:
            continue
        ann = _parse_date(post.get("published"))
        if ann is None:
            continue
        window_elapsed = (today - ann).days >= horizon_weeks * 7
        # Use the first paired safety topic that has any papers (usually one pairing per cap).
        for saf in cap_to_safety[cap]:
            dates = safety_dates.get(saf) or []
            if not dates:
                continue
            after = [d for d in dates if d > ann]
            days_to_first = (after[0] - ann).days if after else None
            base_lo = ann - dt.timedelta(weeks=baseline_weeks)
            baseline = _count_between(dates, base_lo, ann)
            baseline_rate = baseline / baseline_weeks  # papers/week before the announcement
            weeks_to_measurable = None
            for w in range(1, horizon_weeks + 1):
                hi = ann + dt.timedelta(weeks=w)
                if hi > today:
                    break
                cnt = _count_between(dates, ann, hi)
                expected = baseline_rate * w
                if cnt >= max(expected * uptick_factor, expected + 2):
                    weeks_to_measurable = w
                    break
            if weeks_to_measurable is not None:
                status = "responded"
            elif window_elapsed:
                status = "no measurable response"
            else:
                status = "pending"
            rows.append({
                "announcement": post.get("title"), "lab": post.get("source"),
                "published": ann.isoformat(),
                "capability": label_map.get(cap, cap), "capability_key": cap,
                "safety": label_map.get(saf, saf), "safety_key": saf,
                "days_to_first": days_to_first,
                "weeks_to_measurable": weeks_to_measurable,
                "baseline_per_week": round(baseline_rate, 2),
                "status": status, "window_elapsed": window_elapsed,
            })
            break  # one (capability, safety) row per post

    if not rows:
        return {"available": False}

    elapsed = [r for r in rows if r["window_elapsed"]]
    responded = [r for r in rows if r["status"] == "responded"]

    def _unanswered_by(week: int) -> int:
        # Elapsed-window posts with no measurable response by `week` weeks.
        n = 0
        for r in elapsed:
            wm = r["weeks_to_measurable"]
            if wm is None or wm > week:
                n += 1
        return n

    # Per-capability aggregation.
    by_cap: dict[str, list] = {}
    for r in rows:
        by_cap.setdefault(r["capability"], []).append(r)
    by_capability = []
    for cap_label, rs in sorted(by_cap.items(), key=lambda kv: -len(kv[1])):
        by_capability.append({
            "capability": cap_label, "n": len(rs),
            "median_days_to_first": _median([r["days_to_first"] for r in rs]),
            "median_weeks_to_measurable": _median(
                [r["weeks_to_measurable"] for r in rs if r["status"] == "responded"]),
            "n_unanswered": sum(
                1 for r in rs if r["window_elapsed"] and r["status"] != "responded"),
        })

    rows.sort(key=lambda r: r["published"], reverse=True)
    result = {
        "available": True,
        "n_posts_considered": len(rows),
        "n_window_elapsed": len(elapsed),
        "median_days_to_first": _median([r["days_to_first"] for r in rows]),
        "median_weeks_to_measurable": _median(
            [r["weeks_to_measurable"] for r in responded]),
        "unanswered": {"4": _unanswered_by(4), "8": _unanswered_by(8), "12": _unanswered_by(12)},
        "by_capability": by_capability,
        "posts": rows,
    }
    log.info("Lab-lag: %d announcements measured (%d windows elapsed), median measurable lag %s wk",
             len(rows), len(elapsed), result["median_weeks_to_measurable"])
    return result

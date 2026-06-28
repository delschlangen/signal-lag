"""Turn computed metrics into BLUF-style findings and a markdown brief."""
from __future__ import annotations

import datetime as dt

from ..config import Taxonomy


def _label(taxonomy: Taxonomy, key: str) -> str:
    t = taxonomy.topic(key)
    return t.label if t else key


def generate_signals(
    taxonomy: Taxonomy,
    divergence: list[dict],
    inflections: list[dict],
    new_clusters: list[str],
    cluster_labels: dict,
    citations: dict,
    institution_trends,
    sentiment: dict | None = None,
) -> list[dict]:
    """Return a ranked list of {severity, headline, detail} findings."""
    signals: list[dict] = []
    inflection_by_topic = {i["topic_key"]: i for i in inflections}

    # 0. Negative-signal early warnings: critical share rising (esp. if volume flat).
    for topic_key, s in (sentiment or {}).items():
        if not s.get("rising"):
            continue
        inf = inflection_by_topic.get(topic_key, {})
        vol_dir = inf.get("direction", "steady")
        flat = vol_dir != "acceleration"  # not clearly growing
        sev = "high" if flat else "medium"
        vol_phrase = (
            "while volume is flat/declining" if flat
            else "even as volume rises"
        )
        signals.append(
            {
                "severity": sev,
                "category": "sentiment",
                "headline": f"Confidence may be eroding in {_label(taxonomy, topic_key)}",
                "detail": (
                    f"Critical/negative papers rose from {s['prior_share']*100:.0f}% to "
                    f"{s['recent_share']*100:.0f}% of recent work (+{s['trend']*100:.0f} pts) "
                    f"{vol_phrase} — an early sign the field may be hitting limits."
                ),
            }
        )

    # 1. Divergence (headline product).
    for d in divergence:
        if not d["lagging"]:
            continue
        cap = _label(taxonomy, d["capability_topic"])
        saf = _label(taxonomy, d["safety_topic"])
        signals.append(
            {
                "severity": "high",
                "category": "divergence",
                "headline": f"Safety attention may lag capability: {cap} vs {saf}",
                "detail": (
                    f"{cap} velocity grew {d['cap_growth']*100:.0f}% recently while paired "
                    f"safety topic {saf} grew {d['saf_growth']*100:.0f}% "
                    f"(gap {d['gap']*100:.0f} pts)"
                    + (
                        f"; capability now runs ~{d['volume_ratio']:.1f}x the safety volume."
                        if d.get("volume_ratio")
                        else "."
                    )
                ),
            }
        )

    # 2. Acceleration inflections.
    for inf in inflections:
        if inf["direction"] == "acceleration" and inf["recent_mean"] >= 3:
            signals.append(
                {
                    "severity": "medium",
                    "category": "velocity",
                    "headline": f"{_label(taxonomy, inf['topic_key'])} is accelerating",
                    "detail": (
                        f"Submission rate inflected +{inf['change']*100:.0f}% "
                        f"(now ~{inf['recent_mean']:.1f}/quarter)."
                    ),
                }
            )

    # 3. Newly forming clusters (emergent topics).
    for lab in new_clusters:
        name = cluster_labels.get(lab, str(lab)) if isinstance(cluster_labels, dict) else lab
        signals.append(
            {
                "severity": "medium",
                "category": "emerging",
                "headline": "Newly forming research cluster detected",
                "detail": f"A cluster characterized by [{name}] emerged in recent quarters.",
            }
        )

    # 4. Sleeper citation heat.
    for s in citations.get("sleepers", [])[:5]:
        signals.append(
            {
                "severity": "medium",
                "category": "citation",
                "headline": "Sleeper paper heating up",
                "detail": (
                    f"\"{s['title']}\" was quiet early but {s['recent_share']*100:.0f}% "
                    f"of its {s['cited_by_count']} citations are recent."
                ),
            }
        )

    # 5. Institutional talent flow (top movers).
    if institution_trends is not None and len(institution_trends) > 0:
        for _, row in institution_trends.head(3).iterrows():
            if row["recent"] >= 2 and row["growth"] > 0.5:
                signals.append(
                    {
                        "severity": "low",
                        "category": "talent-flow",
                        "headline": f"{row['institution']} ramping in {_label(taxonomy, row['topic_key'])}",
                        "detail": (
                            f"{row['institution']} output in this topic grew "
                            f"{row['growth']*100:.0f}% recently ({row['recent']} recent papers)."
                        ),
                    }
                )

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(signals, key=lambda s: order.get(s["severity"], 3))


def render_brief(signals: list[dict], meta: dict, today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    lines = [
        "# Signal-Lag Foresight Brief",
        "",
        f"_Generated {today.isoformat()} · {meta.get('n_papers', '?')} papers · "
        f"embedding backend: {meta.get('backend', '?')}_",
        "",
        "## BLUF",
        "",
    ]
    highs = [s for s in signals if s["severity"] == "high"]
    if highs:
        for s in highs:
            lines.append(f"- **{s['headline']}** — {s['detail']}")
    else:
        lines.append("- No high-severity capability/safety divergences flagged this run.")
    lines.append("")

    by_cat: dict[str, list[dict]] = {}
    for s in signals:
        by_cat.setdefault(s["category"], []).append(s)

    pretty = {
        "divergence": "Capability vs. Safety Divergence",
        "sentiment": "Confidence / Negative Signals",
        "velocity": "Topic Velocity",
        "emerging": "Emerging Topics",
        "citation": "Citation Dynamics",
        "talent-flow": "Author / Institution Flow",
    }
    for cat, title in pretty.items():
        items = by_cat.get(cat, [])
        if not items:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for s in items:
            lines.append(f"- **{s['headline']}** ({s['severity']}): {s['detail']}")
        lines.append("")

    return "\n".join(lines)

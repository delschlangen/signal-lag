"""Build / load a serialized analysis snapshot.

The dashboard reads a precomputed ``data/snapshot.json`` instead of running the
(slow, network- and model-dependent) pipeline on every page load. A scheduled
GitHub Action regenerates the snapshot weekly from real arXiv + OpenAlex data and
commits it, so the app stays fast while showing fresh, real data.

The snapshot is plain JSON: DataFrames are stored as records, every paper carries
its arXiv URL for source linking, and a few headline lists are precomputed.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .analysis.runner import run_analysis
from .config import Settings, Taxonomy
from .ingest.store import Store

SNAPSHOT_VERSION = 1


def arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype).startswith("period"):
            out[col] = out[col].astype(str)
    return out.to_dict(orient="records")


def build_snapshot(
    settings: Settings, taxonomy: Taxonomy, mode: str, today: dt.date | None = None
) -> dict:
    """Run the full analysis and assemble a JSON-serializable snapshot dict."""
    today = today or dt.date.today()
    results = run_analysis(settings, taxonomy)

    # Re-open the cache (cheap; no re-embedding) to build source links.
    store = Store(settings.path("db_path"))
    papers = store.get_papers()
    tax_tags = store.get_tags("taxonomy")
    # Prefer topic-tagged posts from the analysis; fall back to raw posts.
    posts = results.get("lab_posts") or store.get_posts(limit=60)
    store.close()

    # Provenance breakdown (arxiv / openreview / ...).
    source_counts: dict[str, int] = {}
    for p in papers:
        source_counts[p.source] = source_counts.get(p.source, 0) + 1

    # Semantic Scholar coverage across the *whole* corpus (the recent-papers view
    # alone can't show this, since S2 rarely indexes brand-new arXiv papers).
    s2_enriched = sum(
        1 for p in papers
        if p.s2_tldr or p.s2_influential_citations is not None or p.venue
    )

    by_id = {p.arxiv_id: p for p in papers}
    label_map = {t.key: t.label for t in taxonomy.all_topics}

    # Recent representative papers per topic (for the Sources tab + summary links).
    per_topic: dict[str, list[dict]] = {t.key: [] for t in taxonomy.all_topics}
    for aid, tags in tax_tags.items():
        p = by_id.get(aid)
        if not p:
            continue
        for topic_key, score in tags:
            per_topic.setdefault(topic_key, []).append(
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "url": arxiv_url(p.arxiv_id),
                    "published": p.published.isoformat(),
                    "cited_by_count": p.cited_by_count,
                    "influential_citations": p.s2_influential_citations,
                    "venue": p.venue,
                    "tldr": p.s2_tldr,
                    "abstract": (p.abstract or "")[:300],
                    "source": p.source,
                    "score": round(float(score), 3),
                }
            )
    for k, lst in per_topic.items():
        lst.sort(key=lambda r: r["published"], reverse=True)
        per_topic[k] = lst[:8]

    # Add URLs + S2 fields to citation movers (these are older, cited papers that
    # Semantic Scholar is most likely to have indexed).
    cites = results["citations"]
    for bucket in ("rapid_growth", "sleepers"):
        for r in cites.get(bucket, []):
            r["url"] = arxiv_url(r["arxiv_id"])
            src = by_id.get(r["arxiv_id"])
            if src is not None:
                r["influential_citations"] = src.s2_influential_citations
                r["tldr"] = src.s2_tldr
                r["venue"] = src.venue
                r["abstract"] = (src.abstract or "")[:300]

    dates = [p.published for p in papers] or [today]
    meta = {
        "version": SNAPSHOT_VERSION,
        "refreshed_at": today.isoformat(),
        "mode": mode,  # "live" | "fixtures"
        "backend": results["meta"]["backend"],
        "n_papers": results["meta"]["n_papers"],
        "date_start": min(dates).isoformat(),
        "date_end": max(dates).isoformat(),
        "categories": settings.arxiv_categories,
        "topics_tracked": len(taxonomy.all_topics),
        "n_pairings": len(taxonomy.pairings),
        "n_flagged": sum(1 for d in results["divergence"] if d["lagging"]),
        "source_counts": source_counts,
        "n_posts": len(posts),
        "s2_enriched": s2_enriched,
    }

    snap_out = {
        "meta": meta,
        "label_map": label_map,
        "divergence": results["divergence"],
        "inflections": results["inflections"],
        "quadrant": results["quadrant"],
        "new_clusters": results["new_clusters"],
        "citations": cites,
        "institution_trends": _records(results["institution_trends"]),
        "signals": results["signals"],
        "brief": results["brief"],
        "timeseries": _records(results["taxonomy_timeseries"]),
        "sentiment": results.get("sentiment", {}),
        "sentiment_timeseries": _records(results.get("sentiment_timeseries")),
        "sources": per_topic,
        "lab_activity": posts,
        "analysis": results.get("analysis"),
    }

    # --- Foresight Gap: second Claude pass crossing this week's signals with the
    # living societal context. Same fail-soft, baked-into-snapshot architecture as the
    # primary analysis (no page-load API calls).
    acfg = settings.analysis or {}
    fcfg = acfg.get("foresight") or {}
    if fcfg.get("enabled"):
        from .analysis import foresight

        try:
            prev_snap = load_snapshot(settings.path("snapshot_path"))
        except Exception:
            prev_snap = None
        diff = diff_snapshots(snap_out, prev_snap)
        ctx = foresight.load_context(settings.root / fcfg.get("context_path", "config/context.md"))
        digest = foresight.build_signal_digest(snap_out, diff)
        fg = foresight.synthesize_foresight_gap(
            digest, ctx, acfg.get("api_key"),
            acfg.get("model", "claude-opus-4-8"),
            int(fcfg.get("max_risks", 4)),
        )
        if fg:
            if snap_out.get("analysis") is None:
                snap_out["analysis"] = {}
            snap_out["analysis"]["foresight_gap"] = fg

    return snap_out


def save_snapshot(snapshot: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Archive the previous snapshot so the summary can diff week-over-week.
    if path.exists():
        prev = path.with_name("snapshot_prev.json")
        prev.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")


def load_snapshot(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def diff_snapshots(current: dict, previous: dict | None) -> dict:
    """What changed since the last refresh: new alerts / accelerations / sleepers."""
    if not previous:
        return {"first_run": True, "new_alerts": [], "new_accelerations": [],
                "new_sleepers": [], "prev_date": None}

    prev_flagged = {d["pairing"] for d in previous.get("divergence", []) if d["lagging"]}
    new_alerts = [
        d for d in current.get("divergence", [])
        if d["lagging"] and d["pairing"] not in prev_flagged
    ]

    prev_accel = {
        i["topic_key"] for i in previous.get("inflections", [])
        if i["direction"] == "acceleration"
    }
    new_accel = [
        i for i in current.get("inflections", [])
        if i["direction"] == "acceleration" and i["topic_key"] not in prev_accel
    ]

    prev_sleepers = {s["arxiv_id"] for s in previous.get("citations", {}).get("sleepers", [])}
    new_sleepers = [
        s for s in current.get("citations", {}).get("sleepers", [])
        if s["arxiv_id"] not in prev_sleepers
    ]

    return {
        "first_run": False,
        "prev_date": previous.get("meta", {}).get("refreshed_at"),
        "new_alerts": new_alerts,
        "new_accelerations": new_accel,
        "new_sleepers": new_sleepers,
    }

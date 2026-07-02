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
    harm_tags = store.get_tags("harm")
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

    # Targeted citation-heat: a small, reliable enrichment of just the papers DRIVING
    # the foresight gaps (top divergence pairs + emerging quadrant + verified citation-
    # flow borrowers) so real citation counts land where they're shown — the full-corpus
    # keyless S2 pass can't finish all ~12.7k papers in its time budget. Live-only.
    if mode == "live" and settings.section("citations").get("targeted_heat", True):
        _enrich_targeted_heat(settings, results, taxonomy, tax_tags, by_id)

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

    # Harm/misuse dual-use lens: attach representative ENABLING papers per harm vector
    # (the same per-topic representative-paper pattern), so the Harm Foresight tab can
    # show what's driving each vector. Built from the parallel source="harm" tags.
    harm_block = results.get("harm")
    if harm_block:
        harm_reps: dict[str, list[dict]] = {v["key"]: [] for v in harm_block["vectors"]}
        for aid, tags in harm_tags.items():
            p = by_id.get(aid)
            if not p:
                continue
            for topic_key, score in tags:
                if topic_key in harm_reps:
                    harm_reps[topic_key].append({
                        "arxiv_id": p.arxiv_id, "title": p.title,
                        "url": arxiv_url(p.arxiv_id), "published": p.published.isoformat(),
                        "abstract": (p.abstract or "")[:300], "score": round(float(score), 3),
                    })
        for k, lst in harm_reps.items():
            lst.sort(key=lambda r: r["published"], reverse=True)
            harm_reps[k] = lst[:6]
        for v in harm_block["vectors"]:
            v["rep_papers"] = harm_reps.get(v["key"], [])
        harm_block["timeseries"] = _records(results.get("harm_timeseries"))

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
        "n_tagged": len(tax_tags),
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
        "sentiment_llm_verify": results.get("sentiment_llm_verify"),
        "citation_flow": results.get("citation_flow"),
        "citation_graph": results.get("citation_graph"),
        "author_migration": results.get("author_migration"),
        "harm": harm_block,
        "sources": per_topic,
        "lab_activity": posts,
        "analysis": results.get("analysis"),
    }

    # --- Foresight Gap: second Claude pass crossing this week's signals with the
    # living societal context. At full-build time the on-disk snapshot is last week's,
    # so it's the right baseline for the week-over-week diff.
    try:
        prev_on_disk = load_snapshot(settings.path("snapshot_path"))
    except Exception:
        prev_on_disk = None
    snap_out = augment_foresight(settings, snap_out, prev_on_disk)

    # --- Step 5: real-world incident benchmark (leading research vs lagging incidents) ---
    snap_out = augment_incidents(settings, snap_out)

    # --- Lab-announcement -> safety-response lag (#2): how long paired safety research
    # takes to answer each lab announcement in the arXiv literature. ---
    lcfg = (settings.analysis or {}).get("lab_lag") or {}
    if lcfg.get("enabled", True):
        from .analysis import lab_lag as lab_lag_mod
        snap_out["lab_lag"] = lab_lag_mod.lab_response_lag(
            posts, papers, tax_tags, taxonomy, today,
            baseline_weeks=int(lcfg.get("baseline_weeks", 8)),
            horizon_weeks=int(lcfg.get("horizon_weeks", 12)),
            uptick_factor=float(lcfg.get("uptick_factor", 1.5)),
        )

    # --- "This week" lens: a focused analysis of just the last `window_days` of papers,
    # alongside the quarterly view (which is unaffected).
    snap_out = build_weekly(
        settings, taxonomy, snap_out, papers, tax_tags, by_id, today, prev_on_disk,
        results.get("paper_critical") or {},
    )

    return snap_out


def _enrich_targeted_heat(settings, results, taxonomy, tax_tags, by_id) -> int:
    """Fetch real citation counts for the papers driving the foresight gaps only.

    Targets: the top-3 lagging-divergence pairs' topics, the emerging-quadrant topics,
    and the verified citation-flow borrowers (+ the safety papers they cite). Enriches
    those Paper objects in place (so per_topic + borrower cards pick up the counts) and
    annotates the citation-flow borrowers with ``cited_by_count``. Bounded + fail-soft.
    """
    from .ingest.pipeline import enrich_specific_citations

    cap = int(settings.section("citations").get("targeted_heat_max", 300))

    # Topics that matter most this run.
    target_keys: set[str] = set()
    div = sorted([d for d in results.get("divergence", []) if d.get("lagging")],
                 key=lambda d: d.get("gap", 0), reverse=True)[:3]
    for d in div:
        target_keys.add(d["capability_topic"])
        target_keys.add(d["safety_topic"])
    for q in results.get("quadrant", []) or []:
        if q.get("quadrant") == "emerging":
            target_keys.add(q["topic_key"])

    # Most-recent papers per target topic (mirrors what the Sources tab shows).
    topic_to_ids: dict[str, list[str]] = {}
    for aid, tags in tax_tags.items():
        if aid not in by_id:
            continue
        for tk, _ in tags:
            if tk in target_keys:
                topic_to_ids.setdefault(tk, []).append(aid)
    target_ids: set[str] = set()
    for tk, ids in topic_to_ids.items():
        ids.sort(key=lambda a: by_id[a].published, reverse=True)
        target_ids.update(ids[:8])

    # Verified citation-flow borrowers + the safety papers they cite.
    cf = results.get("citation_flow") or {}
    for b in cf.get("verified_borrowers", []) or []:
        target_ids.add(b.get("arxiv_id"))
        for c in b.get("cited_safety", []) or []:
            target_ids.add(c.get("arxiv_id"))
    target_ids.discard(None)

    targets = [by_id[a] for a in list(target_ids)[:cap] if a in by_id]
    if not targets:
        return 0
    n = enrich_specific_citations(settings, targets)

    # Annotate citation-flow borrowers with the freshly-fetched counts.
    for b in cf.get("verified_borrowers", []) or []:
        sp = by_id.get(b.get("arxiv_id"))
        if sp is not None:
            b["cited_by_count"] = sp.cited_by_count
        for c in b.get("cited_safety", []) or []:
            cp = by_id.get(c.get("arxiv_id"))
            if cp is not None:
                c["cited_by_count"] = cp.cited_by_count
    return n


def build_weekly(
    settings: Settings, taxonomy: Taxonomy, snapshot: dict, papers, tax_tags: dict,
    by_id: dict, today: dt.date, prev_snapshot: dict | None, paper_critical: dict,
) -> dict:
    """Attach snapshot["weekly"]: counts + Claude summary + verified foresight for the
    last `window_days` of papers only. Config-gated and fail-soft; the quarterly view is
    untouched. Computed from the already-tagged cache (no re-embedding)."""
    acfg = settings.analysis or {}
    wcfg = acfg.get("weekly") or {}
    if not wcfg.get("enabled"):
        return snapshot
    from .analysis import foresight, llm

    window_days = int(wcfg.get("window_days", 7))
    cutoff = today - dt.timedelta(days=window_days)
    week_ids = {p.arxiv_id for p in papers if p.published >= cutoff}
    if not week_ids:
        return snapshot

    label_map = snapshot.get("label_map", {})
    safety_keys = {t.key for t in taxonomy.safety_topics}
    cap_keys = {t.key for t in taxonomy.capability_topics}
    counts = {"safety": {}, "capability": {}}
    counts_by_key: dict[str, int] = {}          # topic_key -> count (for chart tabs)
    crit_by_key: dict[str, int] = {}            # topic_key -> # critical this week
    for aid in week_ids:
        is_crit = bool(paper_critical.get(aid))
        for topic_key, _score in tax_tags.get(aid, []):
            counts_by_key[topic_key] = counts_by_key.get(topic_key, 0) + 1
            if is_crit:
                crit_by_key[topic_key] = crit_by_key.get(topic_key, 0) + 1
            bucket = ("safety" if topic_key in safety_keys
                      else "capability" if topic_key in cap_keys else None)
            if bucket:
                name = label_map.get(topic_key, topic_key)
                counts[bucket][name] = counts[bucket].get(name, 0) + 1
    counts = {b: dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))
              for b, d in counts.items()}
    # Per-topic this-week critical share (topics with >=3 papers, so it's not noise).
    weekly_sentiment = {
        k: {"n": n, "critical_share": round(crit_by_key.get(k, 0) / n, 3)}
        for k, n in counts_by_key.items() if n >= 3
    }

    week_papers = [by_id[a] for a in week_ids if a in by_id]
    week_papers.sort(key=lambda p: ((p.cited_by_count or 0), p.published), reverse=True)
    notable = [
        {"arxiv_id": p.arxiv_id, "title": p.title, "url": arxiv_url(p.arxiv_id),
         "published": p.published.isoformat(), "cited_by_count": p.cited_by_count,
         "venue": p.venue, "tldr": p.s2_tldr,
         "topics": [label_map.get(tk, tk) for tk, _ in tax_tags.get(p.arxiv_id, [])],
         "abstract": (p.abstract or "")[:300]}
        for p in week_papers[:15]
    ]

    api_key = acfg.get("api_key")
    model = acfg.get("model", "claude-opus-4-8")
    summary = llm.summarize_week(
        {"window": f"last {window_days} days", "n_papers": len(week_ids),
         "counts_by_topic": counts,
         "notable_papers": [{"arxiv_id": n["arxiv_id"], "title": n["title"],
                             "abstract": n["abstract"]} for n in notable]},
        api_key, model,
    )

    fcfg = acfg.get("foresight") or {}
    ctx = foresight.load_context(settings.root / fcfg.get("context_path", "config/context.md"))
    diff = diff_snapshots(snapshot, prev_snapshot)
    wdigest = foresight.build_weekly_digest(window_days, counts, notable, snapshot, diff)
    tool_version = fcfg.get("web_search_tool", "web_search_20260209")
    # Reuse the overall live web brief if one was fetched this refresh (avoid a 2nd search);
    # else fetch once for the weekly pass when enabled. Fail-soft.
    live_ctx = ((snapshot.get("analysis") or {}).get("foresight_gap") or {}).get("live_context")
    if live_ctx is None and fcfg.get("live_context"):
        live_ctx = foresight.fetch_live_context(
            ctx, foresight.build_signal_digest(snapshot, diff), api_key, model, tool_version)
    lens = (
        f"Reason about what THESE SPECIFIC papers from the last {window_days} days imply. "
        "Focus on what is newly emerging THIS WEEK, not the long-run quarterly trend (given "
        "only as backdrop). Anchor on the backdrop research-trend signal, then cross THIS "
        "WEEK's papers with the societal context to find the seam."
    )
    wfg = foresight.run_foresight(
        wdigest, ctx, api_key, model,
        max_risks=int(wcfg.get("max_risks", 3)),
        verify=bool(fcfg.get("verify_novelty")),
        tool_version=tool_version,
        min_surfaced=int(wcfg.get("min_surfaced", 2)),
        max_rounds=int(wcfg.get("max_rounds", 2)),
        lens=lens,
        live_context=live_ctx,
        verify_cache_path=_verify_cache_path(settings, fcfg),
        today=snapshot.get("meta", {}).get("refreshed_at", ""),
        verify_cache_days=int(fcfg.get("verify_cache_days", 21)),
    )

    snapshot["weekly"] = {
        "window_days": window_days,
        "cutoff": cutoff.isoformat(),
        "n_papers": len(week_ids),
        "counts_by_topic": counts,
        "counts_by_key": counts_by_key,
        "sentiment": weekly_sentiment,
        "notable_papers": notable,
        "summary": summary,
        "foresight_gap": wfg,
    }
    return snapshot


def _history_record(snapshot: dict, prev_snapshot: dict | None) -> dict:
    """A compact weekly briefing record for the History tab (no abstracts/raw data)."""
    meta = snapshot.get("meta", {})
    lm = snapshot.get("label_map", {})
    analysis = snapshot.get("analysis") or {}
    hl = analysis.get("headline") or {}

    alerts = sorted([d for d in snapshot.get("divergence", []) if d.get("lagging")],
                    key=lambda a: a.get("gap", 0), reverse=True)
    if alerts:
        a = alerts[0]
        gap_line = (f"{lm.get(a['capability_topic'], a['capability_topic'])} vs "
                    f"{lm.get(a['safety_topic'], a['safety_topic'])} "
                    f"(cap {a['cap_growth']*100:+.0f}% / saf {a['saf_growth']*100:+.0f}%)")
    else:
        gap_line = "No pairing crosses the safety-lag threshold this week."

    def top_fore(fg, n=3):
        out = []
        for r in (fg or {}).get("risks", []):
            rating = (r.get("verification") or {}).get("novelty_rating")
            if rating == "already_widely_discussed":
                continue
            out.append({"risk": r.get("risk"), "novelty_rating": rating,
                        "domains_crossed": r.get("domains_crossed")})
            if len(out) >= n:
                break
        return out

    sent = snapshot.get("sentiment", {}) or {}
    rising = [lm.get(k, k) for k, v in sent.items() if v.get("rising")]
    diff = diff_snapshots(snapshot, prev_snapshot)
    return {
        "date": meta.get("refreshed_at"),
        "n_papers": meta.get("n_papers"),
        "date_start": meta.get("date_start"),
        "date_end": meta.get("date_end"),
        "n_flagged": meta.get("n_flagged"),
        "n_pairings": meta.get("n_pairings"),
        "headline": {
            "biggest_gap_line": gap_line,
            "meaning": hl.get("meaning"),
            "why_it_matters": hl.get("why_it_matters"),
        },
        "sentiment_rising": rising,
        "overall_foresight": top_fore(analysis.get("foresight_gap")),
        "weekly_foresight": top_fore((snapshot.get("weekly") or {}).get("foresight_gap")),
        "what_changed": {
            "n_alerts": len(diff.get("new_alerts", [])),
            "n_accel": len(diff.get("new_accelerations", [])),
            "n_sleepers": len(diff.get("new_sleepers", [])),
        },
    }


def append_history(snapshot: dict, path: Path, prev_snapshot: dict | None = None) -> None:
    """Append a compact weekly briefing to data/history.json (idempotent per date)."""
    path = Path(path)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        existing = []
    if not isinstance(existing, list):
        existing = []
    rec = _history_record(snapshot, prev_snapshot)
    if not rec.get("date"):
        return
    existing = [e for e in existing if e.get("date") != rec["date"]]
    existing.append(rec)
    existing.sort(key=lambda e: e.get("date") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")


def _verify_cache_path(settings: Settings, fcfg: dict):
    """Path for the novelty-verification cache (#25), or None when disabled."""
    if not fcfg.get("verify_cache", True):
        return None
    return settings.root / "data" / "verify_cache.json"


def _risk_id(statement: str) -> str:
    """Stable id for a risk = sha1 of its normalized statement (first 12 hex chars)."""
    import hashlib
    norm = " ".join((statement or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _register_entries(snapshot: dict) -> list[dict]:
    """Scored risks from the overall + weekly foresight passes, as register-ready dicts."""
    date = snapshot.get("meta", {}).get("refreshed_at")
    out = []
    sources = [("overall", (snapshot.get("analysis") or {}).get("foresight_gap")),
               ("weekly", (snapshot.get("weekly") or {}).get("foresight_gap"))]
    for src, fg in sources:
        for r in (fg or {}).get("risks", []) or []:
            stmt = r.get("risk")
            if not stmt:
                continue
            v = r.get("verification") or {}
            out.append({
                "id": _risk_id(stmt), "risk": stmt, "source": src, "date": date,
                "severity": r.get("severity"), "likelihood": r.get("likelihood"),
                "exposure": r.get("exposure"), "trajectory": r.get("trajectory"),
                "priority": r.get("priority"),
                "confidence": r.get("confidence"),
                "evidence_strength": r.get("evidence_strength"),
                "actionability": r.get("actionability"),
                "novelty_rating": v.get("novelty_rating"),
                # Counterevidence (#23): the web-verifier's dispute findings, persisted so
                # a risk carries its evidence-against over time, not just this refresh's.
                "disputed_claims": v.get("disputed_claims"),
                "prior_coverage": (v.get("prior_coverage") or "")[:500] or None,
                "sources": [{"title": s.get("title"), "url": s.get("url")}
                            for s in (v.get("sources") or [])[:3]],
                "domains_crossed": r.get("domains_crossed"),
                "leading_indicator": r.get("leading_indicator"),
            })
    return out


def _is_disputed(disputed_claims) -> bool:
    """True when the verifier found a real dispute (not 'none found' boilerplate)."""
    t = (disputed_claims or "").strip().lower()
    return bool(t) and not t.startswith("none")


# Recalibration (#5): a total order over the register so risks don't pile up at one tie.
# Primary = priority (severity × likelihood); ties broken by calibrated confidence, then
# trajectory (a worsening signal outranks a fading one), then evidence freshness (more
# recently re-seen = more current), then exposure. Stale risks (not surfaced in the latest
# refresh) take a one-tier penalty so fresh evidence rises — "if everything is urgent,
# nothing is".
_TRAJ_RANK = {"accelerating": 2, "steady": 1, "decelerating": 0}
_STALE_PENALTY = 6  # a full severity×likelihood tier


def register_newest_date(register: list) -> str:
    """The most recent last_seen across the register (the 'current refresh' marker)."""
    return max((e.get("last_seen") or "" for e in register or []), default="")


def register_is_stale(entry: dict, newest_date: str) -> bool:
    """A risk is stale if it wasn't surfaced in the most recent refresh (no new evidence)."""
    return bool(newest_date) and (entry.get("last_seen") or "") < newest_date


def register_sort_key(entry: dict, newest_date: str) -> tuple:
    """Multi-key descending sort key (see module note). Higher tuple = higher rank."""
    lt = entry.get("latest") or {}
    eff_priority = (lt.get("priority") or 0) - (
        _STALE_PENALTY if register_is_stale(entry, newest_date) else 0)
    return (
        eff_priority,
        lt.get("confidence") or 0,
        _TRAJ_RANK.get(lt.get("trajectory"), 1),
        entry.get("last_seen") or "",
        lt.get("exposure") or 0,
    )


def sort_register(register: list) -> list:
    """Return the register in recalibrated priority order (forced ranking, stale-downgraded)."""
    newest = register_newest_date(register)
    return sorted(register or [], key=lambda e: register_sort_key(e, newest), reverse=True)


def register_status(entry: dict, newest_date: str) -> str:
    """Watchlist status (#6), derived from the entry's own score history + staleness.

    - ``dormant``       — not re-surfaced in the latest refresh (no new evidence).
    - ``strengthening`` — re-seen with priority or confidence moving UP vs the prior point.
    - ``weakening``     — re-seen with priority moving DOWN.
    - ``open``          — new this refresh, or re-seen with scores holding steady.

    ``materialized`` / ``invalidated`` / ``partially confirmed`` are deliberately NOT
    auto-derived: they require outcome evidence (a linked incident, a broken mechanism)
    the pipeline can't yet establish — deriving them from score drift would fake
    precision. They join the vocabulary when evidence linkage lands.
    """
    if register_is_stale(entry, newest_date):
        return "dormant"
    h = entry.get("history") or []
    if len(h) >= 2:
        prev_p, last_p = h[-2].get("priority") or 0, h[-1].get("priority") or 0
        lt = entry.get("latest") or {}
        if last_p > prev_p:
            return "strengthening"
        if last_p < prev_p:
            return "weakening"
        # Equal priority: growing persistence with high confidence still reads as
        # strengthening evidence; otherwise it simply stays open.
        if (entry.get("n_appearances") or 1) >= 3 and (lt.get("confidence") or 0) >= 4:
            return "strengthening"
    return "open"


def append_risk_register(snapshot: dict, path: Path) -> None:
    """Upsert this refresh's scored risks into an evergreen register (idempotent per date).

    The register is the JD's "evergreen frontier risk register": every risk ever surfaced,
    keyed by a stable id, with first_seen / last_seen / appearance count and a per-refresh
    score history (severity, likelihood, exposure, trajectory, priority) so trajectory over
    time is visible. Mirrors append_history's compact, idempotent JSON-list pattern.
    """
    path = Path(path)
    try:
        reg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        reg = []
    if not isinstance(reg, list):
        reg = []
    by_id = {e["id"]: e for e in reg if e.get("id")}
    date = snapshot.get("meta", {}).get("refreshed_at")
    if not date:
        return

    # Dedup within this refresh by id (prefer the higher-priority appearance).
    seen: dict[str, dict] = {}
    for e in _register_entries(snapshot):
        cur = seen.get(e["id"])
        if cur is None or (e.get("priority") or 0) > (cur.get("priority") or 0):
            seen[e["id"]] = e

    for e in seen.values():
        rid = e["id"]
        point = {"date": date, "severity": e["severity"], "likelihood": e["likelihood"],
                 "exposure": e["exposure"], "trajectory": e["trajectory"],
                 "priority": e["priority"], "disputed": _is_disputed(e.get("disputed_claims"))}
        rec = by_id.get(rid)
        if rec is None:
            by_id[rid] = rec = {"id": rid, "risk": e["risk"], "first_seen": date,
                                "last_seen": date, "n_appearances": 1, "latest": e,
                                "history": [point]}
        else:
            rec["last_seen"] = date
            rec["risk"] = e["risk"]
            rec["latest"] = e
            if not rec.get("history") or rec["history"][-1].get("date") != date:
                rec["n_appearances"] = rec.get("n_appearances", 1) + 1
                rec.setdefault("history", []).append(point)
            else:                       # same date re-run -> overwrite, don't double-count
                rec["history"][-1] = point
        # Counterevidence trail (#23): keep every DISTINCT dispute the verifier ever found
        # for this risk, dated — evidence-against accumulates instead of being overwritten.
        if _is_disputed(e.get("disputed_claims")):
            trail = rec.setdefault("counterevidence", [])
            if not any(c.get("disputed_claims") == e["disputed_claims"] for c in trail):
                trail.append({"date": date, "disputed_claims": e["disputed_claims"]})

    out = sort_register(list(by_id.values()))
    # Watchlist status (#6), stored so exports/consumers of the JSON carry it too.
    newest = register_newest_date(out)
    for e in out:
        e["status"] = register_status(e, newest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def append_benchmark_history(snapshot: dict, path: Path) -> None:
    """Persist this refresh's harm-vector benchmark rows (#28), idempotent per date.

    Accumulates {date, key, label, quadrant, research_change_pct, n_incidents} per refresh
    so 'foresight lead → later incidents' transitions (time-to-incident, false-positive
    rate) become computable as history builds. Same compact JSON-list pattern as the
    register; fail-soft.
    """
    bench = (snapshot.get("incidents") or {}).get("benchmark") or []
    date = snapshot.get("meta", {}).get("refreshed_at")
    if not bench or not date:
        return
    path = Path(path)
    try:
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        rows = []
    if not isinstance(rows, list):
        rows = []
    rows = [r for r in rows if r.get("date") != date]      # same-date re-run -> overwrite
    for b in bench:
        rows.append({"date": date, "key": b.get("key"), "label": b.get("label"),
                     "quadrant": b.get("quadrant"),
                     "research_change_pct": b.get("research_change_pct"),
                     "n_incidents": b.get("n_incidents")})
    rows.sort(key=lambda r: (r.get("date") or "", r.get("key") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def _paper_lookup(snapshot: dict) -> dict:
    """{arxiv_id -> {title, abstract}} from the snapshot's source + notable papers, so the
    plain-language explainer can reference a risk's source papers by title/abstract."""
    out: dict[str, dict] = {}
    for rows in (snapshot.get("sources") or {}).values():
        for r in rows:
            if r.get("arxiv_id"):
                out.setdefault(r["arxiv_id"],
                               {"title": r.get("title"), "abstract": r.get("abstract")})
    for n in ((snapshot.get("weekly") or {}).get("notable_papers") or []):
        if n.get("arxiv_id"):
            out.setdefault(n["arxiv_id"],
                           {"title": n.get("title"), "abstract": n.get("abstract")})
    return out


def augment_foresight(settings: Settings, snapshot: dict, prev_snapshot: dict | None) -> dict:
    """Run the Foresight Gap pass and attach analysis["foresight_gap"] to ``snapshot``.

    Same fail-soft, baked-into-snapshot architecture as the primary analysis (no
    page-load API calls). Reusable so the synthesis can be re-run against an updated
    ``config/context.md`` without re-pulling data (see refresh_snapshot --foresight-only).
    """
    acfg = settings.analysis or {}
    fcfg = acfg.get("foresight") or {}
    if not fcfg.get("enabled"):
        return snapshot
    from .analysis import foresight

    diff = diff_snapshots(snapshot, prev_snapshot)
    ctx = foresight.load_context(settings.root / fcfg.get("context_path", "config/context.md"))
    digest = foresight.build_signal_digest(snapshot, diff)
    api_key = acfg.get("api_key")
    model = acfg.get("model", "claude-opus-4-8")
    tool_version = fcfg.get("web_search_tool", "web_search_20260209")
    # Pre-synthesis live web brief (#3): pull the CURRENT real-world status of the flagged
    # topics so synthesis doesn't anchor on a stale context.md. Fail-soft (-> None).
    live_ctx = None
    if fcfg.get("live_context"):
        live_ctx = foresight.fetch_live_context(ctx, digest, api_key, model, tool_version)
    # Synthesize -> verify each candidate against current web coverage -> backfill with
    # fresh seams until enough survive (quality over quantity). All baked into the
    # snapshot, so the searches are cached (run once per refresh, never at page load).
    fg = foresight.run_foresight(
        digest, ctx, api_key, model,
        max_risks=int(fcfg.get("max_risks", 4)),
        verify=bool(fcfg.get("verify_novelty")),
        tool_version=tool_version,
        min_surfaced=int(fcfg.get("min_surfaced", 3)),
        max_rounds=int(fcfg.get("max_rounds", 3)),
        live_context=live_ctx,
        verify_cache_path=_verify_cache_path(settings, fcfg),
        today=snapshot.get("meta", {}).get("refreshed_at", ""),
        verify_cache_days=int(fcfg.get("verify_cache_days", 21)),
    )
    if fg:
        # Scenario analysis (Step 3): one Claude pass developing 6-24mo scenarios from the
        # top-priority surfaced risks. Config-gated + fail-soft; baked into the snapshot.
        if fcfg.get("scenarios"):
            scen = foresight.generate_scenarios(
                fg.get("risks") or [], ctx, api_key, model,
                max_scenarios=int(fcfg.get("max_scenarios", 3)),
                live_context=live_ctx,
            )
            if scen:
                fg["scenarios"] = scen
        # Plain-language explanations for the top-N risks (legibility layer): a 5-part
        # walkthrough (technical evidence / societal context / the gap / self-skepticism /
        # bottom line) shown in-app + folded into the exports. Config-gated + fail-soft.
        if fcfg.get("explainers"):
            foresight.attach_explanations(
                fg.get("risks") or [], _paper_lookup(snapshot), live_ctx or ctx,
                api_key, model, max_explainers=int(fcfg.get("max_explainers", 4)),
            )
        if snapshot.get("analysis") is None:
            snapshot["analysis"] = {}
        snapshot["analysis"]["foresight_gap"] = fg
    return snapshot


def augment_incidents(settings: Settings, snapshot: dict) -> dict:
    """Step 5: cross the upstream research-enablement signal with REAL-WORLD incidents.

    Fetches verifiable recent AI-misuse incidents (via Claude web search), tags them to the
    harm vectors, and benchmarks each vector's research momentum against its incident count
    into a leading-vs-lagging 2×2: *materializing* (research up + incidents), *foresight
    lead* (research up, no incidents yet — the tool's edge), *active/known* (incidents but
    research flat/down), *quiet* (neither). Config-gated + fail-soft; baked into the snapshot.
    """
    acfg = settings.analysis or {}
    icfg = acfg.get("incidents") or {}
    if not icfg.get("enabled"):
        return snapshot
    vectors = (snapshot.get("harm") or {}).get("vectors") or []
    if not vectors:
        return snapshot
    from .analysis import foresight

    fcfg = acfg.get("foresight") or {}
    incidents = foresight.fetch_incidents(
        vectors, acfg.get("api_key"), acfg.get("model", "claude-opus-4-8"),
        fcfg.get("web_search_tool", "web_search_20260209"),
        max_incidents=int(icfg.get("max_incidents", 20)),
        today=snapshot.get("meta", {}).get("refreshed_at", ""),
    ) or []
    counts: dict[str, int] = {}
    for inc in incidents:
        counts[inc.get("harm_key")] = counts.get(inc.get("harm_key"), 0) + 1
    rising_pct = float(icfg.get("rising_pct", 4))
    benchmark = []
    for v in vectors:
        n_inc = counts.get(v["key"], 0)
        rising = (v.get("change_pct") or 0) >= rising_pct
        if rising and n_inc:
            quad = "materializing"
        elif rising and not n_inc:
            quad = "foresight lead"
        elif n_inc:
            quad = "active / known"
        else:
            quad = "quiet"
        benchmark.append({
            "key": v["key"], "label": v["label"],
            "research_change_pct": v.get("change_pct"), "n_research": v.get("n_tagged"),
            "n_incidents": n_inc, "quadrant": quad,
        })
    benchmark.sort(key=lambda b: (b["n_incidents"], b.get("research_change_pct") or 0),
                   reverse=True)
    snapshot["incidents"] = {"records": incidents, "benchmark": benchmark, "n": len(incidents)}
    return snapshot


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

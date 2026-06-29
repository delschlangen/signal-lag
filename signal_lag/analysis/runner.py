"""End-to-end analysis: load cache -> embed -> cluster + tag -> velocity ->
citations -> divergence -> author flow -> signals/brief.

Returns a single results dict consumed by the CLI and the Streamlit dashboard.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np

from ..config import Settings, Taxonomy
from ..ingest.store import Store
import pandas as pd

from . import (
    authors, citation_flow as citation_flow_mod, citations, cluster, divergence,
    llm, sentiment, signals, taxonomy as tax_mod, velocity,
)
from .embeddings import Embedder, save_embeddings

log = logging.getLogger("signal_lag.runner")


def _excerpt(text: str | None, limit: int = 420) -> str:
    text = " ".join((text or "").split())
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def _build_llm_payload(
    taxonomy, div, inflections, quad, topic_sent, cite, lab_posts,
    by_id, tax_tags, label_map, n_per_side: int,
) -> dict:
    """Assemble a compact, fully-real payload for the weekly LLM analysis.

    Grounds Claude in: the widest capability/safety pairing + representative real
    abstracts on each side, every pairing's growth, velocity inflections, rising
    critical-share topics, the quadrant map, citation movers, lab announcements,
    and a deduplicated `papers` list (id+title+abstract) for per-paper notes.
    """
    # Recent representative papers per topic (newest first).
    reps: dict[str, list] = {}
    for aid, tags in tax_tags.items():
        p = by_id.get(aid)
        if not p:
            continue
        for topic_key, _score in tags:
            reps.setdefault(topic_key, []).append(p)
    for k, lst in reps.items():
        lst.sort(key=lambda p: p.published, reverse=True)

    def topic_papers(key: str, n: int) -> list[dict]:
        return [
            {"arxiv_id": p.arxiv_id, "title": p.title, "abstract": _excerpt(p.abstract)}
            for p in reps.get(key, [])[:n]
        ]

    papers: dict[str, dict] = {}

    def add_paper(p_like) -> None:
        aid = p_like.get("arxiv_id") if isinstance(p_like, dict) else p_like.arxiv_id
        if not aid or aid in papers:
            return
        if isinstance(p_like, dict):
            papers[aid] = {"arxiv_id": aid, "title": p_like.get("title", ""),
                           "abstract": _excerpt(p_like.get("abstract"))}
        else:
            papers[aid] = {"arxiv_id": aid, "title": p_like.title,
                           "abstract": _excerpt(p_like.abstract)}

    pairings = []
    for d in div:
        cap_reps = topic_papers(d["capability_topic"], n_per_side)
        saf_reps = topic_papers(d["safety_topic"], n_per_side)
        for r in cap_reps + saf_reps:
            add_paper(by_id.get(r["arxiv_id"]) or r)
        pairings.append({
            "pairing": d["pairing"],
            "capability_topic": label_map.get(d["capability_topic"], d["capability_topic"]),
            "safety_topic": label_map.get(d["safety_topic"], d["safety_topic"]),
            "capability_growth_pct_per_qtr": round(d["cap_growth"] * 100, 1),
            "safety_growth_pct_per_qtr": round(d["saf_growth"] * 100, 1),
            "gap": d["gap"],
            "volume_ratio_cap_over_saf": d["volume_ratio"],
            "flagged_safety_lagging": d["lagging"],
            "capability_papers": cap_reps,
            "safety_papers": saf_reps,
        })

    # Citation movers also get per-paper notes.
    for bucket in ("rapid_growth", "sleepers"):
        for r in cite.get(bucket, [])[:5]:
            src = by_id.get(r["arxiv_id"])
            if src:
                add_paper(src)

    velocity_rows = [
        {"topic": label_map.get(i["topic_key"], i["topic_key"]),
         "change_pct": round(i["change"] * 100, 1),
         "recent_per_qtr": round(i["recent_mean"], 1)}
        for i in sorted(inflections, key=lambda i: i.get("change", 0), reverse=True)
    ]
    sentiment_rows = [
        {"topic": label_map.get(k, k),
         "recent_critical_share_pct": round(v.get("recent_share", 0) * 100, 1),
         "trend_pts": round(v.get("trend", 0) * 100, 1),
         "rising": bool(v.get("rising")), "n_recent": v.get("n_recent", 0)}
        for k, v in sorted(topic_sent.items(),
                           key=lambda kv: kv[1].get("trend", 0), reverse=True)
    ]
    quadrant_rows = [
        {"topic": label_map.get(q["topic_key"], q["topic_key"]), "quadrant": q["quadrant"]}
        for q in quad
    ]
    citation_rows = {
        b: [{"arxiv_id": r["arxiv_id"], "title": r["title"],
             "cited_by_count": r.get("cited_by_count")} for r in cite.get(b, [])[:5]]
        for b in ("rapid_growth", "sleepers")
    }
    lab_rows = [
        {"source": p.get("source"), "title": p.get("title"),
         "topic": label_map.get(p.get("topic"), p.get("topic")) if p.get("topic") else None,
         "published": p.get("published"), "summary": _excerpt(p.get("summary"), 240)}
        for p in (lab_posts or [])[:12]
    ]

    return {
        "pairings": pairings,
        "velocity_inflections": velocity_rows,
        "sentiment": sentiment_rows,
        "quadrant": quadrant_rows,
        "citation_movers": citation_rows,
        "lab_announcements": lab_rows,
        "papers": list(papers.values()),
    }


def run_analysis(settings: Settings, taxonomy: Taxonomy) -> dict:
    store = Store(settings.path("db_path"))
    papers = store.get_papers()
    if not papers:
        store.close()
        raise RuntimeError("No papers in cache. Run `ingest` first.")
    log.info("Loaded %d papers", len(papers))

    emb_cfg = settings.section("embeddings")
    embedder = Embedder(
        model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        fallback_svd_components=int(emb_cfg.get("fallback_svd_components", 256)),
        batch_size=int(emb_cfg.get("batch_size", 64)),
    )

    ids = [p.arxiv_id for p in papers]
    texts = [f"{p.title}. {p.abstract}" for p in papers]
    vecs = embedder.embed(texts)
    try:
        save_embeddings(settings.path("embeddings_path"), ids, vecs, embedder.backend)
    except Exception as e:  # non-fatal
        log.warning("Could not cache embeddings: %s", e)

    # --- supervised taxonomy tagging ---
    centroids = tax_mod.build_topic_centroids(taxonomy, embedder)
    tax_rows = tax_mod.tag_papers(ids, vecs, centroids, taxonomy)
    store.replace_tags("taxonomy", tax_rows)
    tax_tags = store.get_tags("taxonomy")

    # --- unsupervised clustering ---
    labels = cluster.cluster_embeddings(vecs, settings.section("clustering"))
    cluster_labels = cluster.label_clusters(
        labels, texts, int(settings.section("clustering").get("label_top_terms", 5))
    )
    store.replace_tags(
        "cluster", [(ids[i], f"c{int(lab)}", 1.0) for i, lab in enumerate(labels) if lab != -1]
    )
    cluster_tags = store.get_tags("cluster")

    # --- velocity (taxonomy topics) ---
    vcfg = settings.section("velocity")
    today = dt.date.today()
    # Trim the current incomplete quarter so trend math sees only complete ones.
    tax_ts = velocity.drop_incomplete_tail(velocity.topic_timeseries(papers, tax_tags), today)
    inflections = velocity.compute_inflections(
        tax_ts, int(vcfg.get("inflection_window", 2)), float(vcfg.get("inflection_threshold", 0.3))
    )

    # --- velocity (clusters) for emergent/new-cluster detection ---
    cluster_ts = velocity.drop_incomplete_tail(
        velocity.topic_timeseries(papers, cluster_tags), today
    )
    new_cluster_keys = velocity.newly_forming(
        cluster_ts, int(vcfg.get("new_cluster_max_age_periods", 3))
    )
    # map "c3" -> readable label
    def _clab(key: str) -> str:
        try:
            return cluster_labels.get(int(key[1:]), key)
        except (ValueError, IndexError):
            return key

    new_clusters = [_clab(k) for k in new_cluster_keys]

    # Analysis (Claude) config, hoisted so the sentiment/citation passes can reuse it.
    acfg = settings.analysis
    acfg_api_key = acfg.get("api_key") if acfg.get("enabled") else None
    acfg_model = acfg.get("model", "claude-opus-4-8")

    # --- negative / critical-signal layer ---
    scfg = settings.section("sentiment")
    neg_centroid = sentiment.build_negativity_centroid(taxonomy, embedder)
    crit = sentiment.critical_scores(vecs, neg_centroid)
    _crit_thr = float(scfg.get("critical_threshold", 0.22))
    paper_critical = {ids[i]: bool(crit[i] >= _crit_thr) for i in range(len(ids))}
    periods = {p.arxiv_id: pd.Period(p.published, freq="Q") for p in papers}

    # --- hybrid LLM sentiment (#1): embedding recall, LLM precision ---
    # The embedding centroid mistakes academic negation ("we overcome the catastrophic
    # failures of prior methods") for genuine criticism. Take only the RECENT-window
    # papers the embedding flagged critical (the subset that drives the rising-share
    # signal), batch-verify them with Claude, and downgrade the false positives. Bounded
    # and fail-soft: no key / disabled / error => pure embedding behavior (unchanged).
    sent_llm_meta = None
    if scfg.get("llm_verify") and acfg_api_key:
        idx_of = {aid: i for i, aid in enumerate(ids)}
        by_id_p = {p.arxiv_id: p for p in papers}
        uniq_periods = sorted(set(periods.values()))
        _win = int(scfg.get("window", 2))
        recent_cut = uniq_periods[-_win] if len(uniq_periods) >= _win else uniq_periods[0]
        flagged_recent = [
            aid for aid in ids
            if paper_critical.get(aid) and periods[aid] >= recent_cut
        ]
        flagged_recent.sort(key=lambda a: periods[a], reverse=True)
        cap = int(scfg.get("llm_verify_max", 400))
        subset = flagged_recent[:cap]
        payload = [
            {"arxiv_id": aid, "title": by_id_p[aid].title, "abstract": by_id_p[aid].abstract}
            for aid in subset if aid in by_id_p
        ]
        labels = llm.classify_limitation_focused(
            payload, acfg_api_key, acfg_model,
            batch_size=int(scfg.get("llm_verify_batch", 50)),
        )
        flipped = 0
        for aid, is_crit in labels.items():
            if not is_crit and paper_critical.get(aid):
                paper_critical[aid] = False
                crit[idx_of[aid]] = 0.0    # snap below threshold so trend math agrees
                flipped += 1
        sent_llm_meta = {"verified": len(labels), "downgraded": flipped,
                         "subset": len(subset)}
        log.info("LLM sentiment verify: downgraded %d of %d recent flagged-critical papers",
                 flipped, len(subset))

    topic_sent = sentiment.topic_sentiment(ids, crit, periods, tax_tags, taxonomy, scfg)
    sent_ts = velocity.drop_incomplete_tail(
        sentiment.sentiment_timeseries(
            ids, crit, periods, tax_tags, float(scfg.get("critical_threshold", 0.22))
        ),
        today,
    )

    # --- lab/blog posts tagged to topics (capability-leading signal) ---
    lab_posts = []
    posts = store.get_posts(limit=80)
    if posts and centroids:
        ptexts = [f"{p.get('title','')}. {p.get('summary','')}" for p in posts]
        pvecs = embedder.embed(ptexts)
        ckeys = list(centroids.keys())
        cmat = np.vstack([centroids[k] for k in ckeys])
        psims = pvecs @ cmat.T
        for i, p in enumerate(posts):
            j = int(psims[i].argmax())
            score = float(psims[i][j])
            lab_posts.append({
                **p,
                "topic": ckeys[j] if score >= 0.18 else None,
                "topic_score": round(score, 3),
            })

    # --- citation dynamics ---
    cite = citations.citation_signals(papers, settings.section("citations"))

    # --- citation-flow verification (does capability work actually cite safety work?) ---
    cfcfg = settings.section("citation_flow")
    cflow = None
    if cfcfg.get("enabled", True):
        cflow = citation_flow_mod.citation_flow(
            papers, tax_tags, taxonomy,
            max_examples=int(cfcfg.get("max_examples", 30)),
        )

    # --- divergence (headline) ---
    dcfg = settings.section("divergence")
    div = divergence.compute_divergence(
        tax_ts, taxonomy,
        int(vcfg.get("inflection_window", 2)),
        float(dcfg.get("gap_threshold", 0.25)),
        float(dcfg.get("min_recent_volume", 3)),
    )
    quad = divergence.quadrant_view(inflections, tax_ts)

    # --- author / institution flow ---
    inst_trends = authors.institution_topic_trends(
        papers, tax_tags, int(vcfg.get("inflection_window", 2))
    )

    # --- author migration (#4, experimental leading indicator) ---
    amcfg = acfg.get("author_migration") or {}
    author_mig = None
    if amcfg.get("enabled"):
        author_mig = authors.author_migration(
            papers, tax_tags, taxonomy,
            min_history=int(amcfg.get("min_history", 2)),
        )

    # --- signals + brief ---
    sigs = signals.generate_signals(
        taxonomy, div, inflections, new_clusters, {}, cite, inst_trends,
        sentiment=topic_sent,
    )
    meta = {"n_papers": len(papers), "backend": embedder.backend}
    brief = signals.render_brief(sigs, meta)

    # --- optional weekly LLM analysis (Anthropic / Claude) ---
    analysis = None
    if acfg.get("enabled"):
        by_id = {p.arxiv_id: p for p in papers}
        label_map = {t.key: t.label for t in taxonomy.all_topics}
        payload = _build_llm_payload(
            taxonomy, div, inflections, quad, topic_sent, cite, lab_posts,
            by_id, tax_tags, label_map, int(acfg.get("papers_per_side", 3)),
        )
        analysis = llm.analyze_weekly(
            payload, acfg.get("api_key"), acfg.get("model", "claude-opus-4-8")
        )

    store.close()
    return {
        "meta": meta,
        "taxonomy_timeseries": tax_ts,
        "cluster_timeseries": cluster_ts,
        "cluster_labels": cluster_labels,
        "inflections": inflections,
        "new_clusters": new_clusters,
        "citations": cite,
        "divergence": div,
        "quadrant": quad,
        "institution_trends": inst_trends,
        "sentiment": topic_sent,
        "sentiment_timeseries": sent_ts,
        "sentiment_llm_verify": sent_llm_meta,
        "paper_critical": paper_critical,
        "citation_flow": cflow,
        "author_migration": author_mig,
        "lab_posts": lab_posts,
        "signals": sigs,
        "brief": brief,
        "analysis": analysis,
    }

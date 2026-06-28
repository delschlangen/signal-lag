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
    authors, citations, cluster, divergence, sentiment, signals,
    taxonomy as tax_mod, velocity,
)
from .embeddings import Embedder, save_embeddings

log = logging.getLogger("signal_lag.runner")


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

    # --- negative / critical-signal layer ---
    scfg = settings.section("sentiment")
    neg_centroid = sentiment.build_negativity_centroid(taxonomy, embedder)
    crit = sentiment.critical_scores(vecs, neg_centroid)
    periods = {p.arxiv_id: pd.Period(p.published, freq="Q") for p in papers}
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

    # --- signals + brief ---
    sigs = signals.generate_signals(
        taxonomy, div, inflections, new_clusters, {}, cite, inst_trends,
        sentiment=topic_sent,
    )
    meta = {"n_papers": len(papers), "backend": embedder.backend}
    brief = signals.render_brief(sigs, meta)

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
        "lab_posts": lab_posts,
        "signals": sigs,
        "brief": brief,
    }

"""Tag papers against the supervised taxonomy via embedding similarity.

Each topic's seed phrases are embedded and averaged into a centroid. A paper is
tagged with a topic when the cosine similarity between its abstract embedding and
the topic centroid exceeds ``tag_threshold`` (top-N kept per paper).
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import Taxonomy
from .embeddings import Embedder

log = logging.getLogger("signal_lag.taxonomy")


def build_topic_centroids_from(topics, embedder: Embedder) -> dict[str, np.ndarray]:
    """Return {topic_key: L2-normalized seed-centroid} for an arbitrary topic list.

    Generic over any taxonomy track (research topics, harm/misuse vectors, ...): each
    topic's seed phrases are embedded and averaged into one normalized vector.
    """
    keys, texts = [], []
    for topic in topics:
        for seed in topic.seeds:
            keys.append(topic.key)
            texts.append(seed)
    if not texts:
        return {}
    seed_vecs = embedder.embed(texts)
    centroids: dict[str, list[np.ndarray]] = {}
    for k, v in zip(keys, seed_vecs):
        centroids.setdefault(k, []).append(v)
    out = {}
    for k, vs in centroids.items():
        c = np.mean(np.vstack(vs), axis=0)
        norm = np.linalg.norm(c)
        out[k] = (c / norm).astype(np.float32) if norm else c.astype(np.float32)
    return out


def build_topic_centroids(taxonomy: Taxonomy, embedder: Embedder) -> dict[str, np.ndarray]:
    """Return {topic_key: centroid vector} for the research taxonomy (all_topics)."""
    return build_topic_centroids_from(taxonomy.all_topics, embedder)


def tag_papers(
    paper_ids: list[str],
    paper_vecs: np.ndarray,
    centroids: dict[str, np.ndarray],
    taxonomy: Taxonomy,
    topic_thresholds: dict[str, float] | None = None,
) -> list[tuple[str, str, float]]:
    """Return tag rows: (arxiv_id, topic_key, score).

    ``topic_thresholds`` (#1) optionally overrides the global ``tag_threshold`` per
    topic — the audit's remedy for over-inclusive topics: an overbroad centroid gets a
    tighter cutoff instead of one global compromise value.
    """
    keys = list(centroids.keys())
    cmat = np.vstack([centroids[k] for k in keys])  # (T, d)
    sims = paper_vecs @ cmat.T  # cosine (vectors are normalized) -> (N, T)
    thr = {k: float((topic_thresholds or {}).get(k, taxonomy.tag_threshold)) for k in keys}

    rows: list[tuple[str, str, float]] = []
    for i, pid in enumerate(paper_ids):
        scores = sims[i]
        order = np.argsort(scores)[::-1][: taxonomy.max_tags_per_paper]
        for j in order:
            if scores[j] >= thr[keys[j]]:
                rows.append((pid, keys[j], float(scores[j])))
    log.info("Taxonomy tagged %d (paper,topic) pairs across %d papers", len(rows), len(paper_ids))
    return rows


def confidence_label(score: float, threshold: float) -> str:
    """Tag-confidence tier (#1): how far above its topic's cutoff a match scored.

    high ≥ threshold+0.10 · medium ≥ threshold+0.04 · else weak (a borderline
    semantic match that just cleared the bar).
    """
    if score >= threshold + 0.10:
        return "high"
    if score >= threshold + 0.04:
        return "medium"
    return "weak"

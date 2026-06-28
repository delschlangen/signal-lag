"""Unsupervised topic discovery over abstract embeddings.

HDBSCAN (default) finds an unknown number of dense clusters and marks the rest as
noise — ideal for surfacing *emergent* topics. k-means is available as a config
switch. Clusters are auto-labeled with their most distinctive terms via a
c-TF-IDF style scoring (terms frequent in the cluster but rare elsewhere).
"""
from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger("signal_lag.cluster")

_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")
_STOP = set(
    "the a an and or of for to in on with we our this that these those is are be "
    "using use used via from by as at it its can model models language large "
    "results show paper propose present approach method methods task tasks new".split()
)


def cluster_embeddings(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    """Return an array of cluster labels (-1 = noise/unclustered)."""
    algo = cfg.get("algorithm", "hdbscan")
    if algo == "kmeans":
        from sklearn.cluster import KMeans

        k = int(cfg.get("kmeans", {}).get("n_clusters", 25))
        k = min(k, max(2, matrix.shape[0] - 1))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)
        log.info("k-means: %d clusters", k)
        return labels

    # default: HDBSCAN. High-dim transformer embeddings (384-d) make HDBSCAN
    # collapse everything to noise, so we first reduce to a low-dim space.
    try:
        import hdbscan

        reduced = _reduce_dims(matrix, int(cfg.get("reduce_dims", 50)))
        h = cfg.get("hdbscan", {})
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(h.get("min_cluster_size", 10)),
            min_samples=int(h.get("min_samples", 3)),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(reduced.astype(np.float64))
        n = len(set(labels)) - (1 if -1 in labels else 0)
        log.info("HDBSCAN: %d clusters, %d noise", n, int((labels == -1).sum()))
        # If it still collapses to noise, fall back to k-means for usable clusters.
        if n == 0:
            log.warning("HDBSCAN found 0 clusters; falling back to k-means")
            return cluster_embeddings(matrix, {**cfg, "algorithm": "kmeans"})
        return labels
    except Exception as e:
        log.warning("HDBSCAN unavailable (%s); falling back to k-means", e)
        return cluster_embeddings(matrix, {**cfg, "algorithm": "kmeans"})


def _reduce_dims(matrix: np.ndarray, n_components: int) -> np.ndarray:
    """SVD-reduce to n_components dims (no-op if already small enough)."""
    if matrix.shape[1] <= n_components or matrix.shape[0] <= n_components:
        return matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return normalize(svd.fit_transform(matrix))


def label_clusters(
    labels: np.ndarray, texts: list[str], top_terms: int = 5
) -> dict[int, str]:
    """c-TF-IDF style labels: terms over-represented in each cluster."""
    from collections import Counter, defaultdict

    cluster_tokens: dict[int, Counter] = defaultdict(Counter)
    global_tokens: Counter = Counter()
    for lab, text in zip(labels, texts):
        toks = [t.lower() for t in _WORD.findall(text) if t.lower() not in _STOP]
        cluster_tokens[lab].update(toks)
        global_tokens.update(toks)

    total = sum(global_tokens.values()) or 1
    out: dict[int, str] = {}
    for lab, counter in cluster_tokens.items():
        if lab == -1:
            out[lab] = "(unclustered)"
            continue
        csize = sum(counter.values()) or 1
        scored = []
        for term, c in counter.items():
            tf = c / csize
            idf = np.log(total / (global_tokens[term] + 1))
            scored.append((tf * idf, term))
        scored.sort(reverse=True)
        out[lab] = ", ".join(t for _, t in scored[:top_terms]) or f"cluster {lab}"
    return out

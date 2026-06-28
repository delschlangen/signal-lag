"""Dual-path text embedder.

Primary: sentence-transformers (e.g. all-MiniLM-L6-v2), a strong local model.
Fallback: scikit-learn TF-IDF + TruncatedSVD, used automatically when the
sentence-transformers model can't be loaded (offline, no HuggingFace access).

Either path returns L2-normalized float32 vectors, so cosine similarity is a dot
product and downstream code is identical regardless of which path ran.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("signal_lag.embeddings")


class Embedder:
    def __init__(self, model_name: str, fallback_svd_components: int = 256, batch_size: int = 64):
        self.model_name = model_name
        self.fallback_svd_components = fallback_svd_components
        self.batch_size = batch_size
        self.backend = "uninitialized"
        self._model = None
        self._fallback = None  # (vectorizer, svd) once fit

    def _try_load_st(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
            log.info("Embedding backend: sentence-transformers (%s)", self.model_name)
            return True
        except Exception as e:  # ImportError or model download failure
            log.warning("sentence-transformers unavailable (%s); using TF-IDF fallback", e)
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts -> (n, d) normalized float32 matrix."""
        if self.backend == "uninitialized":
            if not self._try_load_st():
                self.backend = "tfidf-svd"

        if self.backend == "sentence-transformers":
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype=np.float32)

        return self._embed_fallback(texts)

    def _embed_fallback(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if self._fallback is None:
            vectorizer = TfidfVectorizer(
                max_features=20000, stop_words="english", ngram_range=(1, 2)
            )
            tfidf = vectorizer.fit_transform(texts)
            n_comp = min(self.fallback_svd_components, tfidf.shape[1] - 1, max(2, len(texts) - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            reduced = svd.fit_transform(tfidf)
            self._fallback = (vectorizer, svd)
            return normalize(reduced).astype(np.float32)

        vectorizer, svd = self._fallback
        reduced = svd.transform(vectorizer.transform(texts))
        return normalize(reduced).astype(np.float32)


def save_embeddings(path, ids: list[str], matrix: np.ndarray, backend: str) -> None:
    np.savez_compressed(path, ids=np.array(ids), matrix=matrix, backend=backend)


def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["matrix"], str(data["backend"])

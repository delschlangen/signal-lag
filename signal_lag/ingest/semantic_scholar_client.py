"""Semantic Scholar enrichment (free, optional).

Adds, per paper matched by arXiv id: a TLDR summary, the influential-citation
count (a better "heat" signal than raw citations), the publication venue, and
fields of study. Uses the batch endpoint (up to 500 ids/call), so enriching
thousands of papers is a handful of requests.

Entirely fail-soft: any error (rate limit, outage, no match) just leaves the
fields empty — nothing else in the pipeline is affected.
"""
from __future__ import annotations

import logging
import time

import requests

from ..models import Paper

log = logging.getLogger("signal_lag.s2")

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "externalIds,tldr,influentialCitationCount,venue,fieldsOfStudy"


class SemanticScholarClient:
    def __init__(
        self,
        api_key: str | None = None,
        batch_size: int = 200,
        request_delay: float = 1.0,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.batch_size = batch_size
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "signal-lag/0.1"})
        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def _post_batch(self, ids: list[str]) -> list[dict] | None:
        for attempt, wait in enumerate([0, 5, 15, 30]):
            if wait:
                time.sleep(wait)
            try:
                resp = self.session.post(
                    S2_BATCH,
                    params={"fields": FIELDS},
                    json={"ids": ids},
                    timeout=60,
                )
                if resp.status_code == 429:
                    log.warning("S2 429, backing off (attempt %d)", attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                log.warning("S2 batch error: %s", e)
                continue
        return None

    def enrich(self, papers: list[Paper]) -> int:
        """Enrich papers in place by arXiv id. Returns count enriched."""
        by_id = {p.arxiv_id: p for p in papers}
        ids = [f"ARXIV:{aid}" for aid in by_id]
        enriched = 0
        for i in range(0, len(ids), self.batch_size):
            chunk = ids[i : i + self.batch_size]
            data = self._post_batch(chunk)
            time.sleep(self.request_delay)
            if not data:
                continue
            for rec in data:
                if not rec:
                    continue  # S2 returns null for unmatched ids
                ext = rec.get("externalIds") or {}
                aid = ext.get("ArXiv")
                p = by_id.get(aid)
                if p is None:
                    continue
                tldr = rec.get("tldr") or {}
                p.s2_tldr = tldr.get("text")
                p.s2_influential_citations = rec.get("influentialCitationCount")
                p.venue = rec.get("venue") or None
                p.fields_of_study = rec.get("fieldsOfStudy") or []
                enriched += 1
            log.info("  S2 enriched %d/%d", min(i + self.batch_size, len(ids)), len(ids))
        return enriched

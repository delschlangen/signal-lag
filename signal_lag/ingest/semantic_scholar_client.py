"""Semantic Scholar enrichment (free, optional).

Adds, per paper matched by arXiv id: a TLDR summary, the influential-citation
count (a better "heat" signal than raw citations), the publication venue, and
fields of study. Uses the batch endpoint (up to 500 ids/call), so enriching
thousands of papers is a handful of requests.

Also supplies the data OpenAlex would (OpenAlex is unreachable from the CI
runner): the total ``citationCount``, the paper's outgoing ``references`` (their
arXiv ids — used for citation-FLOW verification), and stable author ids (used for
author-migration tracking). Citation *by-year* series and institutions are NOT
available via this batch endpoint, so those signals stay empty without OpenAlex.

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
# Nested selections (references.externalIds, authors.authorId) are supported by the
# S2 graph API. references gives the outgoing bibliography; authors gives stable ids.
FIELDS = (
    "externalIds,tldr,influentialCitationCount,venue,fieldsOfStudy,"
    "citationCount,references.externalIds,authors.authorId,authors.name,"
    "authors.affiliations"
)


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
        for attempt, wait in enumerate([0, 3, 8]):
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

    def enrich(self, papers: list[Paper], time_budget_s: float = 480.0) -> int:
        """Enrich papers in place by arXiv id. Returns count enriched.

        Keyless access is heavily rate-limited, so this is best-effort: it keeps
        going through transient failures (up to a tolerance) but stops once a
        wall-clock budget is exceeded, so a refresh can't stall on S2.
        """
        import time as _time

        by_id = {p.arxiv_id: p for p in papers}
        ids = [f"ARXIV:{aid}" for aid in by_id]
        enriched = 0
        consecutive_failures = 0
        # Keyless: many small batches fail less often than a few large ones.
        fail_tolerance = 10 if not self.api_key else 3
        start = _time.monotonic()
        for i in range(0, len(ids), self.batch_size):
            if _time.monotonic() - start > time_budget_s:
                log.warning("S2 time budget reached; enriched %d so far", enriched)
                break
            chunk = ids[i : i + self.batch_size]
            data = self._post_batch(chunk)
            time.sleep(self.request_delay)
            if not data:
                consecutive_failures += 1
                if consecutive_failures >= fail_tolerance:
                    log.warning("S2 unavailable after %d tries; stopping (set "
                                "SEMANTIC_SCHOLAR_API_KEY for reliable access)",
                                fail_tolerance)
                    break
                continue
            consecutive_failures = 0
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
                # OpenAlex stand-ins (OpenAlex is unreachable in CI):
                # total citation count for per-paper "heat".
                if rec.get("citationCount") is not None:
                    p.cited_by_count = rec.get("citationCount")
                # Outgoing references -> arXiv ids of cited papers (citation-FLOW #2).
                refs: list[str] = []
                for r in (rec.get("references") or []):
                    rid = ((r or {}).get("externalIds") or {}).get("ArXiv")
                    if rid:
                        refs.append(rid)
                if refs:
                    p.referenced_works = refs
                # Stable author ids (author-migration #4) + affiliations (#19 ecosystem):
                # match S2 authors to ours positionally.
                s2_authors = rec.get("authors") or []
                insts: list[str] = []
                for j, a in enumerate(p.authors):
                    s2a = s2_authors[j] if j < len(s2_authors) else {}
                    if not a.openalex_id and s2a.get("authorId"):
                        a.openalex_id = s2a["authorId"]
                    affs = s2a.get("affiliations") or []
                    if affs and not a.affiliation:
                        a.affiliation = str(affs[0])[:120]
                    for aff in affs:
                        aff = str(aff)[:120]
                        if aff and aff not in insts:
                            insts.append(aff)
                if insts and not p.institutions:
                    p.institutions = insts
                enriched += 1
            log.info("  S2 enriched %d/%d", min(i + self.batch_size, len(ids)), len(ids))
        return enriched

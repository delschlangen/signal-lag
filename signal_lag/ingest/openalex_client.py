"""OpenAlex enrichment: citation counts, yearly citation series, and affiliations.

OpenAlex is free and keyless; supplying a ``mailto`` joins the faster "polite
pool". Papers are matched by arXiv id. Failures degrade gracefully — a paper
that can't be enriched simply keeps null citation fields.
"""
from __future__ import annotations

import logging
import time

import requests

from ..models import Paper

log = logging.getLogger("signal_lag.openalex")

OPENALEX_WORK = "https://api.openalex.org/works/https://arxiv.org/abs/{arxiv_id}"


class OpenAlexClient:
    def __init__(
        self,
        mailto: str = "",
        request_delay: float = 0.2,
        backoff_schedule: list[float] | None = None,
        session: requests.Session | None = None,
    ):
        self.mailto = mailto
        self.request_delay = request_delay
        self.backoff_schedule = backoff_schedule or [2, 4, 8, 16]
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "signal-lag/0.1 (mailto:%s)" % mailto})

    def _get(self, url: str) -> dict | None:
        params = {"mailto": self.mailto} if self.mailto else {}
        for attempt, wait in enumerate([0, *self.backoff_schedule]):
            if wait:
                time.sleep(wait)
            try:
                resp = self.session.get(url, params=params, timeout=45)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    log.warning("OpenAlex 429, backing off (attempt %d)", attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                log.warning("OpenAlex error: %s", e)
                continue
        return None

    def enrich(self, paper: Paper) -> Paper:
        data = self._get(OPENALEX_WORK.format(arxiv_id=paper.arxiv_id))
        time.sleep(self.request_delay)
        if not data:
            return paper
        paper.openalex_id = data.get("id")
        paper.cited_by_count = data.get("cited_by_count")
        paper.counts_by_year = [
            {"year": d.get("year"), "count": d.get("cited_by_count")}
            for d in data.get("counts_by_year", [])
        ]
        institutions: list[str] = []
        for authorship in data.get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("display_name")
                if name and name not in institutions:
                    institutions.append(name)
        paper.institutions = institutions
        return paper

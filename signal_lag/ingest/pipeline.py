"""Orchestrates ingestion: pull from arXiv -> cache -> optionally enrich.

Designed to be resumable and idempotent. ``--use-fixtures`` swaps the live arXiv
pull for a bundled synthetic dataset so the whole pipeline runs without network.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from ..config import Settings
from ..models import Author, Paper
from .arxiv_client import ArxivClient
from .openalex_client import OpenAlexClient
from .store import Store

log = logging.getLogger("signal_lag.pipeline")

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sample_papers.json"


def load_fixture_papers(path: Path = FIXTURES) -> list[Paper]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    papers = []
    for r in raw:
        papers.append(
            Paper(
                arxiv_id=r["arxiv_id"],
                title=r["title"],
                abstract=r["abstract"],
                published=dt.date.fromisoformat(r["published"]),
                updated=dt.date.fromisoformat(r["updated"]) if r.get("updated") else None,
                categories=r.get("categories", []),
                primary_category=r.get("primary_category"),
                authors=[
                    Author(name=a["name"], affiliation=a.get("affiliation"))
                    for a in r.get("authors", [])
                ],
                cited_by_count=r.get("cited_by_count"),
                counts_by_year=r.get("counts_by_year", []),
                institutions=r.get("institutions", []),
            )
        )
    return papers


def ingest(settings: Settings, use_fixtures: bool = False, enrich: bool = True) -> int:
    """Run ingestion. Returns the total paper count in the cache afterward."""
    store = Store(settings.path("db_path"))

    if use_fixtures:
        papers = load_fixture_papers()
        n = store.upsert_papers(papers)
        # Fixtures carry their own enrichment fields; persist them too.
        for p in papers:
            if p.cited_by_count is not None:
                store.update_enrichment(p)
        log.info("Loaded %d fixture papers", n)
        total = store.count_papers()
        store.close()
        return total

    start_date, end_date = settings.date_range
    client = ArxivClient(
        page_size=settings.arxiv_page_size,
        request_delay=settings.arxiv_request_delay_seconds,
        backoff_schedule=settings.backoff_schedule,
    )
    for category in settings.arxiv_categories:
        log.info("Fetching %s (%s..%s)", category, start_date, end_date)
        batch: list[Paper] = []
        for paper in client.search_category(
            category, start_date, end_date, settings.max_results_per_category
        ):
            batch.append(paper)
            if len(batch) >= 200:
                store.upsert_papers(batch)
                log.info("  cached %d (running total %d)", len(batch), store.count_papers())
                batch = []
        if batch:
            store.upsert_papers(batch)
        log.info("  %s done; cache now %d papers", category, store.count_papers())

    if enrich:
        enrich_citations(settings, store)

    total = store.count_papers()
    store.close()
    return total


def enrich_citations(settings: Settings, store: Store | None = None) -> int:
    """Fill OpenAlex citation/affiliation data for papers lacking it."""
    own = store is None
    store = store or Store(settings.path("db_path"))
    client = OpenAlexClient(
        mailto=settings.openalex_mailto,
        backoff_schedule=settings.backoff_schedule,
    )
    todo = store.papers_needing_enrichment(limit=settings.openalex_max_enrich)
    log.info("Enriching %d papers via OpenAlex", len(todo))
    done = 0
    for i, paper in enumerate(todo, 1):
        store.update_enrichment(client.enrich(paper))
        done += 1
        if i % 50 == 0:
            log.info("  enriched %d/%d", i, len(todo))
    if own:
        store.close()
    return done

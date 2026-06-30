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
    # arXiv publishes hundreds of papers/day, so "newest N" would only span days.
    # Instead we stratify by quarter: pull up to `max_per_period` per category per
    # quarter, giving even coverage across the whole window so velocity has a real
    # time axis. (This is a temporally-stratified sample; topic *shares* per quarter
    # are preserved, which is what the velocity/divergence trends rely on.)
    windows = _quarter_windows(start_date, end_date)
    max_per_period = settings.max_per_period
    log.info("Sampling %d quarters x %d categories, up to %d papers each",
             len(windows), len(settings.arxiv_categories), max_per_period)
    failures = 0
    for category in settings.arxiv_categories:
        for (qs, qe) in windows:
            try:
                batch: list[Paper] = list(
                    client.search_category(category, qs, qe, max_per_period)
                )
            except Exception as e:  # one window failing must not abort the run
                failures += 1
                log.warning("  %s %s..%s FAILED: %s", category, qs, qe, e)
                continue
            if batch:
                store.upsert_papers(batch)
            log.info("  %s %s..%s: +%d (total %d)", category, qs, qe,
                     len(batch), store.count_papers())
    if failures:
        log.warning("%d window(s) failed and were skipped", failures)

    # Recent top-up: an extra pull of the last N days so the "this week" lens has
    # complete data even if a category exceeds the quarterly cap in the current quarter.
    wcfg = settings.analysis.get("weekly", {}) if settings.analysis else {}
    topup_days = int(wcfg.get("recent_topup_days", 0)) if wcfg.get("enabled") else 0
    if topup_days > 0:
        tu_end = end_date
        tu_start = tu_end - dt.timedelta(days=topup_days)
        log.info("Recent top-up: last %d days (%s..%s) x %d categories",
                 topup_days, tu_start, tu_end, len(settings.arxiv_categories))
        for category in settings.arxiv_categories:
            try:
                batch = list(client.search_category(category, tu_start, tu_end, 2000))
            except Exception as e:  # never abort the run for the top-up
                log.warning("  top-up %s FAILED: %s", category, e)
                continue
            if batch:
                store.upsert_papers(batch)
            log.info("  top-up %s: +%d (total %d)", category, len(batch),
                     store.count_papers())

    # Optional extra paper source: OpenReview venues (added as papers).
    try:
        ingest_openreview(settings, store)
    except Exception as e:
        log.warning("OpenReview ingestion skipped: %s", e)

    # Optional non-paper source: lab/blog RSS (separate posts table).
    try:
        ingest_blogs(settings, store)
    except Exception as e:
        log.warning("Blog ingestion skipped: %s", e)

    if enrich:
        enrich_citations(settings, store)
        try:
            enrich_semantic_scholar(settings, store)
        except Exception as e:  # fully optional, never block ingestion
            log.warning("Semantic Scholar enrichment skipped: %s", e)

    total = store.count_papers()
    store.close()
    return total


def enrich_specific_citations(settings: Settings, papers: list[Paper]) -> int:
    """Targeted Semantic Scholar enrichment of a specific, SMALL set of papers.

    Enriching the full ~12.7k-paper corpus keyless hits S2's rate limits and the
    client's time budget before finishing, leaving recent/surfaced papers without a
    citation count. This enriches just the handful of papers that drive the foresight
    gaps (top divergence pairs + emerging quadrant + verified citation-flow borrowers),
    so real citation heat lands exactly where it's shown. Enriches in place; fail-soft.
    """
    cfg = settings.semantic_scholar
    if not cfg.get("enabled") or not papers:
        return 0
    from .semantic_scholar_client import SemanticScholarClient

    api_key = cfg.get("api_key")
    client = SemanticScholarClient(
        api_key=api_key,
        batch_size=200 if api_key else 50,
        request_delay=0.5 if api_key else 1.3,
    )
    try:
        return client.enrich(papers)
    except Exception as e:  # never block snapshot assembly
        log.warning("Targeted citation enrichment skipped: %s", e)
        return 0


def enrich_semantic_scholar(settings: Settings, store: Store | None = None) -> int:
    """Optional Semantic Scholar enrichment (TLDR, influential cites, venue, fields)."""
    cfg = settings.semantic_scholar
    if not cfg.get("enabled"):
        return 0
    from .semantic_scholar_client import SemanticScholarClient

    own = store is None
    store = store or Store(settings.path("db_path"))
    papers = store.get_papers()
    cap = int(cfg.get("max_enrich", 0))
    if cap:
        papers = papers[-cap:]  # most recent
    api_key = cfg.get("api_key")
    # Keyless pool is throttled: smaller batches + slower pacing succeed more often.
    client = SemanticScholarClient(
        api_key=api_key,
        batch_size=200 if api_key else 50,
        request_delay=0.5 if api_key else 1.3,
    )
    log.info("Semantic Scholar: enriching %d papers (%s)",
             len(papers), "keyed" if api_key else "keyless best-effort")
    n = client.enrich(papers)
    for p in papers:
        if (p.s2_tldr is not None or p.s2_influential_citations is not None or p.venue
                or p.cited_by_count is not None or p.referenced_works
                or any(a.openalex_id for a in p.authors)):
            store.update_s2_enrichment(p)
    log.info("Semantic Scholar: enriched %d papers", n)
    if own:
        store.close()
    return n


def ingest_openreview(settings: Settings, store: Store | None = None) -> int:
    """Add OpenReview venue papers (with review scores) to the cache."""
    cfg = settings.openreview
    if not cfg.get("enabled") or not cfg.get("venues"):
        return 0
    from .openreview_client import OpenReviewClient

    own = store is None
    store = store or Store(settings.path("db_path"))
    client = OpenReviewClient()
    added = 0
    for venue in cfg["venues"]:
        papers = client.fetch_venue(venue, int(cfg.get("max_per_venue", 400)))
        if papers:
            added += store.upsert_papers(papers)
    log.info("OpenReview: added %d papers", added)
    if own:
        store.close()
    return added


def ingest_blogs(settings: Settings, store: Store | None = None) -> int:
    """Pull lab/blog RSS posts into the posts table (capability signal)."""
    cfg = settings.blogs
    if not cfg.get("enabled") or not cfg.get("feeds"):
        return 0
    from .blog_rss_client import BlogRSSClient

    own = store is None
    store = store or Store(settings.path("db_path"))
    client = BlogRSSClient(max_per_feed=int(cfg.get("max_per_feed", 30)))
    posts = client.fetch(cfg["feeds"])
    n = store.upsert_posts(posts) if posts else 0
    log.info("Blogs: stored %d posts", n)
    if own:
        store.close()
    return n


def _quarter_windows(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Inclusive list of (quarter_start, quarter_end) covering [start, end]."""
    windows: list[tuple[dt.date, dt.date]] = []
    q_start_month = ((start.month - 1) // 3) * 3 + 1
    y, m = start.year, q_start_month
    while dt.date(y, m, 1) <= end:
        qs = dt.date(y, m, 1)
        nm, ny = (m + 3, y) if m + 3 <= 12 else (m + 3 - 12, y + 1)
        qe = dt.date(ny, nm, 1) - dt.timedelta(days=1)
        windows.append((max(qs, start), min(qe, end)))
        y, m = ny, nm
    return windows


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

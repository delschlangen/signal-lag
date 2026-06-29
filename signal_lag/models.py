"""Shared data structures used across ingestion and analysis."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class Author:
    name: str
    affiliation: str | None = None
    openalex_id: str | None = None


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    published: dt.date
    updated: dt.date | None
    categories: list[str] = field(default_factory=list)
    authors: list[Author] = field(default_factory=list)
    primary_category: str | None = None
    pdf_url: str | None = None

    # OpenAlex enrichment (populated later; may stay empty offline).
    openalex_id: str | None = None
    cited_by_count: int | None = None
    # List of {year: int, count: int} citation counts by year.
    counts_by_year: list[dict] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    # arXiv ids this paper cites (outgoing bibliography, from Semantic Scholar
    # references; OpenAlex is unreachable in CI) — for citation-flow checks.
    referenced_works: list[str] = field(default_factory=list)

    # Semantic Scholar enrichment (optional, fail-soft).
    s2_tldr: str | None = None
    s2_influential_citations: int | None = None
    venue: str | None = None
    fields_of_study: list[str] = field(default_factory=list)

    # Provenance: which source(s) this record came from (e.g. "arxiv", "openreview").
    source: str = "arxiv"
    # OpenReview-style review signal (mean rating), when available.
    review_score: float | None = None

    @property
    def quarter(self) -> str:
        q = (self.published.month - 1) // 3 + 1
        return f"{self.published.year}Q{q}"

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

    @property
    def quarter(self) -> str:
        q = (self.published.month - 1) // 3 + 1
        return f"{self.published.year}Q{q}"

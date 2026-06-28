"""OpenReview ingestion — venue papers + peer-review scores.

Pulls submissions for configured venues (e.g. ICLR/NeurIPS) from the OpenReview
API v2, as real Paper records (source="openreview") so they flow through the
normal embedding/tagging/velocity pipeline. When review ratings are available
they're averaged into ``review_score`` — a quality/heat signal papers-only
sources don't have.

Fail-soft and defensive: OpenReview's schema varies by venue, so anything that
doesn't parse is skipped rather than raised. arXiv ids found in the note are used
so records dedup against the arXiv pull; otherwise an ``openreview:<id>`` key is
used.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import requests

from ..models import Author, Paper

log = logging.getLogger("signal_lag.openreview")

NOTES_URL = "https://api2.openreview.net/notes"
ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")


def _text(field):
    """OpenReview v2 wraps values as {'value': ...}; tolerate both forms."""
    if isinstance(field, dict):
        return field.get("value")
    return field


class OpenReviewClient:
    def __init__(self, request_delay: float = 1.0, session: requests.Session | None = None):
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "signal-lag/0.1"})

    def _get(self, params: dict) -> dict | None:
        for attempt, wait in enumerate([0, 5, 15]):
            if wait:
                time.sleep(wait)
            try:
                resp = self.session.get(NOTES_URL, params=params, timeout=45)
                if resp.status_code == 429:
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                log.warning("OpenReview error: %s", e)
                continue
        return None

    def fetch_venue(self, venue_id: str, max_results: int = 400) -> list[Paper]:
        papers: list[Paper] = []
        offset = 0
        page = 200
        while len(papers) < max_results:
            data = self._get(
                {"content.venueid": venue_id, "limit": page, "offset": offset,
                 "details": "directReplies"}
            )
            time.sleep(self.request_delay)
            notes = (data or {}).get("notes", [])
            if not notes:
                break
            for note in notes:
                p = self._note_to_paper(note, venue_id)
                if p:
                    papers.append(p)
            offset += len(notes)
            if len(notes) < page:
                break
        log.info("  OpenReview %s: %d papers", venue_id, len(papers))
        return papers[:max_results]

    def _note_to_paper(self, note: dict, venue_id: str) -> Paper | None:
        content = note.get("content") or {}
        title = _text(content.get("title"))
        abstract = _text(content.get("abstract"))
        if not title or not abstract:
            return None

        # Date from note creation (ms epoch).
        published = None
        ts = note.get("cdate") or note.get("tcdate")
        if ts:
            try:
                published = dt.datetime.utcfromtimestamp(ts / 1000).date()
            except (ValueError, OSError):
                published = None
        if published is None:
            return None

        # arXiv id if linkable (so it dedups against the arXiv pull).
        blob = " ".join(str(_text(content.get(k)) or "") for k in ("pdf", "code", "_bibtex", "html"))
        m = ARXIV_RE.search(blob)
        arxiv_id = m.group(1) if m else f"openreview:{note.get('id')}"

        authors = [Author(name=a) for a in (_text(content.get("authors")) or []) if a]
        review_score = self._mean_rating(note)

        return Paper(
            arxiv_id=arxiv_id,
            title=" ".join(str(title).split()),
            abstract=" ".join(str(abstract).split()),
            published=published,
            updated=None,
            categories=[],
            authors=authors,
            primary_category=None,
            source="openreview",
            venue=venue_id,
            review_score=review_score,
        )

    @staticmethod
    def _mean_rating(note: dict) -> float | None:
        replies = (note.get("details") or {}).get("directReplies", []) or []
        ratings: list[float] = []
        for r in replies:
            c = r.get("content") or {}
            val = _text(c.get("rating")) or _text(c.get("overall_rating"))
            if val is None:
                continue
            m = re.match(r"\s*(\d+(?:\.\d+)?)", str(val))
            if m:
                ratings.append(float(m.group(1)))
        return round(sum(ratings) / len(ratings), 2) if ratings else None

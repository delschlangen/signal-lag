"""arXiv Atom API client with polite pagination and exponential backoff.

arXiv asks clients to keep to roughly one request every three seconds and to
back off on errors. We honor both: a fixed inter-request delay plus an
exponential backoff schedule on 429/503/transient failures.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from ..models import Author, Paper

log = logging.getLogger("signal_lag.arxiv")

ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivClient:
    def __init__(
        self,
        page_size: int = 200,
        request_delay: float = 3.0,
        backoff_schedule: list[float] | None = None,
        session: requests.Session | None = None,
    ):
        self.page_size = page_size
        self.request_delay = request_delay
        self.backoff_schedule = backoff_schedule or [2, 4, 8, 16]
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "signal-lag/0.1 (research foresight tool)"})

    def _get(self, params: dict) -> str:
        last_err: Exception | None = None
        for attempt, wait in enumerate([0, *self.backoff_schedule]):
            if wait:
                log.warning("arXiv retry in %ss (attempt %d)", wait, attempt)
                time.sleep(wait)
            try:
                resp = self.session.get(ARXIV_ENDPOINT, params=params, timeout=60)
                if resp.status_code in (429, 503):
                    last_err = RuntimeError(f"arXiv {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:  # network/transient
                last_err = e
                continue
        raise RuntimeError(f"arXiv request failed after retries: {last_err}")

    def search_category(
        self,
        category: str,
        start_date: dt.date,
        end_date: dt.date,
        max_results: int,
    ) -> Iterator[Paper]:
        """Yield up to ``max_results`` papers in a category within [start, end].

        Uses arXiv's server-side ``submittedDate`` range filter, sorted newest
        first, with a client-side date check as a backstop. Callers cap the count
        per window so that pulling across many windows yields even time coverage.
        """
        s = start_date.strftime("%Y%m%d0000")
        e = end_date.strftime("%Y%m%d2359")
        date_filter = f"submittedDate:[{s} TO {e}]"
        fetched = 0
        start = 0
        while fetched < max_results:
            page = min(self.page_size, max_results - fetched)
            params = {
                "search_query": f"cat:{category} AND {date_filter}",
                "start": start,
                "max_results": page,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            xml = self._get(params)
            entries = list(self._parse_entries(xml))
            if not entries:
                break
            for paper in entries:
                if paper.published < start_date:
                    return  # past the window; newest-first means we're done
                if paper.published <= end_date:
                    yield paper
                    fetched += 1
                    if fetched >= max_results:
                        return
            start += len(entries)
            time.sleep(self.request_delay)

    @staticmethod
    def _parse_entries(xml: str) -> Iterator[Paper]:
        root = ET.fromstring(xml)
        for entry in root.findall(f"{ATOM}entry"):
            raw_id = entry.findtext(f"{ATOM}id", "")
            # e.g. http://arxiv.org/abs/2401.01234v2 -> 2401.01234
            arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]
            published = _parse_date(entry.findtext(f"{ATOM}published"))
            updated = _parse_date(entry.findtext(f"{ATOM}updated"))
            if published is None:
                continue

            authors = []
            for a in entry.findall(f"{ATOM}author"):
                name = a.findtext(f"{ATOM}name", "").strip()
                aff = a.findtext(f"{ARXIV_NS}affiliation")
                if name:
                    authors.append(Author(name=name, affiliation=aff))

            cats = [
                c.attrib.get("term", "")
                for c in entry.findall(f"{ATOM}category")
                if c.attrib.get("term")
            ]
            prim = entry.find(f"{ARXIV_NS}primary_category")
            primary = prim.attrib.get("term") if prim is not None else (cats[0] if cats else None)

            pdf_url = None
            for link in entry.findall(f"{ATOM}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")

            yield Paper(
                arxiv_id=arxiv_id,
                title=" ".join((entry.findtext(f"{ATOM}title") or "").split()),
                abstract=" ".join((entry.findtext(f"{ATOM}summary") or "").split()),
                published=published,
                updated=updated,
                categories=cats,
                authors=authors,
                primary_category=primary,
                pdf_url=pdf_url,
            )


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None

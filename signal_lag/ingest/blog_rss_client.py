"""Lab/blog RSS ingestion — a capability-leading signal.

Capability news often breaks on lab blogs (Anthropic, OpenAI, DeepMind, …)
before it shows up as papers, so tracking post volume sharpens the
capability-vs-safety "lag" thesis.

These are posts, not papers, so they're kept in a separate ``posts`` table and
surfaced in a dedicated dashboard panel — they do NOT enter paper velocity.
Entirely fail-soft: a feed being down just omits its posts.
"""
from __future__ import annotations

import datetime as dt
import logging

log = logging.getLogger("signal_lag.blogs")


def _parse_date(entry) -> dt.date | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                return dt.date(t.tm_year, t.tm_mon, t.tm_mday)
            except (ValueError, TypeError):
                continue
    return None


class BlogRSSClient:
    def __init__(self, max_per_feed: int = 30):
        self.max_per_feed = max_per_feed

    def fetch(self, feeds: list[dict]) -> list[dict]:
        """feeds = [{name, url}]. Returns post dicts (fail-soft per feed)."""
        try:
            import feedparser
        except ImportError:
            log.warning("feedparser not installed; skipping blog ingestion")
            return []

        posts: list[dict] = []
        for feed in feeds:
            name, url = feed.get("name", "?"), feed.get("url")
            if not url:
                continue
            try:
                parsed = feedparser.parse(url)
            except Exception as e:
                log.warning("feed %s failed: %s", name, e)
                continue
            for entry in parsed.entries[: self.max_per_feed]:
                published = _parse_date(entry)
                summary = (entry.get("summary") or "").strip()
                # Strip rudimentary HTML tags from summaries.
                summary = _strip_html(summary)[:1000]
                posts.append(
                    {
                        "id": entry.get("id") or entry.get("link") or f"{name}:{entry.get('title')}",
                        "source": name,
                        "title": (entry.get("title") or "").strip(),
                        "summary": summary,
                        "url": entry.get("link"),
                        "published": published.isoformat() if published else None,
                    }
                )
            log.info("  blog %s: +%d posts", name, len(parsed.entries[: self.max_per_feed]))
        return posts


def _strip_html(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", text).replace("&nbsp;", " ").strip()

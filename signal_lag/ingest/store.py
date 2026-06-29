"""SQLite cache for ingested papers and enrichment.

The store is idempotent: re-running ingestion upserts by ``arxiv_id`` so repeated
pulls never duplicate rows and enrichment can be filled in over multiple passes.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from ..models import Author, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    abstract          TEXT NOT NULL,
    published         TEXT NOT NULL,
    updated           TEXT,
    primary_category  TEXT,
    categories        TEXT,          -- json list
    pdf_url           TEXT,
    -- OpenAlex enrichment (nullable)
    openalex_id       TEXT,
    cited_by_count    INTEGER,
    counts_by_year    TEXT,          -- json list of {year,count}
    institutions      TEXT,          -- json list
    referenced_works  TEXT,          -- json list of OpenAlex work ids (outgoing refs)
    enriched_at       TEXT,
    -- Semantic Scholar enrichment (nullable)
    s2_tldr           TEXT,
    s2_influential    INTEGER,
    venue             TEXT,
    fields_of_study   TEXT,          -- json list
    -- Provenance + review signal
    source            TEXT DEFAULT 'arxiv',
    review_score      REAL
);

CREATE TABLE IF NOT EXISTS authors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    affiliation   TEXT,
    openalex_id   TEXT,
    FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
);
CREATE INDEX IF NOT EXISTS idx_authors_arxiv ON authors(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);

-- Topic tags (taxonomy + cluster assignments) live here so analysis is cacheable.
CREATE TABLE IF NOT EXISTS topic_tags (
    arxiv_id   TEXT NOT NULL,
    source     TEXT NOT NULL,        -- "taxonomy" | "cluster"
    topic_key  TEXT NOT NULL,
    score      REAL,
    PRIMARY KEY (arxiv_id, source, topic_key)
);
CREATE INDEX IF NOT EXISTS idx_tags_topic ON topic_tags(source, topic_key);

-- Lab/blog posts (capability-leading signal); kept separate from papers.
CREATE TABLE IF NOT EXISTS posts (
    id         TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    title      TEXT,
    summary    TEXT,
    url        TEXT,
    published  TEXT
);
CREATE INDEX IF NOT EXISTS idx_posts_published ON posts(published);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns missing from an older cache (CREATE IF NOT EXISTS won't)."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(papers)")}
        adds = {
            "s2_tldr": "TEXT",
            "s2_influential": "INTEGER",
            "venue": "TEXT",
            "fields_of_study": "TEXT",
            "source": "TEXT DEFAULT 'arxiv'",
            "review_score": "REAL",
            "referenced_works": "TEXT",
        }
        for name, decl in adds.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE papers ADD COLUMN {name} {decl}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ writes
    def upsert_papers(self, papers: Iterable[Paper]) -> int:
        n = 0
        for p in papers:
            self.conn.execute(
                """
                INSERT INTO papers
                    (arxiv_id, title, abstract, published, updated,
                     primary_category, categories, pdf_url, source, review_score)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title=excluded.title,
                    abstract=excluded.abstract,
                    published=excluded.published,
                    updated=excluded.updated,
                    primary_category=excluded.primary_category,
                    categories=excluded.categories,
                    pdf_url=excluded.pdf_url,
                    source=excluded.source,
                    review_score=excluded.review_score
                """,
                (
                    p.arxiv_id,
                    p.title,
                    p.abstract,
                    p.published.isoformat(),
                    p.updated.isoformat() if p.updated else None,
                    p.primary_category,
                    json.dumps(p.categories),
                    p.pdf_url,
                    p.source,
                    p.review_score,
                ),
            )
            # Refresh authors for this paper.
            self.conn.execute("DELETE FROM authors WHERE arxiv_id=?", (p.arxiv_id,))
            for a in p.authors:
                self.conn.execute(
                    "INSERT INTO authors (arxiv_id, name, affiliation, openalex_id)"
                    " VALUES (?,?,?,?)",
                    (p.arxiv_id, a.name, a.affiliation, a.openalex_id),
                )
            n += 1
        self.conn.commit()
        return n

    def update_enrichment(self, paper: Paper) -> None:
        self.conn.execute(
            """
            UPDATE papers SET
                openalex_id=?, cited_by_count=?, counts_by_year=?,
                institutions=?, referenced_works=?, enriched_at=?
            WHERE arxiv_id=?
            """,
            (
                paper.openalex_id,
                paper.cited_by_count,
                json.dumps(paper.counts_by_year),
                json.dumps(paper.institutions),
                json.dumps(paper.referenced_works),
                dt.datetime.utcnow().isoformat(timespec="seconds"),
                paper.arxiv_id,
            ),
        )
        # OpenAlex author ids are captured onto paper.authors during enrich; persist them
        # (the authors rows were written id-less at upsert time). Needed for #4 migration.
        for a in paper.authors:
            if a.openalex_id:
                self.conn.execute(
                    "UPDATE authors SET openalex_id=? "
                    "WHERE arxiv_id=? AND name=? AND openalex_id IS NULL",
                    (a.openalex_id, paper.arxiv_id, a.name),
                )
        self.conn.commit()

    def update_s2_enrichment(self, paper: Paper) -> None:
        self.conn.execute(
            "UPDATE papers SET s2_tldr=?, s2_influential=?, venue=?, fields_of_study=?"
            " WHERE arxiv_id=?",
            (
                paper.s2_tldr,
                paper.s2_influential_citations,
                paper.venue,
                json.dumps(paper.fields_of_study),
                paper.arxiv_id,
            ),
        )
        self.conn.commit()

    def replace_tags(self, source: str, rows: Iterable[tuple[str, str, float]]) -> None:
        """rows = (arxiv_id, topic_key, score). Replaces all tags for `source`."""
        self.conn.execute("DELETE FROM topic_tags WHERE source=?", (source,))
        self.conn.executemany(
            "INSERT OR REPLACE INTO topic_tags (arxiv_id, source, topic_key, score)"
            " VALUES (?,?,?,?)",
            [(aid, source, key, score) for (aid, key, score) in rows],
        )
        self.conn.commit()

    def upsert_posts(self, posts: Iterable[dict]) -> int:
        rows = [
            (p["id"], p["source"], p.get("title"), p.get("summary"),
             p.get("url"), p.get("published"))
            for p in posts
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO posts (id, source, title, summary, url, published)"
            " VALUES (?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def get_posts(self, limit: int = 0) -> list[dict]:
        q = "SELECT * FROM posts ORDER BY published DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(q).fetchall()]

    # ------------------------------------------------------------------- reads
    def count_papers(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"]

    def get_papers(self, limit: int | None = None) -> list[Paper]:
        q = "SELECT * FROM papers ORDER BY published"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = self.conn.execute(q).fetchall()
        papers = [self._row_to_paper(r) for r in rows]
        # attach authors
        by_id = {p.arxiv_id: p for p in papers}
        for ar in self.conn.execute("SELECT * FROM authors").fetchall():
            p = by_id.get(ar["arxiv_id"])
            if p is not None:
                p.authors.append(
                    Author(
                        name=ar["name"],
                        affiliation=ar["affiliation"],
                        openalex_id=ar["openalex_id"],
                    )
                )
        return papers

    def papers_needing_enrichment(self, limit: int = 0) -> list[Paper]:
        q = "SELECT * FROM papers WHERE enriched_at IS NULL ORDER BY published DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [self._row_to_paper(r) for r in self.conn.execute(q).fetchall()]

    def get_tags(self, source: str) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {}
        for r in self.conn.execute(
            "SELECT arxiv_id, topic_key, score FROM topic_tags WHERE source=?",
            (source,),
        ).fetchall():
            out.setdefault(r["arxiv_id"], []).append((r["topic_key"], r["score"]))
        return out

    @staticmethod
    def _row_to_paper(r: sqlite3.Row) -> Paper:
        return Paper(
            arxiv_id=r["arxiv_id"],
            title=r["title"],
            abstract=r["abstract"],
            published=dt.date.fromisoformat(r["published"]),
            updated=dt.date.fromisoformat(r["updated"]) if r["updated"] else None,
            primary_category=r["primary_category"],
            categories=json.loads(r["categories"]) if r["categories"] else [],
            pdf_url=r["pdf_url"],
            openalex_id=r["openalex_id"],
            cited_by_count=r["cited_by_count"],
            counts_by_year=json.loads(r["counts_by_year"]) if r["counts_by_year"] else [],
            institutions=json.loads(r["institutions"]) if r["institutions"] else [],
            referenced_works=json.loads(r["referenced_works"]) if _get(r, "referenced_works") else [],
            s2_tldr=_get(r, "s2_tldr"),
            s2_influential_citations=_get(r, "s2_influential"),
            venue=_get(r, "venue"),
            fields_of_study=json.loads(r["fields_of_study"]) if _get(r, "fields_of_study") else [],
            source=_get(r, "source") or "arxiv",
            review_score=_get(r, "review_score"),
        )


def _get(row: sqlite3.Row, key: str):
    """Safe column access (older caches may lack newer columns)."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return None

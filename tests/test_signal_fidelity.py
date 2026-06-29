"""Tests for the signal-fidelity upgrades: citation-flow verification, author
migration, the LLM limitation-classifier merge, and the live-context wiring.

All offline — the LLM/web passes are stubbed so the merge logic is exercised
without a network or an API key.
"""
import datetime as dt

import numpy as np

from signal_lag.analysis import authors, citation_flow, foresight, sentiment
from signal_lag.config import Taxonomy, Topic
from signal_lag.models import Author, Paper


def _taxonomy():
    return Taxonomy(
        safety_topics=[Topic("saf", "Safety", "safety", ["s"])],
        capability_topics=[Topic("cap", "Capability", "capability", ["c"])],
        pairings=[],
    )


def _p(aid, year, q, oa=None, refs=None, auths=None):
    month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    p = Paper(aid, f"title {aid}", "abstract", dt.date(year, month, 15), None)
    p.openalex_id = oa
    p.referenced_works = refs or []
    p.authors = auths or []
    return p


# --------------------------------------------------------------------- #2 flow
def test_citation_flow_detects_verified_borrowing():
    # Capability paper C cites safety paper S by arXiv id (Semantic Scholar references).
    s = _p("S", 2024, 1)
    c = _p("C", 2024, 2, refs=["S", "9999.99999"])
    tax_tags = {"S": [("saf", 1.0)], "C": [("cap", 1.0)]}
    out = citation_flow.citation_flow([s, c], tax_tags, _taxonomy())
    assert out["n_safety_indexed"] == 1
    assert out["verified_ids"] == ["C"]
    b = out["verified_borrowers"][0]
    assert b["arxiv_id"] == "C"
    assert b["n_cited_safety"] == 1
    assert b["cited_safety"][0]["arxiv_id"] == "S"


def test_citation_flow_absence_is_inconclusive():
    # Capability paper cites only an out-of-corpus work -> NOT flagged (no false claim).
    s = _p("S", 2024, 1)
    c = _p("C", 2024, 2, refs=["9999.99999"])
    tax_tags = {"S": [("saf", 1.0)], "C": [("cap", 1.0)]}
    out = citation_flow.citation_flow([s, c], tax_tags, _taxonomy())
    assert out["verified_borrowers"] == []
    assert out["n_candidates_checked"] == 1   # it WAS checked; just no verified link


# ------------------------------------------------------------------- #4 authors
def test_author_migration_flags_capability_to_safety():
    a = Author("Dr X", openalex_id="https://openalex.org/A1")
    # Prior capability-only history, then a recent safety entry.
    papers = [
        _p("c1", 2023, 1, auths=[a]),
        _p("c2", 2023, 2, auths=[a]),
        _p("s1", 2024, 4, auths=[a]),
    ]
    tax_tags = {"c1": [("cap", 1.0)], "c2": [("cap", 1.0)], "s1": [("saf", 1.0)]}
    out = authors.author_migration(papers, tax_tags, _taxonomy(),
                                   min_history=2, recent_window_periods=2)
    assert out["available"] is True
    assert out["n_migrants"] == 1
    assert out["migrants"][0]["entered_safety_topics"] == ["Safety"]


def test_author_migration_ignores_existing_safety_authors():
    a = Author("Dr Y", openalex_id="https://openalex.org/A2")
    papers = [
        _p("s0", 2023, 1, auths=[a]),   # already did safety historically
        _p("c1", 2023, 2, auths=[a]),
        _p("s1", 2024, 4, auths=[a]),
    ]
    tax_tags = {"s0": [("saf", 1.0)], "c1": [("cap", 1.0)], "s1": [("saf", 1.0)]}
    out = authors.author_migration(papers, tax_tags, _taxonomy(),
                                   min_history=2, recent_window_periods=2)
    assert out["n_migrants"] == 0   # not a migration: prior history already had safety


def test_author_migration_empty_without_author_ids():
    papers = [_p("c1", 2023, 1, auths=[Author("No Id")])]
    out = authors.author_migration(papers, {"c1": [("cap", 1.0)]}, _taxonomy())
    assert out["available"] is False
    assert out["migrants"] == []


# ------------------------------------------------------- #1 sentiment LLM merge
def test_limitation_classifier_flips_embedding_false_positive(monkeypatch):
    # Two papers both embedding-flagged critical; the LLM says only one really is.
    from signal_lag.analysis import llm

    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        return '{"labels": [{"arxiv_id": "p1", "is_limitation_focused": false},' \
               ' {"arxiv_id": "p2", "is_limitation_focused": true}]}'

    monkeypatch.setattr(llm, "call_claude", fake_call)
    labels = llm.classify_limitation_focused(
        [{"arxiv_id": "p1", "title": "t", "abstract": "we overcome the failures of X"},
         {"arxiv_id": "p2", "title": "t", "abstract": "X does not work; an audit"}],
        api_key="present",
    )
    assert labels == {"p1": False, "p2": True}

    # Apply the merge the way runner does: snap the false positive below threshold.
    ids = ["p1", "p2"]
    crit = np.array([0.5, 0.5], dtype=np.float32)   # both above 0.22
    paper_critical = {"p1": True, "p2": True}
    for aid, is_crit in labels.items():
        if not is_crit and paper_critical[aid]:
            paper_critical[aid] = False
            crit[ids.index(aid)] = 0.0
    assert paper_critical == {"p1": False, "p2": True}
    assert crit[0] == 0.0 and crit[1] == 0.5


def test_classifier_failsoft_without_key():
    assert sentiment.critical_scores(np.zeros((0, 3)), None).shape == (0,)
    # No key -> empty dict, caller keeps embedding flags.
    from signal_lag.analysis import llm
    assert llm.classify_limitation_focused([{"arxiv_id": "p"}], api_key=None) == {}


# ------------------------------------------------------- #3 live-context wiring
def test_synthesize_includes_live_brief(monkeypatch):
    captured = {}

    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        captured["user"] = user
        return '{"risks": []}'

    monkeypatch.setattr(foresight.llm, "call_claude", fake_call)
    foresight._synthesize_risks(
        {"what_changed_this_week": {}}, "standing context", "key", "m", 2,
        live_context="2026-06-01: the law took effect today.",
    )
    assert "LIVE_WEB_BRIEF" in captured["user"]
    assert "took effect today" in captured["user"]


def test_fetch_live_context_failsoft_without_key():
    assert foresight.fetch_live_context("ctx", {}, api_key=None) is None


# ----------------------------------------------------- ingestion round-trip (#2/#4)
def test_store_persists_referenced_works_and_author_ids(tmp_path):
    from signal_lag.ingest.store import Store

    p = _p("a1", 2024, 1, auths=[Author("Dr X"), Author("Dr Y")])
    store = Store(tmp_path / "db.sqlite")
    store.upsert_papers([p])              # authors written id-less
    # Enrichment fills referenced_works + author ids; update_enrichment must persist both.
    p.referenced_works = ["https://openalex.org/W1", "https://openalex.org/W2"]
    p.openalex_id = "https://openalex.org/Wpaper"
    p.cited_by_count = 3
    p.authors[0].openalex_id = "https://openalex.org/A1"
    store.update_enrichment(p)

    got = store.get_papers()[0]
    store.close()
    assert got.referenced_works == ["https://openalex.org/W1", "https://openalex.org/W2"]
    ids = {a.name: a.openalex_id for a in got.authors}
    assert ids["Dr X"] == "https://openalex.org/A1"
    assert ids["Dr Y"] is None          # untouched author stays null


def test_s2_enrichment_persists_refs_counts_and_author_ids(tmp_path):
    from signal_lag.ingest.store import Store

    p = _p("a1", 2024, 1, auths=[Author("Dr X"), Author("Dr Y")])
    store = Store(tmp_path / "db.sqlite")
    store.upsert_papers([p])
    # Semantic Scholar fills citation count, references (arXiv ids), and author ids.
    p.cited_by_count = 12
    p.referenced_works = ["2301.00001", "2301.00002"]
    p.s2_tldr = "tldr"
    p.authors[0].openalex_id = "S2-AUTHOR-1"
    store.update_s2_enrichment(p)

    got = store.get_papers()[0]
    store.close()
    assert got.cited_by_count == 12
    assert got.referenced_works == ["2301.00001", "2301.00002"]
    ids = {a.name: a.openalex_id for a in got.authors}
    assert ids["Dr X"] == "S2-AUTHOR-1"
    assert ids["Dr Y"] is None

"""Tests for the signal-fidelity upgrades: citation-flow verification, author
migration, the LLM limitation-classifier merge, and the live-context wiring.

All offline — the LLM/web passes are stubbed so the merge logic is exercised
without a network or an API key.
"""
import datetime as dt
import json

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


# ------------------------------------------------------- harm/misuse dual-use lens
def test_harm_tagging_via_generic_centroids():
    from signal_lag.analysis import taxonomy as taxmod
    from signal_lag.config import Taxonomy, Topic

    vocab = ["cyber", "bioweapon", "influence"]

    class FakeEmbedder:
        def embed(self, texts):
            out = []
            for t in texts:
                v = np.zeros(len(vocab), dtype=np.float32)
                for i, w in enumerate(vocab):
                    if w in t:
                        v[i] = 1.0
                nrm = np.linalg.norm(v)
                out.append(v / nrm if nrm else v)
            return np.vstack(out)

    harm = [Topic("cyber_offense", "Cyber", "harm", ["cyber exploitation"]),
            Topic("bio", "Bio", "harm", ["bioweapon uplift"]),
            Topic("influence_ops", "Influence", "harm", ["influence operations"])]
    tax = Taxonomy(safety_topics=[], capability_topics=[], pairings=[],
                   harm_topics=harm, tag_threshold=0.5, max_tags_per_paper=3)
    emb = FakeEmbedder()
    cents = taxmod.build_topic_centroids_from(tax.harm_topics, emb)
    assert set(cents) == {"cyber_offense", "bio", "influence_ops"}
    # harm topics must NOT be in the research all_topics list
    assert tax.all_topics == []

    pv = emb.embed(["a paper about cyber attacks", "unrelated text with no signal"])
    rows = taxmod.tag_papers(["p1", "p2"], pv, cents, tax)
    tagged = {(a, k) for a, k, _ in rows}
    assert ("p1", "cyber_offense") in tagged
    assert not any(a == "p2" for a, k, _ in rows)   # no harm matched -> untagged


# ----------------------------------------------------- risk scoring + register
def test_attach_scores_clamps_and_computes_priority():
    risks = foresight._attach_scores([
        {"risk": "R1", "severity": "5", "likelihood": 4, "exposure": 3,
         "trajectory": "ACCELERATING"},
        {"risk": "R2", "severity": 99, "likelihood": None, "trajectory": "weird"},
    ])
    assert (risks[0]["severity"], risks[0]["likelihood"], risks[0]["exposure"]) == (5, 4, 3)
    assert risks[0]["trajectory"] == "accelerating"
    assert risks[0]["priority"] == 20                       # 5 × 4
    # garbage clamps to [1,5] and defaults (3); unknown trajectory -> steady
    assert risks[1]["severity"] == 5 and risks[1]["likelihood"] == 3
    assert risks[1]["trajectory"] == "steady" and risks[1]["priority"] == 15


def test_generate_scenarios_parses_and_uses_top_risks(monkeypatch):
    captured = {}

    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        captured["user"] = user
        return ('{"scenarios": [{"title": "Agentic fraud wave", "horizon": "12 months",'
                ' "estimative_likelihood": "likely", "narrative": "n",'
                ' "drivers": ["d"], "leading_indicators": ["li"], "branch_points": ["bp"],'
                ' "candidate_mitigations": ["m"], "linked_risks": ["R-high"]}]}')

    monkeypatch.setattr(foresight.llm, "call_claude", fake_call)
    risks = [{"risk": "R-low", "priority": 4}, {"risk": "R-high", "priority": 20}]
    scen = foresight.generate_scenarios(risks, "ctx", api_key="k", max_scenarios=2)
    assert scen and scen[0]["title"] == "Agentic fraud wave"
    assert "R-high" in captured["user"]            # top-priority risk fed to the pass


def test_generate_scenarios_failsoft_without_key():
    assert foresight.generate_scenarios([{"risk": "x", "priority": 5}], "c", api_key=None) is None
    assert foresight.generate_scenarios([], "c", api_key="k") is None


def test_attach_explanations_top_n_only(monkeypatch):
    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        return ('{"technical_evidence": "te", "societal_evidence": "se", "the_gap": "g",'
                ' "skepticism": "sk", "bottom_line": "bl"}')

    monkeypatch.setattr(foresight.llm, "call_claude", fake_call)
    risks = [{"risk": "low", "priority": 4, "source_arxiv_ids": []},
             {"risk": "high", "priority": 20, "source_arxiv_ids": ["a1"]}]
    out = foresight.attach_explanations(risks, {"a1": {"title": "T", "abstract": "A"}},
                                        "ctx", api_key="k", max_explainers=1)
    hi = next(r for r in out if r["risk"] == "high")
    lo = next(r for r in out if r["risk"] == "low")
    assert hi["plain_explanation"]["bottom_line"] == "bl"   # top-priority explained
    assert "plain_explanation" not in lo                    # below the cap, untouched


def test_attach_explanations_failsoft_without_key():
    out = foresight.attach_explanations([{"risk": "x", "priority": 5}], {}, "c", api_key=None)
    assert "plain_explanation" not in out[0]


def test_fetch_incidents_filters_unverifiable(monkeypatch):
    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        return ('{"incidents": ['
                '{"title": "A", "date": "2026-01", "harm_key": "cyber_offense",'
                ' "source_url": "http://x", "summary": "s"},'
                '{"title": "B", "date": "2026-02", "harm_key": "NOTAKEY",'
                ' "source_url": "http://y"},'
                '{"title": "C", "harm_key": "cyber_offense", "source_url": "http://z"}]}')

    monkeypatch.setattr(foresight.llm, "call_claude", fake_call)
    out = foresight.fetch_incidents([{"key": "cyber_offense", "label": "Cyber"}], api_key="k")
    assert len(out) == 1 and out[0]["title"] == "A"   # bad key + missing date dropped


def test_fetch_incidents_failsoft():
    assert foresight.fetch_incidents([{"key": "k", "label": "L"}], api_key=None) is None
    assert foresight.fetch_incidents([], api_key="k") is None


def test_augment_incidents_benchmark_quadrants(monkeypatch):
    from signal_lag import snapshot as snap_mod
    from signal_lag.config import Settings

    monkeypatch.setattr(foresight, "fetch_incidents", lambda *a, **k: [
        {"title": "I1", "date": "2026-01", "harm_key": "cyber_offense", "source_url": "u"}])
    snap = {"harm": {"vectors": [
        {"key": "cyber_offense", "label": "Cyber", "change_pct": 10, "n_tagged": 100},
        {"key": "bio", "label": "Bio", "change_pct": 10, "n_tagged": 50},
        {"key": "calm", "label": "Calm", "change_pct": 0, "n_tagged": 20},
    ]}}
    settings = Settings(raw={"analysis": {"incidents": {"enabled": True, "rising_pct": 4},
                                          "foresight": {}}})
    out = snap_mod.augment_incidents(settings, snap)
    bench = {b["key"]: b["quadrant"] for b in out["incidents"]["benchmark"]}
    assert bench["cyber_offense"] == "materializing"    # rising + incident
    assert bench["bio"] == "foresight lead"             # rising, no incident
    assert bench["calm"] == "quiet"                     # flat, no incident
    assert out["incidents"]["n"] == 1


def test_risk_register_upsert_and_idempotency(tmp_path):
    from signal_lag import snapshot as snap_mod

    def snap(date):
        r = foresight._attach_scores([{"risk": "Persistent risk", "severity": 4,
                                       "likelihood": 4, "exposure": 2, "trajectory": "steady"}])
        return {"meta": {"refreshed_at": date},
                "analysis": {"foresight_gap": {"risks": r}}, "weekly": {}}

    path = tmp_path / "reg.json"
    snap_mod.append_risk_register(snap("2026-06-22"), path)
    snap_mod.append_risk_register(snap("2026-06-22"), path)   # same date -> no double count
    reg = json.loads(path.read_text())
    assert len(reg) == 1 and reg[0]["n_appearances"] == 1
    assert reg[0]["latest"]["priority"] == 16                # 4 × 4
    assert len(reg[0]["history"]) == 1

    snap_mod.append_risk_register(snap("2026-06-29"), path)   # new date -> increment
    reg = json.loads(path.read_text())
    assert reg[0]["n_appearances"] == 2 and len(reg[0]["history"]) == 2
    assert reg[0]["first_seen"] == "2026-06-22" and reg[0]["last_seen"] == "2026-06-29"


def test_digest_includes_harm_vectors_and_filters_thin_ones():
    snap = {"label_map": {}, "harm": {"vectors": [
        {"key": "cyber_offense", "label": "Cyber", "n_tagged": 5, "change_pct": 120.0,
         "recent_per_qtr": 4.0, "direction": "acceleration",
         "rep_papers": [{"title": "Autonomous exploit paper"}]},
        {"key": "rare", "label": "Rare", "n_tagged": 1, "change_pct": 0.0,
         "recent_per_qtr": 0.0, "direction": "steady", "rep_papers": []}]}}
    diff = {"first_run": True, "prev_date": None, "new_alerts": [],
            "new_accelerations": [], "new_sleepers": []}
    d = foresight.build_signal_digest(snap, diff)
    hv = d["harm_vectors_dual_use"]
    assert any(x["harm_vector"] == "Cyber" for x in hv)
    assert all(x["harm_vector"] != "Rare" for x in hv)   # n_tagged < 3 filtered out
    assert hv[0]["enabling_papers"] == ["Autonomous exploit paper"]


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


def test_targeted_heat_selects_driving_papers_and_annotates(monkeypatch):
    from signal_lag import snapshot as snap_mod
    from signal_lag.ingest import pipeline as pl
    from signal_lag.config import Settings

    s = _p("S", 2024, 1)
    c = _p("C", 2024, 2, refs=["S"])
    by_id = {"S": s, "C": c}
    tax_tags = {"S": [("saf", 1.0)], "C": [("cap", 1.0)]}
    results = {
        "divergence": [{"capability_topic": "cap", "safety_topic": "saf",
                        "gap": 0.5, "lagging": True}],
        "quadrant": [],
        "citation_flow": {"verified_borrowers": [
            {"arxiv_id": "C", "title": "C", "capability_topics": ["Capability"],
             "cited_safety": [{"arxiv_id": "S", "title": "S"}], "n_cited_safety": 1}]},
    }
    captured = {}

    def fake_enrich(settings, papers):
        captured["ids"] = {p.arxiv_id for p in papers}
        for p in papers:
            p.cited_by_count = 42
        return len(papers)

    monkeypatch.setattr(pl, "enrich_specific_citations", fake_enrich)
    settings = Settings(raw={"citations": {"targeted_heat_max": 300}})
    n = snap_mod._enrich_targeted_heat(settings, results, _taxonomy(), tax_tags, by_id)
    assert n == 2
    assert captured["ids"] == {"S", "C"}           # only the driving papers
    b = results["citation_flow"]["verified_borrowers"][0]
    assert b["cited_by_count"] == 42               # borrower annotated
    assert b["cited_safety"][0]["cited_by_count"] == 42


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

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


def test_incident_text_cleanup_and_confidence_tier():
    r = foresight._normalize_incident({
        "title": "  Al agent  scams users ", "summary": "-- An Al   model failed",
        "deployer": "Alibaba bot", "harm_key": "scams_fraud", "date": "2026-03",
        "source_url": "http://x", "ai_involvement_confidence": "high",
        "attribution_confidence": "low", "source_quality": "medium", "severity": "HIGH"})
    assert r["title"] == "AI agent scams users"          # 'Al'->'AI', whitespace collapsed
    assert r["summary"] == "An AI model failed"          # leading junk stripped, 'Al'->'AI'
    assert r["deployer"] == "Alibaba bot"                # real word 'Alibaba' untouched
    assert r["severity"] == "high"                       # normalized to lowercase vocab
    assert r["confidence"] == "low"                      # weakest link (attribution=low) wins


def test_fetch_incidents_dedupes_and_requires_title(monkeypatch):
    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        return ('{"incidents": ['
                '{"title": "Dup", "date": "2026-01", "harm_key": "cyber_offense", "source_url": "u1"},'
                '{"title": "dup", "date": "2026-01", "harm_key": "cyber_offense", "source_url": "u2"},'
                '{"title": "", "date": "2026-02", "harm_key": "cyber_offense", "source_url": "u3"}]}')

    monkeypatch.setattr(foresight.llm, "call_claude", fake_call)
    out = foresight.fetch_incidents([{"key": "cyber_offense", "label": "Cyber"}], api_key="k")
    assert len(out) == 1 and out[0]["title"] == "Dup"    # case-insensitive dupe + empty title dropped


# -------------------------------------- #3/#13/#14 derived-signal alerts module
def _alerts_snapshot():
    return {
        "timeseries": [
            {"topic_key": "cap", "period": "2025Q1", "count": 10},
            {"topic_key": "cap", "period": "2025Q2", "count": 20},
            {"topic_key": "saf", "period": "2025Q1", "count": 5},
            {"topic_key": "saf", "period": "2025Q2", "count": 6},
        ],
        "divergence": [
            {"pairing": "Cap vs. Saf", "capability_topic": "cap", "safety_topic": "saf",
             "cap_growth": 0.5, "saf_growth": 0.0, "cap_recent": 20, "saf_recent": 6,
             "volume_ratio": 3.3, "lagging": True},
        ],
        "sentiment": {"cap": {"trend": -0.05, "n_recent": 40}},
        "lab_activity": [{"topic": "cap", "published": "2026-06-01"}],
        "weekly": {"counts_by_key": {"cap": 40}},
    }


def test_monitoring_debt_accumulates_capability_minus_safety():
    from signal_lag.analysis import alerts
    debt = alerts.monitoring_debt(_alerts_snapshot())
    assert len(debt) == 1
    # Q1: (10-5)=5 ; Q2 cumulative: 5 + (20-6)=19
    assert debt[0]["debt"] == [5, 19]
    assert debt[0]["rising"] is True and debt[0]["latest"] == 19


def test_weekly_momentum_flags_spike_vs_expected():
    from signal_lag.analysis import alerts
    mom = alerts.weekly_momentum(_alerts_snapshot(), window_days=7)
    row = next(m for m in mom if m["topic_key"] == "cap")
    # recent quarterly mean = (10+20)/2 = 15 -> expected weekly = 15 * 7/91.31 ≈ 1.15
    assert row["expected"] < 3 and row["actual"] == 40
    assert row["z"] > 2                                   # 40 vs ~1.15 is a huge spike


def test_false_confidence_fires_on_rising_cap_falling_critique():
    from signal_lag.analysis import alerts
    fc = alerts.false_confidence_alerts(_alerts_snapshot())
    assert len(fc) == 1
    assert fc[0]["capability_topic"] == "cap" and fc[0]["lab_active"] is True


def test_false_confidence_silent_when_safety_growing():
    from signal_lag.analysis import alerts
    snap = _alerts_snapshot()
    snap["divergence"][0]["saf_growth"] = 0.4          # safety keeping pace -> no alert
    assert alerts.false_confidence_alerts(snap) == []


# ------------------------------------------------ #5 register recalibration
def test_attach_scores_normalizes_tiebreaker_fields():
    r = foresight._attach_scores([{"risk": "R", "severity": 4, "likelihood": 3,
                                   "confidence": "5", "evidence_strength": 99}])[0]
    assert r["confidence"] == 5 and r["evidence_strength"] == 5   # clamped
    assert r["actionability"] == 3                                # default when absent


def _reg_entry(rid, priority, conf, traj, last_seen, exposure=3):
    return {"id": rid, "risk": rid, "first_seen": "2026-01-01", "last_seen": last_seen,
            "n_appearances": 1,
            "latest": {"priority": priority, "confidence": conf, "trajectory": traj,
                       "exposure": exposure}}


def test_sort_register_breaks_priority_ties_by_confidence():
    from signal_lag import snapshot as snap_mod
    reg = [_reg_entry("A", 12, 3, "steady", "2026-07-01"),
           _reg_entry("B", 12, 5, "steady", "2026-07-01")]
    ranked = snap_mod.sort_register(reg)
    assert [e["id"] for e in ranked] == ["B", "A"]   # same priority, higher confidence first


def test_sort_register_downgrades_stale_risks():
    from signal_lag import snapshot as snap_mod
    # Fresh P12 vs stale P16 (not re-seen): stale penalty (6) drops it below the fresh one.
    reg = [_reg_entry("stale", 16, 5, "accelerating", "2026-06-01"),
           _reg_entry("fresh", 12, 3, "steady", "2026-07-01")]
    ranked = snap_mod.sort_register(reg)
    assert ranked[0]["id"] == "fresh"
    assert snap_mod.register_is_stale(reg[0], snap_mod.register_newest_date(reg)) is True


# ---------------------------------------- #6 watchlist statuses
def test_register_status_derivation():
    from signal_lag import snapshot as snap_mod
    newest = "2026-07-02"

    def _e(hist, n_app, last_seen="2026-07-02", conf=3):
        return {"history": hist, "n_appearances": n_app, "last_seen": last_seen,
                "latest": {"confidence": conf}}

    up = _e([{"date": "2026-06-25", "priority": 9}, {"date": newest, "priority": 16}], 2)
    down = _e([{"date": "2026-06-25", "priority": 16}, {"date": newest, "priority": 9}], 2)
    flat = _e([{"date": "2026-06-25", "priority": 12}, {"date": newest, "priority": 12}], 2)
    persistent = _e([{"date": d, "priority": 12} for d in
                     ("2026-06-18", "2026-06-25", newest)], 3, conf=4)
    new = _e([{"date": newest, "priority": 12}], 1)
    stale = _e([{"date": "2026-06-25", "priority": 20}], 1, last_seen="2026-06-25")
    assert snap_mod.register_status(up, newest) == "strengthening"
    assert snap_mod.register_status(down, newest) == "weakening"
    assert snap_mod.register_status(flat, newest) == "open"
    assert snap_mod.register_status(persistent, newest) == "strengthening"  # 3× + conf 4
    assert snap_mod.register_status(new, newest) == "open"
    assert snap_mod.register_status(stale, newest) == "dormant"


def test_register_write_stores_status(tmp_path):
    from signal_lag import snapshot as snap_mod
    path = tmp_path / "register.json"
    snap_mod.append_risk_register(_snap_with_risk("2026-07-01", "none found"), path)
    e = json.loads(path.read_text())[0]
    assert e["status"] == "open"                    # first appearance, fresh


# ---------------------------------------- #8 epistemic claim labels
def test_attach_scores_normalizes_claim_basis():
    r = foresight._attach_scores([{
        "risk": "R", "severity": 3, "likelihood": 3,
        "claims": [{"text": "measured", "basis": "OBSERVED"},
                   {"text": "guessy", "basis": "wild-guess"},
                   {"basis": "observed"},                     # no text -> dropped
                   {"text": "reasoned", "basis": "inferred"}],
    }])[0]
    assert [c["basis"] for c in r["claims"]] == ["observed", "speculative", "inferred"]


# ---------------------------------------- #9/#28 validation scaffolds
def test_register_calibration_counts_movement_and_disputes():
    from signal_lag.analysis import alerts
    reg = [
        {"n_appearances": 3, "history": [
            {"date": "2026-06-01", "priority": 9, "disputed": False},
            {"date": "2026-07-01", "priority": 16, "disputed": True}],
         "counterevidence": [{"date": "2026-07-01", "disputed_claims": "x"}]},
        {"n_appearances": 1, "history": [{"date": "2026-07-01", "priority": 12}]},
        {"n_appearances": 2, "history": [
            {"date": "2026-06-01", "priority": 12}, {"date": "2026-07-01", "priority": 6}]},
    ]
    cal = alerts.register_calibration(reg)
    assert cal["n"] == 3 and cal["n_reseen"] == 2
    assert cal["n_upgraded"] == 1 and cal["n_downgraded"] == 1
    assert cal["n_ever_disputed"] == 1
    assert cal["n_refreshes"] == 2 and cal["last_date"] == "2026-07-01"


def test_benchmark_history_appends_and_transitions(tmp_path):
    from signal_lag import snapshot as snap_mod
    from signal_lag.analysis import alerts
    path = tmp_path / "bench.json"

    def _snap(date, quadrant, n_inc):
        return {"meta": {"refreshed_at": date},
                "incidents": {"benchmark": [
                    {"key": "cyber", "label": "Cyber", "quadrant": quadrant,
                     "research_change_pct": 10, "n_incidents": n_inc}]}}

    snap_mod.append_benchmark_history(_snap("2026-07-01", "foresight lead", 0), path)
    snap_mod.append_benchmark_history(_snap("2026-07-01", "foresight lead", 0), path)  # idempotent
    snap_mod.append_benchmark_history(_snap("2026-07-08", "materializing", 2), path)
    rows = json.loads(path.read_text())
    assert len(rows) == 2                                   # same-date re-run overwrote
    trans = alerts.benchmark_transitions(rows)
    assert trans["n_refreshes"] == 2
    assert trans["materialized"] == [{"key": "cyber", "label": "Cyber",
                                      "lead_date": "2026-07-01",
                                      "incident_date": "2026-07-08"}]
    assert trans["open_leads"] == []


# ------------------------------- #16/#17/#18 citation matrix / bridges / safety impact
def test_citation_graph_matrix_bridges_and_impact():
    from signal_lag.analysis import citation_graph
    tax = _taxonomy()
    saf_p = _p("saf1", 2024, 1)                      # safety paper (the citation target)
    cap_p = _p("cap1", 2025, 1, refs=["saf1"])       # capability paper citing it
    dual_p = _p("dual1", 2025, 2, refs=["saf1"])     # tagged BOTH sides + cites across
    plain = _p("plain", 2025, 2)                     # no refs -> coverage denominator only
    saf_p.cited_by_count = 40
    papers = [saf_p, cap_p, dual_p, plain]
    tags = {"saf1": [("saf", 0.5)], "cap1": [("cap", 0.5)],
            "dual1": [("cap", 0.5), ("saf", 0.4)], "plain": [("cap", 0.3)]}
    out = citation_graph.citation_graph(papers, tags, tax)
    # Matrix: capability -> safety edge counted (from cap1 and dual1).
    assert out["matrix_cap_to_saf"]["Capability"]["Safety"] == 2
    # Bridges: dual1 (dual-tagged + cross-citing) ranks first.
    assert out["bridge_papers"][0]["arxiv_id"] == "dual1"
    assert out["bridge_papers"][0]["dual_tagged"] is True
    # Impact: saf1 cited by 2 capability-side papers in-corpus.
    imp = {r["arxiv_id"]: r for r in out["safety_impact"]}
    assert imp["saf1"]["n_capability_citers"] == 2
    # Coverage honest: 2 of 4 tagged papers had references.
    assert out["coverage"]["n_with_references"] == 2
    assert out["coverage"]["n_tagged"] == 4


# ------------------------------------- #11/#12/#22 sentiment quadrants + adjusted gap + CIs
def test_wilson_interval_bounds_and_width():
    from signal_lag.analysis import alerts
    lo, hi = alerts.wilson_interval(0.2, 100)
    assert 0.12 < lo < 0.2 < hi < 0.3
    lo2, hi2 = alerts.wilson_interval(0.2, 10)
    assert (hi2 - lo2) > (hi - lo)                 # smaller n -> wider interval
    assert alerts.wilson_interval(0.5, 0) == (0.0, 1.0)


def test_sentiment_quadrants_classify_by_momentum_and_critique():
    from signal_lag.analysis import alerts
    snap = {
        "inflections": [{"topic_key": "a", "change": 0.4, "recent_mean": 50},
                        {"topic_key": "b", "change": 0.4, "recent_mean": 50},
                        {"topic_key": "c", "change": -0.3, "recent_mean": 20}],
        "sentiment": {"a": {"trend": 0.05, "n_recent": 30, "recent_share": 0.2},
                      "b": {"trend": -0.04, "n_recent": 30, "recent_share": 0.1},
                      "c": {"trend": 0.06, "n_recent": 20, "recent_share": 0.3}},
    }
    q = {r["topic_key"]: r["quadrant"] for r in alerts.sentiment_quadrants(snap)}
    assert q == {"a": "growing & straining", "b": "growing & confident",
                 "c": "contracting & critical"}


def test_confidence_adjusted_divergence_strengthens_uncritical_capability():
    from signal_lag.analysis import alerts
    snap = {
        "divergence": [{"pairing": "P", "capability_topic": "cap", "safety_topic": "saf",
                        "cap_growth": 0.4, "saf_growth": 0.2, "gap": 0.2}],
        "sentiment": {"cap": {"recent_share": 0.02, "n_recent": 40},   # little self-critique
                      "saf": {"recent_share": 0.40, "n_recent": 40}},  # very critical safety
    }
    a = alerts.confidence_adjusted_divergence(snap)[0]
    # cap 0.4*0.98 - saf 0.2*0.60 = 0.392-0.12 = 0.272 > raw 0.2 -> warning strengthens
    assert a["adjusted_gap"] > a["raw_gap"]
    assert "self-critique" in a["reason"]


# ------------------------------------------------ #25 verification cache
def test_verify_attach_reuses_fresh_cache(monkeypatch):
    calls = {"n": 0}

    def fake_verify(risk, api_key, model, tool_version):
        calls["n"] += 1
        return {"novelty_rating": "genuinely_unsurfaced"}

    monkeypatch.setattr(foresight, "verify_novelty", fake_verify)
    fp = foresight._risk_fingerprint("Cached risk")
    cache = {fp: {"date_checked": "2026-06-25",
                  "verification": {"novelty_rating": "partially_anticipated"}}}
    risks = [{"risk": "Cached risk"}, {"risk": "Fresh risk"}]
    foresight._verify_attach(risks, "k", "m", "tv", cache=cache, today="2026-07-02",
                             max_age_days=21)
    assert calls["n"] == 1                                        # only the uncached one
    assert risks[0]["verification"]["novelty_rating"] == "partially_anticipated"
    assert risks[0].get("verification_cached") is True
    # The live result was written back into the cache.
    assert cache[foresight._risk_fingerprint("Fresh risk")]["date_checked"] == "2026-07-02"


def test_verify_attach_expires_stale_cache(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(foresight, "verify_novelty",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or
                        {"novelty_rating": "genuinely_unsurfaced"})
    fp = foresight._risk_fingerprint("Old risk")
    cache = {fp: {"date_checked": "2026-05-01",
                  "verification": {"novelty_rating": "partially_anticipated"}}}
    risks = [{"risk": "Old risk"}]
    foresight._verify_attach(risks, "k", "m", "tv", cache=cache, today="2026-07-02",
                             max_age_days=21)
    assert calls["n"] == 1                                        # expired -> re-verified
    assert risks[0]["verification"]["novelty_rating"] == "genuinely_unsurfaced"


def test_risk_fingerprint_changes_on_rewording():
    a = foresight._risk_fingerprint("Agents  fail  SILENTLY")
    assert a == foresight._risk_fingerprint("agents fail silently")   # normalization
    assert a != foresight._risk_fingerprint("agents fail loudly")     # rewording -> re-verify


# ------------------------------------------------ #39 per-tab deltas
def test_tab_deltas_reports_movement_per_tab():
    from signal_lag.analysis import alerts
    prev = {
        "meta": {"refreshed_at": "2026-06-25"},
        "divergence": [{"pairing": "A vs. B", "lagging": True},
                       {"pairing": "C vs. D", "lagging": False}],
        "inflections": [{"topic_key": "t1", "direction": "acceleration"}],
        "sentiment": {"s1": {"rising": False, "trend": 0.01}},
        "analysis": {"foresight_gap": {"risks": [{"risk": "Old risk"}]}},
        "incidents": {"records": [{"title": "I-old", "date": "2026-05"}]},
    }
    cur = {
        "meta": {"refreshed_at": "2026-07-02"},
        "divergence": [{"pairing": "A vs. B", "lagging": False},
                       {"pairing": "C vs. D", "lagging": True}],
        "inflections": [{"topic_key": "t1", "direction": "acceleration"},
                        {"topic_key": "t2", "direction": "acceleration"}],
        "sentiment": {"s1": {"rising": True, "trend": 0.09}},
        "analysis": {"foresight_gap": {"risks": [{"risk": "New risk"}]}},
        "incidents": {"records": [{"title": "I-old", "date": "2026-05"},
                                  {"title": "I-new", "date": "2026-06"}]},
    }
    d = alerts.tab_deltas(cur, prev)
    assert d["divergence"]["new_lagging"] == ["C vs. D"]
    assert d["divergence"]["resolved"] == ["A vs. B"]
    assert d["velocity"]["new_accelerating"] == ["t2"]
    assert d["sentiment"]["new_rising"] == ["s1"]
    assert d["sentiment"]["biggest_shifts"][0]["shift_pts"] == 8.0
    assert d["foresight"]["new_risks"] == ["New risk"]
    assert d["foresight"]["dropped_risks"] == ["Old risk"]
    assert d["incidents"]["new"] == [{"title": "I-new", "date": "2026-06"}]
    assert alerts.tab_deltas(cur, None) == {}     # first run -> no deltas


# ------------------------------------------------ #23 counterevidence persistence
def _snap_with_risk(date, disputed):
    risk = foresight._attach_scores([{
        "risk": "Contested risk", "severity": 4, "likelihood": 3, "exposure": 3,
        "trajectory": "steady",
        "verification": {"novelty_rating": "partially_anticipated",
                         "disputed_claims": disputed,
                         "prior_coverage": "some coverage",
                         "sources": [{"title": "T", "url": "http://u"}]},
    }])[0]
    return {"meta": {"refreshed_at": date},
            "analysis": {"foresight_gap": {"risks": [risk]}}, "weekly": {}}


def test_register_persists_counterevidence_without_duplicates(tmp_path):
    from signal_lag import snapshot as snap_mod
    path = tmp_path / "register.json"
    snap_mod.append_risk_register(_snap_with_risk("2026-07-01", "Claim X is contested"), path)
    snap_mod.append_risk_register(_snap_with_risk("2026-07-08", "Claim X is contested"), path)
    snap_mod.append_risk_register(_snap_with_risk("2026-07-15", "New dispute about Y"), path)
    reg = json.loads(path.read_text())
    e = reg[0]
    assert e["latest"]["disputed_claims"] == "New dispute about Y"
    assert e["latest"]["sources"] == [{"title": "T", "url": "http://u"}]
    # Distinct disputes accumulate; the repeat on 07-08 does not duplicate.
    assert [c["disputed_claims"] for c in e["counterevidence"]] == [
        "Claim X is contested", "New dispute about Y"]
    assert e["history"][-1]["disputed"] is True


def test_register_none_found_is_not_a_dispute(tmp_path):
    from signal_lag import snapshot as snap_mod
    path = tmp_path / "register.json"
    snap_mod.append_risk_register(_snap_with_risk("2026-07-01", "none found"), path)
    e = json.loads(path.read_text())[0]
    assert "counterevidence" not in e
    assert e["history"][-1]["disputed"] is False


# ------------------------------------------------ #2 lab -> safety-response lag
def _lag_taxonomy():
    from signal_lag.config import Pairing
    return Taxonomy(
        safety_topics=[Topic("saf", "Oversight", "safety", ["s"])],
        capability_topics=[Topic("cap", "Agents", "capability", ["c"])],
        pairings=[Pairing(name="Agents vs. Oversight", capability="cap", safety="saf")],
    )


def _paper_on(aid, d):
    return Paper(aid, aid, "abstract", d, None)


def test_lab_lag_measures_response_and_marks_pending():
    from signal_lag.analysis import lab_lag
    tax = _lag_taxonomy()
    today = dt.date(2026, 7, 1)
    ann = dt.date(2026, 3, 1)
    # Baseline: 1 safety paper in the 8 weeks before; then a burst of 6 within 3 weeks after.
    papers = [_paper_on("b0", dt.date(2026, 2, 1))]
    papers += [_paper_on(f"a{i}", ann + dt.timedelta(days=3 + i)) for i in range(6)]
    tax_tags = {p.arxiv_id: [("saf", 0.5)] for p in papers}
    posts = [
        {"source": "OpenAI", "title": "Agent launch", "published": "2026-03-01", "topic": "cap"},
        {"source": "OpenAI", "title": "Too recent", "published": "2026-06-20", "topic": "cap"},
        {"source": "OpenAI", "title": "Untagged", "published": "2026-03-01", "topic": None},
    ]
    out = lab_lag.lab_response_lag(posts, papers, tax_tags, tax, today)
    assert out["available"] is True
    assert out["n_posts_considered"] == 2            # untagged post excluded
    responded = [p for p in out["posts"] if p["announcement"] == "Agent launch"][0]
    assert responded["status"] == "responded" and responded["weeks_to_measurable"] == 1
    recent = [p for p in out["posts"] if p["announcement"] == "Too recent"][0]
    assert recent["status"] == "pending"             # window not elapsed -> not "unanswered"
    assert out["n_window_elapsed"] == 1


def test_lab_lag_failsoft_without_posts_or_pairings():
    from signal_lag.analysis import lab_lag
    tax = _lag_taxonomy()
    assert lab_lag.lab_response_lag([], [], {}, tax, dt.date(2026, 7, 1)) == {"available": False}


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


# ------------------------------------------- #7 falsification + #33 action map
def test_instructions_request_falsification_and_action_fields():
    instr = foresight._instructions(3)
    for key in ("change_of_mind", "upgrade_if", "downgrade_if", "invalidate_if",
                "action_map", "eval_to_run", "benchmark_to_monitor", "policy_question",
                "owner_community", "data_source_to_watch"):
        assert key in instr, f"schema field {key} missing from synthesis instructions"


def test_synthesize_preserves_change_of_mind_and_action_map(monkeypatch):
    def fake_call(system, user, api_key, model="x", max_tokens=8000, tools=None):
        return ('{"risks": [{"risk": "R", "severity": 4, "likelihood": 3, "exposure": 3,'
                ' "trajectory": "accelerating",'
                ' "change_of_mind": {"upgrade_if": "u", "downgrade_if": "d", "invalidate_if": "i"},'
                ' "action_map": {"eval_to_run": "e", "benchmark_to_monitor": "b",'
                ' "mitigation": "m", "policy_question": "p", "owner_community": "o",'
                ' "data_source_to_watch": "s"}}]}')

    monkeypatch.setattr(foresight.llm, "call_claude", fake_call)
    risks = foresight._synthesize_risks({}, "", api_key="k", model="x", max_risks=1)
    assert risks and risks[0]["priority"] == 12                 # 4 × 3, scoring still runs
    assert risks[0]["change_of_mind"]["invalidate_if"] == "i"   # falsification preserved
    assert risks[0]["action_map"]["owner_community"] == "o"     # action map preserved

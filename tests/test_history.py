"""History append + compact-record tests (no network, no DB)."""
import json

from signal_lag.snapshot import _history_record, append_history


def _snap(date, n_flagged=1):
    return {
        "meta": {"refreshed_at": date, "n_papers": 100, "date_start": "2023-01-01",
                 "date_end": date, "n_flagged": n_flagged, "n_pairings": 6},
        "label_map": {"cap": "Capability X", "saf": "Safety Y"},
        "divergence": [{"capability_topic": "cap", "safety_topic": "saf",
                        "cap_growth": 0.2, "saf_growth": -0.05, "gap": 0.25, "lagging": True}],
        "sentiment": {"saf": {"rising": True, "recent_share": 0.3, "trend": 0.09}},
        "analysis": {"headline": {"meaning": "m", "why_it_matters": "w"},
                     "foresight_gap": {"risks": [
                         {"risk": "novel one", "domains_crossed": ["Economic"],
                          "verification": {"novelty_rating": "partially_anticipated"}},
                         {"risk": "old one",
                          "verification": {"novelty_rating": "already_widely_discussed"}}]}},
        "weekly": {"foresight_gap": {"risks": [
            {"risk": "weekly one",
             "verification": {"novelty_rating": "genuinely_unsurfaced"}}]}},
    }


def test_history_record_compact():
    rec = _history_record(_snap("2026-06-29"), None)
    assert rec["date"] == "2026-06-29"
    assert rec["n_flagged"] == 1
    assert "Capability X vs Safety Y" in rec["headline"]["biggest_gap_line"]
    # already-discussed risks are excluded from the compact top list
    assert [r["risk"] for r in rec["overall_foresight"]] == ["novel one"]
    assert [r["risk"] for r in rec["weekly_foresight"]] == ["weekly one"]
    assert rec["sentiment_rising"] == ["Safety Y"]


def test_append_history_idempotent_and_sorted(tmp_path):
    path = tmp_path / "history.json"
    append_history(_snap("2026-06-29"), path, None)
    append_history(_snap("2026-06-29", n_flagged=3), path, None)  # same date -> replace
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["n_flagged"] == 3  # replaced with the newer record

    append_history(_snap("2026-06-22"), path, None)  # older date
    data = json.loads(path.read_text())
    assert [e["date"] for e in data] == ["2026-06-22", "2026-06-29"]  # sorted ascending

import datetime as dt

from signal_lag.analysis import divergence, velocity
from signal_lag.config import Pairing, Taxonomy, Topic
from signal_lag.models import Paper


def _paper(aid, year, q):
    month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    return Paper(aid, "t", "a", dt.date(year, month, 15), None)


def _taxonomy():
    return Taxonomy(
        safety_topics=[Topic("saf", "Safety", "safety", ["s"])],
        capability_topics=[Topic("cap", "Capability", "capability", ["c"])],
        pairings=[Pairing("cap vs saf", "cap", "saf")],
    )


def test_divergence_flags_lagging_safety():
    # capability accelerates (1,1,4,4); safety flat (1,1,1,1)
    papers, pairs = [], []
    aid = 0
    for q, n in {1: 1, 2: 1, 3: 4, 4: 4}.items():
        for _ in range(n):
            aid += 1
            papers.append(_paper(str(aid), 2023, q))
            pairs.append((str(aid), "cap"))
    for q in (1, 2, 3, 4):
        aid += 1
        papers.append(_paper(str(aid), 2023, q))
        pairs.append((str(aid), "saf"))

    tags = {}
    for a, k in pairs:
        tags.setdefault(a, []).append((k, 1.0))

    ts = velocity.topic_timeseries(papers, tags)
    div = divergence.compute_divergence(ts, _taxonomy(), window=2, gap_threshold=0.25)
    rec = div[0]
    assert rec["lagging"] is True
    assert rec["cap_growth"] > rec["saf_growth"]

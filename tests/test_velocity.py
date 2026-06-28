import datetime as dt

import pandas as pd

from signal_lag.analysis import velocity
from signal_lag.models import Paper


def _paper(aid, year, q):
    month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    return Paper(aid, "t", "a", dt.date(year, month, 15), None)


def _tags(pairs):
    out = {}
    for aid, key in pairs:
        out.setdefault(aid, []).append((key, 1.0))
    return out


def test_timeseries_counts_by_quarter():
    papers = [_paper("1", 2023, 1), _paper("2", 2023, 1), _paper("3", 2023, 2)]
    tags = _tags([("1", "X"), ("2", "X"), ("3", "X")])
    ts = velocity.topic_timeseries(papers, tags)
    s = velocity.topic_series(ts, "X")
    assert s.iloc[0] == 2  # 2023Q1
    assert s.iloc[1] == 1  # 2023Q2


def test_inflection_acceleration():
    # 4 quarters: 1,1 then 3,3 -> +200% acceleration with window=2
    papers, pairs = [], []
    counts = {1: 1, 2: 1, 3: 3, 4: 3}
    aid = 0
    for q, n in counts.items():
        for _ in range(n):
            aid += 1
            papers.append(_paper(str(aid), 2023, q))
            pairs.append((str(aid), "X"))
    ts = velocity.topic_timeseries(papers, _tags(pairs))
    infl = velocity.compute_inflections(ts, window=2, threshold=0.3)
    rec = next(r for r in infl if r["topic_key"] == "X")
    assert rec["direction"] == "acceleration"
    assert rec["change"] > 0.3


def test_drop_incomplete_tail_removes_current_quarter():
    import datetime as dt
    import pandas as pd

    today = dt.date.today()
    cur = pd.Period(today, freq="Q")
    papers = [_paper("1", 2024, 1), Paper("2", "t", "a", today, None)]
    ts = velocity.topic_timeseries(papers, _tags([("1", "X"), ("2", "X")]))
    trimmed = velocity.drop_incomplete_tail(ts, today)
    assert (ts["period"] == cur).any()
    assert not (trimmed["period"] == cur).any()


def test_newly_forming():
    papers = [_paper("1", 2023, 1), _paper("2", 2023, 4)]
    tags = _tags([("1", "OLD"), ("2", "NEW")])
    ts = velocity.topic_timeseries(papers, tags)
    new = velocity.newly_forming(ts, max_age_periods=1)
    assert "NEW" in new and "OLD" not in new

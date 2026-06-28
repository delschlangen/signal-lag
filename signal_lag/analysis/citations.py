"""Citation dynamics: rapid recent growth and 'sleeper' early-heat detection.

Relies on OpenAlex ``counts_by_year`` enrichment. Offline fixtures carry the same
shape, so this runs without network.
"""
from __future__ import annotations


def _recent_share(counts_by_year, recent_years: int) -> float:
    if not counts_by_year:
        return 0.0
    years = sorted(c["year"] for c in counts_by_year if c.get("year") is not None)
    if not years:
        return 0.0
    cutoff = max(years) - recent_years + 1
    total = sum((c.get("count") or 0) for c in counts_by_year)
    recent = sum((c.get("count") or 0) for c in counts_by_year if c.get("year", 0) >= cutoff)
    return (recent / total) if total else 0.0


def citation_signals(papers, cfg: dict) -> dict:
    """Return rapidly-growing and sleeper papers.

    - rapid_growth: high total citations AND a large share arriving recently.
    - sleepers: previously quiet papers now spiking (low early, high recent share).
    """
    min_cit = int(cfg.get("min_citations", 5))
    recent_years = int(cfg.get("recent_window_periods", 4)) // 4 or 1
    share_thr = float(cfg.get("recent_share_threshold", 0.5))

    rows = []
    for p in papers:
        total = p.cited_by_count or 0
        if total < min_cit or not p.counts_by_year:
            continue
        share = _recent_share(p.counts_by_year, recent_years)
        years = sorted(c["year"] for c in p.counts_by_year if c.get("year") is not None)
        early = 0
        if years:
            first_year = years[0]
            early = sum(
                (c.get("count") or 0)
                for c in p.counts_by_year
                if c.get("year") in (first_year, first_year + 1)
            )
        early_share = (early / total) if total else 0.0
        rows.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "cited_by_count": total,
                "recent_share": round(share, 3),
                "early_share": round(early_share, 3),
            }
        )

    rapid = sorted(
        [r for r in rows if r["recent_share"] >= share_thr],
        key=lambda r: (r["recent_share"], r["cited_by_count"]),
        reverse=True,
    )
    sleepers = sorted(
        [r for r in rows if r["early_share"] < 0.2 and r["recent_share"] >= share_thr],
        key=lambda r: r["cited_by_count"],
        reverse=True,
    )
    return {"rapid_growth": rapid[:20], "sleepers": sleepers[:20]}

"""Citation cross-pollination (#16), bridge papers (#17), safety-impact score (#18).

One pass over the corpus's real outgoing references (arXiv ids captured from Semantic
Scholar during enrichment) builds three views of whether the capability and safety
communities actually ENGAGE each other — beyond sharing vocabulary:

- **matrix**: capability-topic × safety-topic citation counts (both directions), the
  "which fields talk to each other" heatmap.
- **bridge papers**: papers that connect the two sides — tagged on both, or citing
  across the boundary. Bridge papers often precede field convergence.
- **safety impact**: the safety papers most cited BY capability work in-corpus — safety
  *uptake* by builders, not just safety output.

All positive-only and coverage-honest: Semantic Scholar reference coverage is partial
(brand-new papers are rarely indexed), so absence of an edge is inconclusive, and the
returned ``coverage`` block states exactly how much of the corpus was checkable.
Pure local computation; fail-soft on missing references.
"""
from __future__ import annotations

import logging

log = logging.getLogger("signal_lag.citation_graph")


def citation_graph(papers, tax_tags, taxonomy, max_bridges: int = 15,
                   max_impact: int = 15) -> dict:
    """Build the citation matrix, bridge-paper list, and safety-impact leaderboard."""
    safety_keys = {t.key for t in taxonomy.safety_topics}
    cap_keys = {t.key for t in taxonomy.capability_topics}
    lm = {t.key: t.label for t in taxonomy.all_topics}
    by_id = {p.arxiv_id: p for p in papers}

    def _sides(tags):
        saf = [k for k, _ in tags if k in safety_keys]
        cap = [k for k, _ in tags if k in cap_keys]
        return saf, cap

    side_of: dict[str, tuple[list, list]] = {
        aid: _sides(tags) for aid, tags in tax_tags.items() if aid in by_id
    }

    # --- one pass over references: matrix edges + per-paper cross-cites + impact ---
    cap_to_saf: dict[str, dict[str, int]] = {}
    saf_to_cap: dict[str, dict[str, int]] = {}
    cross_cites: dict[str, int] = {}          # aid -> n references across the boundary
    cap_citers_of: dict[str, set] = {}        # safety aid -> set of capability citer aids
    n_with_refs = 0
    for aid, (saf, cap) in side_of.items():
        p = by_id[aid]
        if not p.referenced_works:
            continue
        n_with_refs += 1
        for ref in p.referenced_works:
            if ref == aid or ref not in side_of:
                continue
            rsaf, rcap = side_of[ref]
            if cap and rsaf:                  # capability paper cites safety work
                cross_cites[aid] = cross_cites.get(aid, 0) + 1
                for ck in cap:
                    row = cap_to_saf.setdefault(ck, {})
                    for sk in rsaf:
                        row[sk] = row.get(sk, 0) + 1
                cap_citers_of.setdefault(ref, set()).add(aid)
            if saf and rcap:                  # safety paper cites capability work
                cross_cites[aid] = cross_cites.get(aid, 0) + 1
                for sk in saf:
                    row = saf_to_cap.setdefault(sk, {})
                    for ck in rcap:
                        row[ck] = row.get(ck, 0) + 1

    # --- bridge papers (#17): dual-tagged and/or citing across the boundary ---
    bridges = []
    for aid, (saf, cap) in side_of.items():
        dual = bool(saf and cap)
        n_cross = cross_cites.get(aid, 0)
        cited_by_both = aid in cap_citers_of and bool(saf)
        score = (2 if dual else 0) + min(n_cross, 3) + (1 if cited_by_both else 0)
        if score >= 2:
            p = by_id[aid]
            bridges.append({
                "arxiv_id": aid, "title": p.title,
                "published": p.published.isoformat() if p.published else None,
                "capability_topics": [lm.get(k, k) for k in cap],
                "safety_topics": [lm.get(k, k) for k in saf],
                "dual_tagged": dual, "n_cross_citations": n_cross,
                "cited_by_count": p.cited_by_count, "bridge_score": score,
            })
    bridges.sort(key=lambda b: (b["bridge_score"], b["cited_by_count"] or 0), reverse=True)

    # --- safety-impact leaderboard (#18): safety papers capability work actually cites ---
    impact = []
    for aid, citers in cap_citers_of.items():
        p = by_id.get(aid)
        if p is None:
            continue
        saf, _cap = side_of.get(aid, ([], []))
        impact.append({
            "arxiv_id": aid, "title": p.title,
            "published": p.published.isoformat() if p.published else None,
            "safety_topics": [lm.get(k, k) for k in saf],
            "n_capability_citers": len(citers),
            "cited_by_count": p.cited_by_count,
            "influential_citations": p.s2_influential_citations,
        })
    impact.sort(key=lambda r: (r["n_capability_citers"], r["cited_by_count"] or 0),
                reverse=True)

    def _label_matrix(m):
        return {lm.get(rk, rk): {lm.get(ck, ck): v for ck, v in row.items()}
                for rk, row in m.items()}

    out = {
        "matrix_cap_to_saf": _label_matrix(cap_to_saf),
        "matrix_saf_to_cap": _label_matrix(saf_to_cap),
        "bridge_papers": bridges[:max_bridges],
        "safety_impact": impact[:max_impact],
        "coverage": {
            "n_tagged": len(side_of),
            "n_with_references": n_with_refs,
            "pct_with_references": round(100 * n_with_refs / len(side_of))
            if side_of else 0,
        },
    }
    log.info("Citation graph: %d cap→saf topic edges, %d bridges, %d impacted safety "
             "papers (%d%% of tagged corpus had references)",
             sum(len(r) for r in cap_to_saf.values()), len(out["bridge_papers"]),
             len(out["safety_impact"]), out["coverage"]["pct_with_references"])
    return out

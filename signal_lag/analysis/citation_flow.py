"""Citation-flow verification — does an applied/capability paper actually *cite*
core safety work, not merely share its vocabulary?

The cross-silo "borrowing" thesis elsewhere in signal-lag rests on shared taxonomy
tags (vocabulary overlap). That can be a false positive: two communities can use the
same words without one building on the other. A paper's outgoing ``references`` (the
arXiv ids of the works it cites, captured from Semantic Scholar during enrichment —
OpenAlex is unreachable in CI) let us check the real link: intersect a capability
paper's references with the arXiv ids of our safety-tagged papers.

This is deliberately **positive-only**. A hit ("this capability paper cites that
safety paper") is strong, verifiable evidence of borrowing. The *absence* of a hit is
inconclusive, not evidence of no borrowing — the cited safety paper may simply sit
outside our temporally-stratified sample, or the citing paper may not be enriched.
So we surface "citation-verified: yes" and never "no borrowing".

Pure local computation (no network); fail-soft on missing references.
"""
from __future__ import annotations

import logging

log = logging.getLogger("signal_lag.citation_flow")


def _topic_kinds(tags, safety_keys, cap_keys):
    """Split a paper's tag list into (safety topic keys, capability topic keys)."""
    saf = [k for k, _ in tags if k in safety_keys]
    cap = [k for k, _ in tags if k in cap_keys]
    return saf, cap


def citation_flow(papers, tax_tags, taxonomy, max_examples: int = 30) -> dict:
    """Verify cross-domain citation flow from capability papers into safety work.

    Builds the set of safety-tagged papers' arXiv ids, then for every capability-tagged
    paper whose ``referenced_works`` (cited arXiv ids) we have, intersects its outgoing
    references with that safety set. A non-empty intersection is a citation-verified
    borrowing (positive-only).

    Returns a dict:
      - ``verified_borrowers``: list of {arxiv_id, title, capability_topics,
        safety_topics, cited_safety:[{arxiv_id,title}], n_cited_safety}
      - ``verified_ids``: arxiv_ids with a verified borrowing (for digest annotation)
      - ``n_safety_indexed``: how many safety papers are in the searchable target set
      - ``n_candidates_checked``: capability papers that actually had references
    """
    safety_keys = {t.key for t in taxonomy.safety_topics}
    cap_keys = {t.key for t in taxonomy.capability_topics}
    by_id = {p.arxiv_id: p for p in papers}

    # Target set: arXiv ids of safety-tagged papers in the corpus.
    safety_ids = {
        aid for aid, tags in tax_tags.items()
        if aid in by_id and any(k in safety_keys for k, _ in tags)
    }

    borrowers: list[dict] = []
    n_checked = 0
    for aid, tags in tax_tags.items():
        saf, cap = _topic_kinds(tags, safety_keys, cap_keys)
        if not cap:                       # only capability-side papers are candidates
            continue
        p = by_id.get(aid)
        if p is None or not p.referenced_works:
            continue
        n_checked += 1
        cited = []
        for ref_aid in p.referenced_works:
            if ref_aid in safety_ids and ref_aid != aid:   # ignore self-reference
                t = by_id.get(ref_aid)
                cited.append({"arxiv_id": ref_aid, "title": t.title if t else None})
        if cited:
            lm = {t.key: t.label for t in taxonomy.all_topics}
            borrowers.append({
                "arxiv_id": aid,
                "title": p.title,
                "capability_topics": [lm.get(k, k) for k in cap],
                "safety_topics": [lm.get(k, k) for k in saf],
                "cited_safety": cited[:8],
                "n_cited_safety": len(cited),
            })

    borrowers.sort(key=lambda b: b["n_cited_safety"], reverse=True)
    borrowers = borrowers[:max_examples]
    log.info(
        "Citation-flow: %d verified borrowers of %d candidates (%d safety papers indexed)",
        len(borrowers), n_checked, len(safety_ids),
    )
    return {
        "verified_borrowers": borrowers,
        "verified_ids": [b["arxiv_id"] for b in borrowers],
        "n_safety_indexed": len(safety_ids),
        "n_candidates_checked": n_checked,
    }

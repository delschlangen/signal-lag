"""Topic-tagging precision audit (#1) — is each topic's tag actually about that topic?

Embedding tagging is recall-friendly but can be over-inclusive: a broad centroid
(e.g. Mechanistic Interpretability) sweeps in loosely-related papers and inflates that
topic's volume — which, for safety topics, makes the safety-lag headline LOOK BETTER
than it is. Once per refresh this samples tagged papers per topic and has an LLM judge
each (title+abstract vs. the topic's definition) as true_positive / partial /
false_positive / unclear, yielding a per-topic precision estimate that is displayed in
Methodology and used by a human to set per-topic thresholds
(``taxonomy.topic_thresholds`` in settings).

Runs on the cheap model tier (boolean-ish judgments, bounded sample). Sampling is
seeded per refresh date so a re-run of the same snapshot audits the same papers.
Fail-soft: no key / any error -> None (the audit is advisory, never load-bearing).
"""
from __future__ import annotations

import json
import logging
import random

from . import llm

log = logging.getLogger("signal_lag.tag_audit")

AUDIT_SYSTEM = (
    "You are auditing an automatic paper-tagging system for an AI-safety research "
    "tracker. For each paper you are given the TOPIC it was tagged with (name + "
    "definition) and the paper's title and abstract. Judge whether the tag is right: "
    "'true_positive' = the paper is squarely about this topic; 'partial' = meaningfully "
    "related but the topic is not a main subject; 'false_positive' = only superficially/"
    "vocabulary-related; 'unclear' = cannot tell from the abstract. Judge the actual "
    "subject matter, not keyword overlap."
)

AUDIT_INSTRUCTIONS = """\
For each item below, judge the tag. Return ONLY a JSON object of this exact shape:

{"labels": [{"arxiv_id": "<id>", "verdict": "true_positive | partial | false_positive | unclear"}]}

Include every arxiv_id from the input. Output valid JSON only."""

_VERDICTS = ("true_positive", "partial", "false_positive", "unclear")


def audit_tags(
    papers, tax_tags: dict, taxonomy, api_key: str | None,
    model: str = "claude-haiku-4-5", sample_per_topic: int = 30,
    batch_size: int = 25, seed: str = "",
) -> dict | None:
    """Sample tagged papers per topic, LLM-judge each tag, estimate per-topic precision.

    Returns {"topics": [{key, label, n_sampled, true_positive, partial, false_positive,
    unclear, precision}], "n_judged": int} or None (fail-soft). Precision counts a
    partial as half a hit over the judgeable sample (unclear excluded).
    """
    if not api_key:
        return None
    by_id = {p.arxiv_id: p for p in papers}
    label_map = {t.key: t.label for t in taxonomy.all_topics}
    seeds_map = {t.key: t.seeds for t in taxonomy.all_topics}

    per_topic: dict[str, list[str]] = {}
    for aid, tags in tax_tags.items():
        if aid not in by_id:
            continue
        for k, _score in tags:
            if k in label_map:
                per_topic.setdefault(k, []).append(aid)

    rng = random.Random(f"tag-audit-{seed}")
    items = []  # (topic_key, arxiv_id)
    for k, aids in per_topic.items():
        aids = sorted(aids)
        rng.shuffle(aids)
        items += [(k, aid) for aid in aids[:sample_per_topic]]
    if not items:
        return None

    counts: dict[str, dict[str, int]] = {
        k: dict.fromkeys(_VERDICTS, 0) for k in per_topic
    }
    n_judged = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        payload = [
            {"arxiv_id": aid,
             "topic": label_map[k],
             "topic_definition": "; ".join(seeds_map.get(k, [])[:4]),
             "title": by_id[aid].title,
             "abstract": (by_id[aid].abstract or "")[:600]}
            for k, aid in batch
        ]
        text = llm.call_claude(
            AUDIT_SYSTEM,
            AUDIT_INSTRUCTIONS + "\n\nITEMS:\n" + json.dumps(payload, ensure_ascii=False),
            api_key, model,
        )
        if not text:
            continue
        result = llm.extract_json(text)
        if not result:
            continue
        verdict_of = {r.get("arxiv_id"): r.get("verdict") for r in result.get("labels", [])}
        for k, aid in batch:
            v = verdict_of.get(aid)
            if v in _VERDICTS:
                counts[k][v] += 1
                n_judged += 1

    topics = []
    for k, c in counts.items():
        judgeable = c["true_positive"] + c["partial"] + c["false_positive"]
        precision = (round((c["true_positive"] + 0.5 * c["partial"]) / judgeable, 2)
                     if judgeable else None)
        topics.append({
            "key": k, "label": label_map.get(k, k),
            "n_sampled": sum(c.values()), **c, "precision": precision,
        })
    topics.sort(key=lambda t: (t["precision"] is not None, t["precision"] or 0))
    log.info("Tag audit: judged %d tags across %d topics", n_judged, len(topics))
    return {"topics": topics, "n_judged": n_judged} if n_judged else None

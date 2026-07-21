"""Optional LLM analysis layer (Anthropic / Claude).

Once per weekly refresh, this sends the computed metrics + real paper abstracts to
Claude and gets back genuine analytical narrative: what each capability/safety
pairing *means*, what the papers are actually about, why the gap matters, a
per-tab read of the week, and a "why it matters to AI safety" line per paper.

The result is baked into the snapshot, so the dashboard just renders it — no
API calls at page-load time. Entirely fail-soft: with no ANTHROPIC_API_KEY (or
any error) it returns None and the dashboard falls back to its templated text.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("signal_lag.llm")

SYSTEM = (
    "You are a senior AI-safety research analyst writing a quarterly research-trend "
    "intelligence brief. You read research metrics (quarter-over-quarter trends) and "
    "real paper abstracts and explain, in plain analytical language, what is happening "
    "across the recent quarters and why it matters for AI safety. Be concrete and "
    "specific — name what the research is about, not generic statements. Never fabricate: "
    "rely only on the data and abstracts provided. Be concise and direct. Describe the "
    "current quarter / recent-quarter trend; do NOT call it 'this week'."
)

INSTRUCTIONS = """\
Analyze this quarter's AI-safety research-trend data (provided as JSON below; growth \
figures are quarter-over-quarter) and return ONLY a JSON object (no markdown, no \
preamble) with exactly this shape:

{
  "headline": {
    "meaning": "1-2 sentences: what this capability<->safety pairing actually means",
    "capability_focus": "what the capability papers are concretely working on",
    "safety_focus": "what the paired safety work concretely covers",
    "why_it_matters": "why this specific gap matters for AI safety"
  },
  "tabs": {
    "divergence": "2-3 sentence analytical read of the capability-vs-safety picture",
    "velocity": "2-3 sentences on what the momentum shifts mean",
    "sentiment": "2-3 sentences on what rising/low critical-share signals suggest",
    "quadrant": "2-3 sentences interpreting the emerging/hot/cooling/white-space map",
    "citations": "2-3 sentences on what the citation movers indicate"
  },
  "papers": [
    {"arxiv_id": "<id from input>", "summary": "1 sentence: what the paper does",
     "why_it_matters": "1 sentence: what it signals for AI or AI safety"}
  ]
}

Cover every arxiv_id present in input.papers. Ground every statement in the \
provided abstracts/metrics. Output must be valid JSON and nothing else.
"""


def extract_json(text: str) -> dict | None:
    """Best-effort parse of a JSON object from a model response (handles ``` fences)."""
    text = text.strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} balanced span.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


# Backwards-compatible private alias.
_extract_json = extract_json


def final_text(blocks) -> str:
    """The model's FINAL answer: text blocks after the last tool block.

    With server-side tools (web search), the response interleaves the model's
    thinking-aloud narration ("Let me run a few searches...") between tool calls,
    with the real answer in the trailing text blocks. Naively joining every text
    block leaked that narration into displayed briefs. If no tool blocks are
    present, all text blocks are the answer.
    """
    last_tool = -1
    for i, b in enumerate(blocks):
        if getattr(b, "type", None) != "text":
            last_tool = i
    return "".join(
        b.text for i, b in enumerate(blocks)
        if i > last_tool and getattr(b, "type", None) == "text"
    )


def call_claude(
    system: str, user: str, api_key: str | None,
    model: str = "claude-opus-4-8", max_tokens: int = 8000,
    tools: list | None = None,
) -> str | None:
    """Single Claude message call. Returns the concatenated text, or None on failure.

    Shared by every LLM pass (weekly analysis, foresight, novelty verification) so
    there is exactly one client/call pattern. Pass ``tools`` to enable server-side
    tools (e.g. web search); those run on Anthropic's side and the final answer comes
    back as text blocks. Fail-soft: missing key, missing SDK, or any API error -> None.
    """
    if not api_key:
        log.info("No ANTHROPIC_API_KEY; skipping LLM call")
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; skipping LLM call")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = dict(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        if tools:
            kwargs["tools"] = tools
        resp = client.messages.create(**kwargs)
        return final_text(resp.content)
    except Exception as e:  # any API failure -> fail soft
        log.warning("Claude call failed: %s", e)
        return None


def analyze_weekly(payload: dict, api_key: str | None, model: str = "claude-opus-4-8",
                   retries: int = 3) -> dict | None:
    user = INSTRUCTIONS + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False)
    result = None
    for attempt in range(1, retries + 1):
        text = call_claude(SYSTEM, user, api_key, model)
        if text is None:
            return None
        result = extract_json(text)
        if result is not None:
            break
        log.warning("Could not parse LLM analysis JSON (attempt %d/%d)", attempt, retries)
    if result is None:
        return None
    log.info("LLM analysis complete (%d paper notes)", len(result.get("papers", [])))
    return result


WEEK_SYSTEM = (
    "You are a senior AI-safety research analyst writing a short 'what landed THIS WEEK' "
    "brief. You read only the papers submitted in the last several days (not the long-run "
    "trend) and explain, in plain analytical language, what actually showed up this week "
    "and why it matters. Be concrete and specific; ground every statement in the provided "
    "titles/abstracts/counts. Never fabricate. Be concise."
)

WEEK_INSTRUCTIONS = """\
Analyze ONLY this week's papers (provided as JSON below: counts by topic + notable \
papers with abstracts) and return ONLY a JSON object (no markdown, no preamble) with \
exactly this shape:

{
  "summary": "3-5 sentences: what landed this week across AI-safety/capability research, what's notable, and what it suggests — this week only, not the long-run trend",
  "themes": ["short theme phrase", "..."],
  "notable": [
    {"arxiv_id": "<id from input>", "why_it_matters": "1 sentence: why this specific paper is worth attention this week"}
  ]
}

Cover the most notable arxiv_ids from input.notable_papers. Ground every statement in the \
provided data. Output must be valid JSON and nothing else.
"""


LIMITATION_SYSTEM = (
    "You are an AI-safety research analyst doing precise sentiment labeling. For each "
    "paper you read the title and abstract and decide ONE thing: is the paper's PRIMARY "
    "stance that something is failing, broken, unsafe, eroding, or fundamentally limited "
    "— a genuinely critical / limitation-focused contribution (e.g. an audit, a negative "
    "result, a vulnerability disclosure, evidence that a method does not work)? Crucially: "
    "papers that merely USE negative vocabulary while reporting PROGRESS are NOT critical "
    "(e.g. 'we overcome the catastrophic failures of prior methods', 'we fix the "
    "limitations of X', 'robust against attacks'). Those are positive/constructive. Judge "
    "the paper's actual stance, not its keywords."
)

LIMITATION_INSTRUCTIONS = """\
For each paper below, decide if it is genuinely LIMITATION-FOCUSED / critical (its core \
message is that something fails, is unsafe, or is fundamentally limited) versus \
constructive (it proposes/advances/fixes something, even if it uses negative words to \
describe prior work or threats it defends against).

Return ONLY a JSON object (no markdown, no preamble) of this exact shape:

{"labels": [{"arxiv_id": "<id>", "is_limitation_focused": true_or_false}]}

Include every arxiv_id from the input. Output must be valid JSON and nothing else.
"""


def classify_limitation_focused(
    papers: list[dict], api_key: str | None, model: str = "claude-opus-4-8",
    batch_size: int = 50,
) -> dict[str, bool]:
    """Batched boolean classifier: {arxiv_id -> is_limitation_focused}.

    ``papers`` is a list of {arxiv_id, title, abstract}. Used to correct embedding-based
    critical-sentiment false-positives (academic negation: "we overcome the failures of
    …" is constructive, not critical). Fail-soft: missing key / error -> {} (caller keeps
    the embedding flags). Batched to bound cost.
    """
    if not api_key or not papers:
        return {}
    out: dict[str, bool] = {}
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        payload = [
            {"arxiv_id": p["arxiv_id"], "title": p.get("title", ""),
             "abstract": (p.get("abstract") or "")[:600]}
            for p in batch
        ]
        text = call_claude(
            LIMITATION_SYSTEM,
            LIMITATION_INSTRUCTIONS + "\n\nPAPERS:\n" + json.dumps(payload, ensure_ascii=False),
            api_key, model,
        )
        if not text:
            continue
        result = extract_json(text)
        if not result:
            continue
        for row in result.get("labels", []):
            aid = row.get("arxiv_id")
            if aid is not None and "is_limitation_focused" in row:
                out[aid] = bool(row["is_limitation_focused"])
    log.info("LLM limitation classifier labeled %d/%d papers", len(out), len(papers))
    return out


def summarize_week(payload: dict, api_key: str | None, model: str = "claude-opus-4-8",
                   retries: int = 3) -> dict | None:
    """Focused 'this week only' Claude summary. Fail-soft (-> None).

    Retries on an unparseable reply: the call usually succeeds (HTTP 200) but the model
    occasionally wraps the JSON in prose, which ``extract_json`` can't recover — a single
    such miss used to leave the whole this-week summary empty. Retrying re-rolls it.
    """
    user = WEEK_INSTRUCTIONS + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False)
    result = None
    for attempt in range(1, retries + 1):
        text = call_claude(WEEK_SYSTEM, user, api_key, model)
        if text is None:
            return None                      # no key / hard API failure -> give up
        result = extract_json(text)
        if result is not None:
            break
        log.warning("Could not parse weekly summary JSON (attempt %d/%d)", attempt, retries)
    if result is None:
        return None
    log.info("Weekly summary complete (%d notable notes)", len(result.get("notable", [])))
    return result

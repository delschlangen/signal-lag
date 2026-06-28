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
    "You are a senior AI-safety research analyst writing a weekly intelligence "
    "brief. You read research metrics and real paper abstracts and explain, in "
    "plain analytical language, what is actually happening and why it matters for "
    "AI safety. Be concrete and specific — name what the research is about, not "
    "generic statements. Never fabricate: rely only on the data and abstracts "
    "provided. Be concise and direct."
)

INSTRUCTIONS = """\
Analyze this week's AI-safety research data (provided as JSON below) and return \
ONLY a JSON object (no markdown, no preamble) with exactly this shape:

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


def call_claude(
    system: str, user: str, api_key: str | None,
    model: str = "claude-opus-4-8", max_tokens: int = 8000,
) -> str | None:
    """Single Claude message call. Returns the text, or None on any failure.

    Shared by every LLM pass (weekly analysis, foresight) so there is exactly one
    client/call pattern. Fail-soft: missing key, missing SDK, or any API error -> None.
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
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except Exception as e:  # any API failure -> fail soft
        log.warning("Claude call failed: %s", e)
        return None


def analyze_weekly(payload: dict, api_key: str | None, model: str = "claude-opus-4-8") -> dict | None:
    text = call_claude(
        SYSTEM,
        INSTRUCTIONS + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False),
        api_key, model,
    )
    if text is None:
        return None
    result = extract_json(text)
    if result is None:
        log.warning("Could not parse LLM analysis JSON")
        return None
    log.info("LLM analysis complete (%d paper notes)", len(result.get("papers", [])))
    return result

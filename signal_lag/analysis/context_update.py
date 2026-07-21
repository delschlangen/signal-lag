"""Automatic weekly curation of the societal-context file (config/context.md).

The foresight synthesis crosses research signals with ``config/context.md`` — a dated,
verifiable "state of the world" board. Hand-maintained, it goes stale between edits;
this pass keeps it current: once a week (Sunday, before Monday's data refresh) Claude
web-searches the CURRENT status of each domain, corrects stale facts, adds major new
dated developments, prunes superseded items, and returns the updated body.

Guardrails, because this file is ground truth for the synthesis:
- The instructional header is preserved verbatim (only the "Last updated" line changes).
- The reply must pass structural validation (all sections present, dated with the
  current year, sane length) or the file is left untouched — a bad reply can never
  clobber a good context.
- Same facts-only rules as the live-context pass: verifiable, dated, no speculation,
  conditions not conclusions. Every change lands as a reviewable git commit, so a
  human can always diff, revert, or hand-edit (hand edits become the next week's base).
Fail-soft throughout: any failure -> None and the existing file stands.
"""
from __future__ import annotations

import logging
import re

from . import llm

log = logging.getLogger("signal_lag.context_update")

CURATOR_SYSTEM = (
    "You are the curator of a 'societal context' briefing file used to ground an AI "
    "foresight synthesis. The file states CURRENT, VERIFIABLE, DATED real-world "
    "conditions across societal domains — the board, not the move. You SEARCH THE WEB "
    "for what has changed since the file was last updated, then return the updated "
    "file body. You state conditions of the world, never conclusions about which are "
    "dangerous. Every claim must be verifiable and dated; if you cannot verify a "
    "change, keep the existing text. You never speculate, never editorialize, and "
    "never invent statistics, polls, laws, or events."
)

CURATOR_INSTRUCTIONS = """\
Below is the CURRENT body of the context file (sections '## Social' through
'## Demographic') plus the research topics the tool flagged this week. Run targeted web
searches for what has MATERIALLY changed since the file's facts, then return the updated
body. Rules:

- KEEP THE EXACT STRUCTURE: the same '## <Domain>' sections in the same order, bullet
  items, each fact dated (month/year) and verifiable.
- UPDATE stale facts to their current status (e.g. a law's actual current stage, a
  newer poll superseding an old one). CORRECT anything that has since changed.
- ADD only major, well-sourced new developments from the last few weeks — especially
  ones relevant to the flagged topics. A quiet week means few or no changes.
- PRUNE items that are superseded or no longer describe the current state; keep
  landmark items that still define the board.
- Keep roughly the same overall length (this is a briefing, not an archive).
- BREADTH over depth: conditions across ALL domains, not a deep-dive on one.

Return ONLY the updated markdown body, starting exactly with '## Social'. No preamble,
no code fences, no commentary."""

# The body starts at the first section heading; everything above is the header.
_BODY_START = "## Social"


def split_context(raw: str) -> tuple[str, str]:
    """Split the raw file into (instructional header, body). Body may be ''."""
    i = raw.find(_BODY_START)
    if i == -1:
        return raw, ""
    return raw[:i], raw[i:]


def validate_body(new_body: str, old_body: str, year: str) -> str | None:
    """Return a rejection reason, or None when the new body is safe to write."""
    new_body = (new_body or "").strip()
    if not new_body.startswith("## "):
        return "does not start with a section heading"
    old_sections = re.findall(r"^## .+$", old_body, flags=re.MULTILINE)
    new_sections = re.findall(r"^## .+$", new_body, flags=re.MULTILINE)
    missing = [s for s in old_sections if s not in new_sections]
    if missing:
        return f"missing sections: {missing}"
    if year and year not in new_body:
        return f"no {year} dates present"
    if len(new_body) < 800:
        return "suspiciously short"
    if old_body and not (0.5 <= len(new_body) / max(len(old_body), 1) <= 2.0):
        return f"length changed too much ({len(old_body)} -> {len(new_body)} chars)"
    return None


def _stamp_header(header: str, today: str) -> str:
    """Refresh the '# Last updated:' line in the header (ISO date -> 'Month YYYY')."""
    import datetime as dt
    try:
        d = dt.date.fromisoformat(today[:10])
        stamp = d.strftime("%B %Y")
    except ValueError:
        return header
    return re.sub(r"(# Last updated:).*", rf"\1 {stamp}", header)


def update_context(
    raw: str, flagged_topics: list[str], api_key: str | None,
    model: str = "claude-opus-4-8", tool_version: str = "web_search_20260209",
    today: str = "", retries: int = 2,
) -> str | None:
    """Return the updated full file text, or None (fail-soft; caller keeps the old file)."""
    if not api_key or not raw:
        return None
    header, old_body = split_context(raw)
    if not old_body:
        log.warning("Context file has no recognizable body; not updating")
        return None
    year = today[:4]
    user = (CURATOR_INSTRUCTIONS
            + (f"\n\nTODAY IS {today}." if today else "")
            + "\n\nTOPICS FLAGGED THIS WEEK:\n"
            + "\n".join(f"- {t}" for t in flagged_topics[:12] or ["(none)"])
            + "\n\nCURRENT FILE BODY:\n" + old_body)
    for attempt in range(1, retries + 1):
        for tv in (tool_version, "web_search_20250305"):
            text = llm.call_claude(
                CURATOR_SYSTEM, user, api_key, model, max_tokens=20000,
                tools=[{"type": tv, "name": "web_search"}],
            )
            if not text:
                continue
            body = text.strip()
            # Tolerate a stray fence or preamble before the first heading.
            j = body.find(_BODY_START)
            if j > 0:
                body = body[j:]
            body = re.sub(r"\s*```\s*$", "", body)
            reason = validate_body(body, old_body, year)
            if reason is None:
                log.info("Context update validated (%d -> %d chars)",
                         len(old_body), len(body))
                return _stamp_header(header, today) + body + "\n"
            log.warning("Context update rejected (attempt %d): %s", attempt, reason)
    return None

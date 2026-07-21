"""Weekly societal-context curation (Sunday, before Monday's data refresh).

Web-updates config/context.md via signal_lag.analysis.context_update: corrects stale
facts, adds major dated developments, prunes superseded items — preserving the header
and structure, with validation guards so a bad reply never clobbers the file.
Fail-soft: on any failure the existing file stands and we exit 0 (a transient miss
shouldn't page anyone weekly); the change lands as a reviewable git commit.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signal_lag.analysis import context_update  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("update_context")


def flagged_topics_from_snapshot(path: Path) -> list[str]:
    """The topics the tool currently flags (lagging pairings + rising sentiment)."""
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    lm = snap.get("label_map") or {}
    out = [d.get("pairing") for d in snap.get("divergence") or [] if d.get("lagging")]
    out += [lm.get(k, k) for k, v in (snap.get("sentiment") or {}).items()
            if v.get("rising")]
    return [t for t in out if t]


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("No ANTHROPIC_API_KEY; skipping context update")
        return 0
    ctx_path = ROOT / "config" / "context.md"
    raw = ctx_path.read_text(encoding="utf-8")
    updated = context_update.update_context(
        raw,
        flagged_topics_from_snapshot(ROOT / "data" / "snapshot.json"),
        api_key,
        today=dt.date.today().isoformat(),
    )
    if updated is None:
        log.warning("Context update failed or was rejected; keeping the existing file")
        return 0
    if updated == raw:
        log.info("No material changes this week")
        return 0
    ctx_path.write_text(updated, encoding="utf-8")
    log.info("Wrote updated %s (%d -> %d chars)", ctx_path, len(raw), len(updated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

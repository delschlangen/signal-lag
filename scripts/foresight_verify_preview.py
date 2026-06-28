"""Preview the novelty-verification step against the current snapshot's risks.

Loads the committed ``data/snapshot.json``, takes the foresight-gap risks it already
contains, and runs each through the web-search-backed novelty verification — printing
the prior-coverage check, disputed-claims finding, and recalibrated novelty rating to
the terminal. This is for tuning the verification BEFORE it's wired into the snapshot
and the dashboard.

Needs ``ANTHROPIC_API_KEY`` and web-search access, so run it in CI.

  python scripts/foresight_verify_preview.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signal_lag.analysis import foresight  # noqa: E402
from signal_lag.config import load_all  # noqa: E402
from signal_lag.snapshot import load_snapshot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("verify-preview")
RULE = "=" * 78


def main() -> int:
    settings, _ = load_all()
    acfg = settings.analysis or {}
    fcfg = acfg.get("foresight") or {}
    model = acfg.get("model", "claude-opus-4-8")
    tool_version = fcfg.get("web_search_tool", "web_search_20260209")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — run this in CI.")
        return 1

    snap = load_snapshot(settings.path("snapshot_path"))
    fg = ((snap or {}).get("analysis") or {}).get("foresight_gap") if snap else None
    if not fg or not fg.get("risks"):
        print("No foresight_gap risks in the snapshot to verify. Run a foresight refresh first.")
        return 1

    risks = fg["risks"]
    print(f"Verifying {len(risks)} risk(s) from the live snapshot via web search "
          f"(tool: {tool_version})...\n")
    for i, r in enumerate(risks, 1):
        print(RULE)
        print(f"[{i}] CANDIDATE: {r.get('risk','')}")
        print(f"    Model's own calibration: {r.get('calibration','')}")
        v = foresight.verify_novelty(r, os.environ["ANTHROPIC_API_KEY"], model, tool_version)
        if not v:
            print("    >> verification failed (fail-soft None)\n")
            continue
        print(f"    --> NOVELTY: {v.get('novelty_rating','?')}   ACTION: {v.get('recommended_action','?')}")
        print(f"    Prior coverage: {v.get('prior_coverage','')}")
        print(f"    Disputed claims: {v.get('disputed_claims','')}")
        print(f"    Recalibrated: {v.get('recalibrated_calibration','')}")
        srcs = v.get("sources") or []
        if srcs:
            print("    Sources:")
            for s in srcs[:6]:
                print(f"      - {s.get('title','')}: {s.get('url','')}")
        print()
    print(RULE)
    print("Check: does the contested/already-covered risk get flagged "
          "(already_widely_discussed / drop) rather than sailing through as novel?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

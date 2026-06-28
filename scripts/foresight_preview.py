"""Preview the Foresight Gap synthesis against the committed real snapshot.

Runs ONLY the foresight pass (no data pull, no re-analysis): it loads the existing
``data/snapshot.json`` (real arXiv/OpenAlex data) + the previous snapshot for the
week-over-week diff, builds the signal digest, loads ``config/context.md``, calls
Claude, and prints the generated risks to the terminal.

This exists so the synthesis PROMPT can be tuned on real signals before the dashboard
tab is built. Needs ``ANTHROPIC_API_KEY`` in the environment (so run it in CI where the
secret lives, or locally if you have a key).

  python scripts/foresight_preview.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signal_lag.analysis import foresight  # noqa: E402
from signal_lag.config import load_all  # noqa: E402
from signal_lag.snapshot import diff_snapshots, load_snapshot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("foresight-preview")

RULE = "=" * 78


def main() -> int:
    settings, _taxonomy = load_all()
    acfg = settings.analysis or {}
    fcfg = acfg.get("foresight") or {}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — cannot run the foresight pass. "
              "Run this in CI where the secret is available.")
        return 1

    snap = load_snapshot(settings.path("snapshot_path"))
    if snap is None:
        print(f"No snapshot at {settings.path('snapshot_path')}. Run a refresh first.")
        return 1
    prev = load_snapshot(settings.path("snapshot_path").with_name("snapshot_prev.json"))

    diff = diff_snapshots(snap, prev)
    ctx = foresight.load_context(settings.root / fcfg.get("context_path", "config/context.md"))
    digest = foresight.build_signal_digest(snap, diff)

    print(RULE)
    print("SIGNAL DIGEST (what feeds the synthesis)")
    print(RULE)
    print(json.dumps(digest, indent=2, ensure_ascii=False))
    print()
    print(RULE)
    print(f"SOCIETAL CONTEXT ({len(ctx)} chars from config/context.md)")
    print(RULE)
    print(ctx if ctx.strip() else "(empty — synthesis will use the scanning framework only)")
    print()

    fg = foresight.synthesize_foresight_gap(
        digest, ctx, api_key, acfg.get("model", "claude-opus-4-8"),
        int(fcfg.get("max_risks", 4)),
    )
    if not fg:
        print("Foresight synthesis returned nothing (see warnings above).")
        return 1

    risks = fg.get("risks", [])
    print(RULE)
    print(f"GENERATED FORESIGHT-GAP RISKS ({len(risks)})")
    print(RULE)
    for i, r in enumerate(risks, 1):
        print(f"\n[{i}] {r.get('risk', '(no statement)')}")
        if r.get("domains_crossed"):
            print(f"    Seam (domains crossed): {', '.join(r['domains_crossed'])}")
        print(f"    Derived from: {r.get('derived_from', '')}")
        if r.get("source_topics"):
            print(f"    Source topics: {', '.join(r['source_topics'])}")
        if r.get("source_arxiv_ids"):
            print(f"    Source arxiv ids: {', '.join(r['source_arxiv_ids'])}")
        print(f"    Why under-discussed: {r.get('why_underdiscussed', '')}")
        print(f"    Mechanism: {r.get('mechanism', '')}")
        print(f"    Leading indicator: {r.get('leading_indicator', '')}")
        print(f"    Calibration: {r.get('calibration', '')}")
        print(f"    Extrapolation: {r.get('extrapolation', '')}")
    print("\n" + RULE)
    print("Tune the prompt in signal_lag/analysis/foresight.py, then re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

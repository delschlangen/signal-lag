"""Pull fresh data and rebuild data/snapshot.json.

Run by the weekly GitHub Action (and usable locally). Pulls real arXiv +
OpenAlex data unless --use-fixtures is passed. Volume caps can be overridden via
env vars so CI runs stay within reasonable time:

  SIGNAL_LAG_MAX_PER_PERIOD  arXiv papers per category per quarter (default: settings.yaml)
  SIGNAL_LAG_OPENALEX_MAX    max papers to enrich via OpenAlex (default: settings.yaml)
  SIGNAL_LAG_YEARS_BACK      rolling window length in years
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from signal_lag.config import load_all  # noqa: E402
from signal_lag.ingest.pipeline import ingest  # noqa: E402
from signal_lag.snapshot import (  # noqa: E402
    augment_foresight, build_snapshot, load_snapshot, save_snapshot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("refresh")


def _apply_env_overrides(settings) -> None:
    ing = settings.raw["ingestion"]
    if os.getenv("SIGNAL_LAG_MAX_PER_PERIOD"):
        ing["max_per_period"] = int(os.environ["SIGNAL_LAG_MAX_PER_PERIOD"])
    if os.getenv("SIGNAL_LAG_OPENALEX_MAX"):
        ing["openalex_max_enrich"] = int(os.environ["SIGNAL_LAG_OPENALEX_MAX"])
    if os.getenv("SIGNAL_LAG_YEARS_BACK"):
        ing["date_range"]["years_back"] = int(os.environ["SIGNAL_LAG_YEARS_BACK"])
    # Optional Semantic Scholar API key (greatly improves enrichment reliability).
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        ing.setdefault("semantic_scholar", {})["api_key"] = key
    # Optional Anthropic key enables the weekly Claude analysis layer.
    akey = os.getenv("ANTHROPIC_API_KEY")
    if akey:
        settings.raw.setdefault("analysis", {})["api_key"] = akey


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-fixtures", action="store_true",
                    help="build from bundled fixtures instead of live data")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the cache before ingesting")
    ap.add_argument("--foresight-only", action="store_true",
                    help="re-run ONLY the Foresight Gap pass against the existing "
                         "snapshot + current config/context.md (no data pull)")
    args = ap.parse_args(argv)

    settings, taxonomy = load_all()
    _apply_env_overrides(settings)

    out = ROOT / "data" / "snapshot.json"

    if args.foresight_only:
        snap = load_snapshot(out)
        if snap is None:
            log.error("No snapshot at %s — run a full refresh first.", out)
            return 1
        prev = load_snapshot(out.with_name("snapshot_prev.json"))
        snap = augment_foresight(settings, snap, prev)
        fg = (snap.get("analysis") or {}).get("foresight_gap")
        # Write in place WITHOUT archiving (don't disturb snapshot_prev.json).
        out.write_text(json.dumps(snap, indent=1, ensure_ascii=False), encoding="utf-8")
        n = len(fg.get("risks", [])) if fg else 0
        log.info("Foresight-only refresh wrote %s (%d risks, %d context chars)",
                 out, n, fg.get("n_context_chars", 0) if fg else 0)
        return 0 if fg else 1

    if args.fresh:
        db = settings.path("db_path")
        if db.exists():
            db.unlink()
            log.info("Removed existing cache %s", db)

    log.info("Ingesting (%s)...", "fixtures" if args.use_fixtures else "live arXiv+OpenAlex")
    total = ingest(settings, use_fixtures=args.use_fixtures, enrich=not args.use_fixtures)
    log.info("Cache holds %d papers", total)

    mode = "fixtures" if args.use_fixtures else "live"
    snapshot = build_snapshot(settings, taxonomy, mode=mode)
    save_snapshot(snapshot, out)
    m = snapshot["meta"]
    log.info("Wrote %s: %d papers, %s..%s, %d/%d pairings flagged, backend=%s",
             out, m["n_papers"], m["date_start"], m["date_end"],
             m["n_flagged"], m["n_pairings"], m["backend"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

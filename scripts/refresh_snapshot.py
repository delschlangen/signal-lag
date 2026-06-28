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

from signal_lag.config import load_all  # noqa: E402
from signal_lag.ingest.pipeline import ingest  # noqa: E402
from signal_lag.snapshot import build_snapshot, save_snapshot  # noqa: E402

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-fixtures", action="store_true",
                    help="build from bundled fixtures instead of live data")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the cache before ingesting")
    args = ap.parse_args(argv)

    settings, taxonomy = load_all()
    _apply_env_overrides(settings)

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
    out = ROOT / "data" / "snapshot.json"
    save_snapshot(snapshot, out)
    m = snapshot["meta"]
    log.info("Wrote %s: %d papers, %s..%s, %d/%d pairings flagged, backend=%s",
             out, m["n_papers"], m["date_start"], m["date_end"],
             m["n_flagged"], m["n_pairings"], m["backend"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

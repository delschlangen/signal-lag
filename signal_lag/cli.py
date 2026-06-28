"""signal-lag command line interface.

Subcommands:
  ingest    Pull arXiv papers (+OpenAlex enrich) into the SQLite cache.
            Use --use-fixtures to load the bundled offline dataset instead.
  enrich    Backfill OpenAlex citation/affiliation data for cached papers.
  analyze   Run the full analysis and write the markdown brief.
  signals   Print the BLUF findings to the terminal.
  dashboard Launch the Streamlit dashboard.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .analysis.runner import run_analysis
from .config import load_all
from .ingest.pipeline import enrich_citations, ingest


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args) -> int:
    settings, _ = load_all()
    total = ingest(settings, use_fixtures=args.use_fixtures, enrich=not args.no_enrich)
    print(f"Cache now holds {total} papers ({settings.path('db_path')}).")
    return 0


def cmd_enrich(args) -> int:
    settings, _ = load_all()
    n = enrich_citations(settings)
    print(f"Enriched {n} papers via OpenAlex.")
    return 0


def cmd_analyze(args) -> int:
    settings, taxonomy = load_all()
    results = run_analysis(settings, taxonomy)
    out = settings.path("brief_output")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(results["brief"], encoding="utf-8")
    print(f"Analyzed {results['meta']['n_papers']} papers "
          f"(backend: {results['meta']['backend']}).")
    print(f"Brief written to {out}.")
    flagged = [d for d in results["divergence"] if d["lagging"]]
    print(f"Divergence pairings flagged: {len(flagged)}/{len(results['divergence'])}.")
    return 0


def cmd_signals(args) -> int:
    settings, taxonomy = load_all()
    results = run_analysis(settings, taxonomy)
    print(results["brief"])
    return 0


def cmd_dashboard(args) -> int:
    import subprocess
    from pathlib import Path

    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signal-lag", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="info-level logging")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="pull papers into the cache")
    pi.add_argument("--use-fixtures", action="store_true", help="load bundled offline dataset")
    pi.add_argument("--no-enrich", action="store_true", help="skip OpenAlex enrichment")
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("enrich", help="backfill OpenAlex enrichment")
    pe.set_defaults(func=cmd_enrich)

    pa = sub.add_parser("analyze", help="run analysis + write brief")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("signals", help="print findings")
    ps.set_defaults(func=cmd_signals)

    pd = sub.add_parser("dashboard", help="launch Streamlit dashboard")
    pd.set_defaults(func=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

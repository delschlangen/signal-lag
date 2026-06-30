"""Configuration loading for signal-lag.

Loads ``config/settings.yaml`` and ``config/taxonomy.yaml`` into light dataclasses
so the rest of the codebase reads typed attributes instead of nested dicts.
Everything tunable lives in YAML; this module only parses and validates it.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dateutil.relativedelta import relativedelta

# Repo root = parent of the signal_lag package directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Topic:
    key: str
    label: str
    kind: str  # "safety" | "capability"
    seeds: list[str]


@dataclass
class Pairing:
    name: str
    capability: str  # Topic.key
    safety: str      # Topic.key


@dataclass
class Taxonomy:
    safety_topics: list[Topic]
    capability_topics: list[Topic]
    pairings: list[Pairing]
    negativity_seeds: list[str] = field(default_factory=list)
    # Dual-use harm/misuse "vectors" — a PARALLEL lens, tagged independently
    # (source="harm"). Deliberately NOT part of `all_topics` so they don't dilute the
    # capability/safety research tagging or the research label_map.
    harm_topics: list[Topic] = field(default_factory=list)
    tag_threshold: float = 0.28
    max_tags_per_paper: int = 3

    @property
    def all_topics(self) -> list[Topic]:
        return self.safety_topics + self.capability_topics

    def topic(self, key: str) -> Topic | None:
        for t in self.all_topics:
            if t.key == key:
                return t
        return None


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path = ROOT

    # --- ingestion ---
    @property
    def arxiv_categories(self) -> list[str]:
        return self.raw["ingestion"]["arxiv_categories"]

    @property
    def date_range(self) -> tuple[dt.date, dt.date]:
        dr = self.raw["ingestion"]["date_range"]
        end = (
            dt.date.fromisoformat(dr["end"])
            if dr.get("end")
            else dt.date.today()
        )
        if dr.get("start"):
            start = dt.date.fromisoformat(dr["start"])
        else:
            start = end - relativedelta(years=int(dr.get("years_back", 3)))
        return start, end

    @property
    def max_results_per_category(self) -> int:
        return int(self.raw["ingestion"].get("max_results_per_category", 1500))

    @property
    def max_per_period(self) -> int:
        """Papers per category per quarter for temporally-stratified sampling."""
        return int(self.raw["ingestion"].get("max_per_period", 150))

    @property
    def arxiv_page_size(self) -> int:
        return int(self.raw["ingestion"]["arxiv_page_size"])

    @property
    def arxiv_request_delay_seconds(self) -> float:
        return float(self.raw["ingestion"]["arxiv_request_delay_seconds"])

    @property
    def backoff_schedule(self) -> list[float]:
        return [float(x) for x in self.raw["ingestion"]["backoff_schedule"]]

    @property
    def openalex_mailto(self) -> str:
        return self.raw["ingestion"].get("openalex_mailto", "")

    @property
    def openalex_max_enrich(self) -> int:
        return int(self.raw["ingestion"].get("openalex_max_enrich", 0))

    @property
    def semantic_scholar(self) -> dict[str, Any]:
        return self.raw["ingestion"].get("semantic_scholar", {"enabled": False})

    @property
    def openreview(self) -> dict[str, Any]:
        return self.raw["ingestion"].get("openreview", {"enabled": False})

    @property
    def blogs(self) -> dict[str, Any]:
        return self.raw["ingestion"].get("blogs", {"enabled": False})

    @property
    def analysis(self) -> dict[str, Any]:
        """Optional weekly LLM analysis (Anthropic / Claude) config."""
        return self.raw.get("analysis", {"enabled": False})

    # --- generic section accessor ---
    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    # --- paths (resolved against repo root) ---
    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]


def load_settings(path: Path | None = None) -> Settings:
    path = path or (CONFIG_DIR / "settings.yaml")
    return Settings(raw=_read_yaml(path))


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or (CONFIG_DIR / "taxonomy.yaml")
    raw = _read_yaml(path)

    def _topics(items: list[dict[str, Any]]) -> list[Topic]:
        return [
            Topic(
                key=it["key"],
                label=it["label"],
                kind=it["kind"],
                seeds=list(it["seeds"]),
            )
            for it in (items or [])
        ]

    tax_cfg = raw.get("taxonomy_settings", {})  # optional override block
    return Taxonomy(
        safety_topics=_topics(raw.get("safety_topics", [])),
        capability_topics=_topics(raw.get("capability_topics", [])),
        pairings=[
            Pairing(name=p["name"], capability=p["capability"], safety=p["safety"])
            for p in raw.get("pairings", [])
        ],
        negativity_seeds=list(raw.get("negativity_seeds", [])),
        harm_topics=_topics(raw.get("harm_topics", [])),
        tag_threshold=float(tax_cfg.get("tag_threshold", 0.28)),
        max_tags_per_paper=int(tax_cfg.get("max_tags_per_paper", 3)),
    )


def load_all(
    settings_path: Path | None = None, taxonomy_path: Path | None = None
) -> tuple[Settings, Taxonomy]:
    settings = load_settings(settings_path)
    taxonomy = load_taxonomy(taxonomy_path)
    # Pull taxonomy tag thresholds from settings.yaml if present (single source).
    tax_settings = settings.section("taxonomy")
    if tax_settings:
        taxonomy.tag_threshold = float(
            tax_settings.get("tag_threshold", taxonomy.tag_threshold)
        )
        taxonomy.max_tags_per_paper = int(
            tax_settings.get("max_tags_per_paper", taxonomy.max_tags_per_paper)
        )
    return settings, taxonomy

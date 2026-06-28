# signal-lag

Patent-landscape-style **foresight on AI-safety research**. It treats research
papers the way patent analysts treat filings — tracking *what* is being worked on,
*how fast*, *by whom*, and crucially *what isn't* — to surface where the field is
heading and where **safety attention lags capability**.

The headline output is the **capability-vs-safety divergence**: for paired topics
(e.g. *agentic/autonomy capability* ↔ *agentic monitoring*), it measures whether
capability research is accelerating while the paired safety work stays flat.

## 🚀 Try it live

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://signal-lag-aw6rmrmp65nit9m9f8wmcq.streamlit.app)

**▶️ Live demo: https://signal-lag-aw6rmrmp65nit9m9f8wmcq.streamlit.app**

The hosted dashboard self-seeds from a bundled synthetic dataset, so it works
immediately with no setup. Hosting is free on [Streamlit Community
Cloud](https://share.streamlit.io); to run your own copy, see
[Deploy your own](#deploy-your-own).

---

## What it does

1. **Ingestion** — pulls papers from the arXiv API (`cs.AI`, `cs.LG`, `cs.CL`) over a
   configurable date window and enriches them with OpenAlex citation counts, yearly
   citation series, and author affiliations. Everything is cached in local SQLite so
   re-runs don't re-pull. arXiv pagination is rate-limited (~1 req/3s) with
   exponential backoff on 429/503.
2. **Topic modeling via embeddings** — embeds every abstract (sentence-transformers
   `all-MiniLM-L6-v2`), then:
   - **discovers emergent topics** by clustering with HDBSCAN (k-means optional), and
   - **tags papers against a supervised safety taxonomy** (interpretability, CoT
     faithfulness, scalable oversight, deceptive alignment, reward hacking, evals,
     RSI/control, agentic monitoring) via cosine similarity to topic centroids.
3. **Velocity analysis** — submission rate per topic/cluster per quarter; flags
   acceleration/deceleration inflections and newly-forming clusters.
4. **Citation dynamics** — papers with rapid recent citation growth, plus "sleepers":
   previously-quiet papers now spawning downstream work (early-heat signals).
5. **Divergence layer (the product)** — per configured capability↔safety pairing,
   compares recent velocities and flags where capability is accelerating but the
   paired safety topic is flat.
6. **Author/institution flow** — which labs are ramping activity in which subfields
   (a talent-flow leading indicator).
7. **Output** — a Streamlit dashboard (velocity time series, divergence chart,
   emerging/cooling/white-space quadrant, signals panel) plus an exportable
   **markdown brief** of BLUF-style findings.

---

## Methodology notes

- **Why embeddings, not keywords:** keyword filters can only find topics you already
  named. Embedding clusters surface *emergent* directions; the supervised taxonomy is
  an overlay on top, not a replacement.
- **Velocity & inflection:** counts are bucketed by calendar quarter. An inflection
  compares the mean of the last *N* quarters against the prior *N* (default N=2); a
  relative change beyond ±30% is an acceleration/deceleration.
- **Divergence metric:** for each pairing, `gap = capability_growth − safety_growth`.
  A pairing is flagged "lagging" when the gap exceeds the threshold *and* capability
  growth is positive. `volume_ratio` shows how lopsided the absolute volumes are now.
- **Citation heat:** uses OpenAlex `counts_by_year`. "Sleepers" have a low early
  citation share but a high recent share — quiet papers now heating up.
- **Everything is config-driven.** The taxonomy, capability↔safety pairings, date
  range, caps, thresholds, and clustering choice all live in `config/*.yaml` — no code
  edits needed to retune.

---

## Install

```bash
# Core: runs the whole pipeline + dashboard using built-in fallbacks
pip install -r requirements.txt

# Optional: best-quality backends (sentence-transformers + HDBSCAN; heavier)
pip install -r requirements-full.txt
```

`sentence-transformers` and `hdbscan` are the **preferred** backends. If
sentence-transformers (or its model download) is unavailable, the embedder
**automatically falls back** to a local scikit-learn TF-IDF + SVD vectorizer; if
HDBSCAN is unavailable it falls back to k-means. The whole pipeline therefore runs
even with only the core install, and even fully offline. The heavy backends are
kept out of the default install so the hosted demo stays lightweight.

---

## Quick start

The repo ships a synthetic fixture dataset so you can run the entire pipeline with no
network access:

```bash
# 1. Load the bundled offline sample (1k+ synthetic papers)
python -m signal_lag.cli ingest --use-fixtures

# 2. Run the analysis and write the markdown brief
python -m signal_lag.cli analyze            # -> data/foresight_brief.md

# 3. Print the BLUF findings to the terminal
python -m signal_lag.cli signals

# 4. Launch the dashboard
python -m signal_lag.cli dashboard          # or: streamlit run signal_lag/dashboard/app.py
```

### Real data

Run where arXiv and OpenAlex are reachable:

```bash
python -m signal_lag.cli ingest             # pull live arXiv + OpenAlex enrichment
python -m signal_lag.cli analyze
python -m signal_lag.cli dashboard
```

`ingest` is idempotent (upserts by arXiv id), so re-running extends the cache rather
than duplicating. Use `--no-enrich` to skip OpenAlex, and `enrich` later to backfill:

```bash
python -m signal_lag.cli ingest --no-enrich
python -m signal_lag.cli enrich
```

Regenerate the fixtures (deterministic) with:

```bash
python scripts/generate_fixtures.py
```

---

## Deploy your own

The dashboard hosts well on **[Streamlit Community Cloud](https://share.streamlit.io)** (free):

1. Push this repo to your GitHub (already done if you're reading this there).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repo/branch, and set the main file to
   `signal_lag/dashboard/app.py` (the [badge above](#-try-it-live) pre-fills this).
4. Deploy. The app installs `requirements.txt` and **auto-seeds the bundled demo
   dataset** on first load — no cache or ingestion step required.

You'll get a URL like `https://<your-app>.streamlit.app`; paste it into the
**Live demo** line near the top of this README.

> The deploy install is intentionally lightweight (TF-IDF + k-means fallbacks), which
> keeps it within the free tier's resource limits. To run the demo on real arXiv data,
> ingest locally and point the deployment at a populated cache, or run it locally with
> `requirements-full.txt`.

---

## Configuration

- `config/settings.yaml` — arXiv categories, date window, caps, rate limits, embedding
  model, clustering algorithm, velocity/citation/divergence thresholds, output paths.
- `config/taxonomy.yaml` — the safety taxonomy, capability topics (each with seed
  phrases), and the **capability↔safety pairings** that drive the divergence layer.
  Edit these to change what the tool tracks.

---

## Project structure

```
config/                 settings.yaml + taxonomy.yaml (all tunables)
signal_lag/
  config.py             YAML -> dataclasses
  models.py             Paper / Author
  cli.py                ingest | enrich | analyze | signals | dashboard
  ingest/               arxiv_client, openalex_client, store (SQLite), pipeline
  analysis/             embeddings, taxonomy, cluster, velocity, citations,
                        divergence, authors, signals, runner
  dashboard/app.py      Streamlit dashboard
  fixtures/             sample_papers.json (offline dataset)
scripts/generate_fixtures.py
tests/                  velocity / divergence / taxonomy logic
```

---

## Tests

```bash
python -m pytest tests/ -q
```

---

## Limitations

- Topic tagging quality depends on the embedding backend; the TF-IDF fallback is
  serviceable but the sentence-transformers path is materially better.
- arXiv covers preprints only; OpenAlex citation coverage lags for very recent papers,
  so recent-quarter citation signals are noisier than older ones.
- The bundled fixtures are synthetic — useful for verifying the pipeline and for demos,
  not for drawing real conclusions. Use live ingestion for that.

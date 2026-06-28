# signal-lag

Patent-landscape-style **foresight on AI-safety research**. It treats research
papers the way patent analysts treat filings — tracking *what* is being worked on,
*how fast*, *by whom*, and crucially *what isn't* — to surface where the field is
heading and where **safety attention lags capability**.

The headline output is the **capability-vs-safety divergence**: for paired topics
(e.g. *agentic/autonomy capability* ↔ *agentic monitoring*), it measures whether
capability research is accelerating while the paired safety work stays flat.

> ### 🧭 Analyst's note — read this first
> signal-lag measures **research attention, not research success.** A spike in a topic
> can mean a *breakthrough* **or** a field *thrashing against a wall* — those look
> identical in volume. So treat this as a **triage instrument** that shows you *where to
> investigate*, not *what to conclude*. The **Sentiment** layer (share of critical /
> limitation-focused papers) exists specifically to help tell those two cases apart.
> Everything shown is **real data** — there is no synthetic or demo content anywhere.

## 🚀 Try it live

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://signal-lag-aw6rmrmp65nit9m9f8wmcq.streamlit.app)

**▶️ Live demo: https://signal-lag-aw6rmrmp65nit9m9f8wmcq.streamlit.app**

The live dashboard runs on **real arXiv + OpenAlex data**, refreshed **weekly** by a
GitHub Action that pulls fresh papers, re-runs the analysis, and publishes a snapshot
the app reads (see [How the live data works](#how-the-live-data-works)). Before the
first refresh runs it shows a bundled synthetic demo dataset. Hosting is free on
[Streamlit Community Cloud](https://share.streamlit.io); to run your own copy, see
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

## Data sources & coverage

signal-lag pulls from several free sources (all config-driven, all fail-soft — a
source being down or rate-limited just omits its signal):

- **arXiv** — the papers themselves (title, abstract, authors, dates), from the
  `cs.AI`, `cs.LG`, `cs.CL` categories (extendable in config).
- **OpenAlex** — enrichment matched to those papers: citation counts, year-by-year
  citation series, and author institutions.
- **Semantic Scholar** — enrichment: TLDR summaries, **influential**-citation counts
  (a sharper heat signal than raw counts), venue, and fields of study.
- **OpenReview** *(optional, config-gated)* — venue papers (e.g. ICLR/NeurIPS) added
  as records with peer-**review scores**, a quality/heat signal papers-only sources lack.
- **Lab/blog RSS** *(optional, config-gated)* — posts from major labs (Anthropic,
  OpenAI, DeepMind, …) as a **capability-leading** signal, shown separately from paper
  velocity since they aren't papers.

Each source is an isolated client; new ones slot in alongside `arxiv_client.py`
without touching the rest of the pipeline.

**What this covers well:** arXiv is where the large majority of frontier AI/ML/NLP
research appears first, so coverage of the fast-moving preprint literature is high.

**Temporally-stratified sampling.** arXiv publishes hundreds of papers per day, so
naively pulling "the newest N" would only span days. Instead ingestion samples up to
`max_per_period` papers **per category per quarter** across the whole window, giving
even time coverage. Velocity therefore tracks each topic's *share* of activity per
quarter (the trend that divergence relies on), not raw absolute counts.

**What it does _not_ cover (yet):** other arXiv categories (`cs.CV`, `cs.CR`,
`cs.RO`, `stat.ML`, …), venues that don't post to arXiv (OpenReview / ICLR / NeurIPS,
ACL Anthology, PMLR, journals), and industry tech reports or lab blog posts. So treat
it as **high coverage of the AI preprint literature, not "everything published."**
The arXiv categories are config-driven (add more in `config/settings.yaml`); adding a
genuinely new source (e.g. OpenReview, Semantic Scholar) means adding a small
ingestion client alongside `arxiv_client.py`.

---

## Methodology notes

### How papers are categorized

Two complementary layers, both **semantic (embedding-based), not keyword matching**:

1. **Supervised tagging against the taxonomy.** Each topic in `taxonomy.yaml` has a
   few **seed phrases**. Those are embedded and averaged into a **centroid** vector per
   topic. Every paper's title+abstract is embedded into the same space, and its
   **cosine similarity** to each centroid is computed; if it clears `tag_threshold`
   (default 0.28) the paper gets that tag (up to `max_tags_per_paper`). Because it's
   semantic, a paper about "models that strategically hide their objectives" tags to
   *deceptive alignment* even without those literal words.
2. **Unsupervised clustering for emergent topics.** All paper embeddings are clustered
   with **HDBSCAN** (auto-discovers the cluster count, marks outliers as noise). Each
   cluster is auto-labeled by its most distinctive terms (c-TF-IDF). This surfaces
   directions that *aren't* in the predefined taxonomy.

Embeddings use `all-MiniLM-L6-v2` (sentence-transformers); offline it falls back to a
TF-IDF + SVD vectorizer. Categorization quality depends on the seed phrases and
threshold — both tunable in YAML.

### Other definitions

- **Why embeddings, not keywords:** keyword filters can only find topics you already
  named. Embedding clusters surface *emergent* directions; the supervised taxonomy is
  an overlay on top, not a replacement.
- **Negative/sentiment signal:** a "negativity" centroid is built from limitation/
  failure seed phrases (`negativity_seeds` in `taxonomy.yaml`). A paper is *critical*
  when its abstract embedding is close to that centroid; per topic we track the
  critical **share** and its quarter-over-quarter trend. Rising critical share (esp.
  with flat volume) is flagged as eroding confidence.
- **Labs-lead signal:** lab/blog posts are embedded and tagged to topics, then shown
  against the paired safety topic's velocity — "labs announce → safety responds on a
  delay → the delay is the risk window."
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

## Dashboard tabs explained

- **📋 Weekly Summary** — the BLUF page. Data freshness, *what changed since the last
  refresh* (new safety-lag alerts, newly accelerating topics, new citation sleepers),
  and the headline pairings where safety is lagging, each with source links.
- **⚖️ Divergence** — for every capability↔safety pair, the recent growth rate of each
  side as horizontal bars. A long capability bar next to a short safety bar = safety
  lagging. This is the core product.
- **📈 Velocity** — papers per quarter per topic over time, plus an inflection table of
  which topics accelerated or decelerated. Momentum (attention, not success).
- **🔬 Sentiment** — the **negative-signal layer**: the share of *critical / negative /
  limitation-focused* papers within each topic, and whether that share is **rising**.
  A rising critical share while volume is flat is an early warning that a field may be
  losing confidence in an approach — detected via embeddings (cosine similarity to
  negativity seed phrases), not keywords.
- **🧭 Quadrant** — topics plotted by recent volume (x) vs. growth (y): *emerging*
  (small but surging), *hot* (big and growing), *cooling* (shrinking), *white-space*
  (quiet). A strategic map of the field.
- **🔍 Sources** — the receipts: actual arXiv papers behind each topic, plus
  rapid-citation-growth and "sleeper" papers, all linked.
- **🚨 Signals** — the full ranked list of auto-generated BLUF findings, with the
  downloadable markdown brief.

---

## How the live data works

The dashboard never pulls data at page-load time — that would be slow and
network-dependent. Instead:

1. **`.github/workflows/refresh.yml`** runs weekly (Mondays 06:00 UTC) and on demand.
2. It pulls real arXiv + OpenAlex data, runs the full analysis, and writes
   **`data/snapshot.json`** via `scripts/refresh_snapshot.py`.
3. It commits that snapshot; Streamlit Community Cloud redeploys on the push.
4. The app reads the snapshot — fast — and the **Weekly Summary** tab diffs it against
   the previous snapshot to show what changed.

Trigger it manually anytime from the repo's **Actions → Weekly data refresh → Run
workflow** (you can tune the per-category and OpenAlex caps there). To produce a
snapshot locally:

```bash
python scripts/refresh_snapshot.py            # real data (needs network)
python scripts/refresh_snapshot.py --use-fixtures   # offline demo snapshot
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
- **No synthetic data on the live tool, ever.** The dashboard only renders a real,
  published snapshot; if none is present it shows an honest "data not available yet"
  message rather than any demo content. Synthetic fixtures exist *solely* for the
  automated test suite (`tests/`) and never reach the site.
- **Sentiment is a proxy.** The critical/negative share is embedding-based and
  approximate — a triage signal for where to read, not a verdict on a field.
- **Lab-announcement history is shallow.** RSS feeds only expose recent posts, so the
  announce-vs-response view reflects current announcements against the paired safety
  topic's velocity, rather than a deep historical lead-time series.
- **Semantic Scholar enrichment needs an API key.** The keyless pool is rate-limited
  and usually returns nothing, so TLDR/influential-citation/venue fields stay empty
  until you add a free key as the repo secret `SEMANTIC_SCHOLAR_API_KEY` (the refresh
  workflow already passes it through). Everything else works without it.
- **Velocity is a temporally-stratified sample**, and the **current incomplete quarter
  is dropped** from trend math so a mid-quarter refresh doesn't read as a slowdown —
  i.e. divergence/inflections reflect the last *complete* quarter.
- OpenReview papers are dated by submission, so they cluster around venue cycles; their
  review scores are captured but not yet surfaced prominently in the UI.

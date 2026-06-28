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
the app reads (see [How the live data works](#how-the-live-data-works)). It only ever
shows a real, published snapshot — never synthetic or demo content. Hosting is free on
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
- **Claude (Anthropic API)** *(optional, config-gated)* — not a data source but an
  **analysis layer**: once per refresh the computed metrics + the real abstracts are
  sent to `claude-opus-4-8`, which writes the analytical headline, the per-tab read,
  and the per-paper *what-it-does / why-it-matters*. Baked into the snapshot; fail-soft
  (skipped with no `ANTHROPIC_API_KEY`).

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

### 10. Foresight Gap synthesis (the 🔮 tab)

A **second weekly Claude pass** (`claude-opus-4-8`, using the Claude API exactly like the
analysis layer in “Data sources” — same client, same fail-soft, same baked-into-snapshot,
no page-load calls). Its job is to surface **novel, not-yet-in-the-news risks** that live
in the **seam between AI research and broader societal forces** — risks no single community
is tracking because they sit between domains.

How it works:
1. **Signal digest** — it pulls the strongest signals already computed this week: flagged
   capability-vs-safety divergences, velocity inflections, rising critical-share
   (eroding-confidence) flags, quadrant emerging/white-space topics, citation sleepers &
   rapid-growth papers, new emergent clusters, recent lab activity, **and what changed
   week-over-week** (so it weights *movement*, not just static state).
2. **Scanning framework** — a fixed STEEP/PESTLE-plus taxonomy of *domains* (Social,
   Technological, Economic, Environmental, Political, Legal/Regulatory,
   Security/Geopolitical, Demographic) so the synthesis is comprehensive by construction
   and never tunnels on technology alone. It defines *dimensions*, never specific trends.
3. **Living societal context** — `config/context.md`, a **user-maintained** file where
   *you* paste the current real-world state across those domains and keep it updated week
   to week. It is **not** baked into code, so it never goes stale; any examples in it are
   **illustrative of the format only — not a prescribed or exhaustive list**, and the
   synthesis is explicitly told never to treat them as the only factors that matter. If
   the file is missing/empty the pass still runs (just without the societal layer).
4. **Synthesis** — Claude returns 2–4 candidate risks, each with a fixed six-part
   structure: **risk statement · derived-from (citing the actual digest signals, so it's
   traceable) · why it's under-discussed · mechanism · leading indicator · calibration ·
   extrapolation** (an honest flag of what goes beyond the data). It's instructed to
   ground every claim in the provided signals and to refuse to restate well-known AI risks.

**These are AI-surfaced candidate hypotheses for an analyst to pressure-test — not
predictions.** The model widens the aperture; human judgment goes on top. The tab shows
the digest, context, and framework that fed each synthesis, so the reasoning is fully
transparent. Config lives under `analysis.foresight` in `settings.yaml` (enable, number of
risks, context-file path). Like the rest of the Claude layer it needs the
`ANTHROPIC_API_KEY` repo secret; without it the tab shows an honest "unavailable" message.

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
kept out of the default install so the hosted app stays lightweight.

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

- **📋 Weekly Summary** — the self-contained briefing. Data freshness, *what changed
  since the last refresh* (new safety-lag alerts, newly accelerating topics, new
  citation sleepers), the **Claude-written analytical headline** (what the widest gap
  means and why it matters) with the driving papers, a plain-language read of every
  other tab so you needn't open them, and the labs-announce→safety-responds view. The
  former *Signals* tab is folded in here (full ranked list + downloadable brief).
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
- **🔮 Foresight Gap** — a second Claude pass that crosses this week's signals with
  broader societal forces to surface **novel, not-yet-in-the-news risks** living in the
  *seam between domains*. Each is a candidate hypothesis (not a prediction) with a six-part
  structure — statement, derived-from, why-under-discussed, mechanism, leading indicator,
  calibration, extrapolation — and it shows the signal digest + societal context + scanning
  framework that fed it. See methodology section 10. Exportable as a brief.
- **🔍 Sources** — the receipts: actual arXiv papers behind each topic (each with a
  short *what-it-does / why-it-matters* note — written by Claude when the analysis
  layer is on, data-derived otherwise), plus rapid-citation-growth and "sleeper" papers,
  all linked.
- **📖 Methodology** — how every layer works, on the live data behind the current snapshot.

---

## How the live data works

The dashboard never pulls data at page-load time — that would be slow and
network-dependent. Instead:

1. **`.github/workflows/refresh.yml`** runs weekly (Mondays 06:00 UTC) and on demand.
2. It pulls real arXiv + OpenAlex data, runs the full analysis (including the optional
   Claude analysis layer when `ANTHROPIC_API_KEY` is set), and writes
   **`data/snapshot.json`** via `scripts/refresh_snapshot.py`.
3. It commits that snapshot; Streamlit Community Cloud redeploys on the push.
4. The app reads the snapshot — fast — and the **Weekly Summary** tab diffs it against
   the previous snapshot to show what changed.

Trigger it manually anytime from the repo's **Actions → Weekly data refresh → Run
workflow** (you can tune the per-category and OpenAlex caps there). To produce a
snapshot locally:

```bash
python scripts/refresh_snapshot.py            # real data (needs network)
python scripts/refresh_snapshot.py --use-fixtures   # offline fixture snapshot (local dev only)
```

---

## Deploy your own

The dashboard hosts well on **[Streamlit Community Cloud](https://share.streamlit.io)** (free):

1. Push this repo to your GitHub (already done if you're reading this there).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repo/branch, and set the main file to
   `signal_lag/dashboard/app.py` (the [badge above](#-try-it-live) pre-fills this).
4. Deploy. The app installs `requirements.txt` and reads the committed
   `data/snapshot.json`. It only ever renders a **real, published snapshot** — if none
   is present it shows an honest "data not available yet" message (run the refresh
   workflow to publish one). No synthetic/demo content is ever shown on the site.

You'll get a URL like `https://<your-app>.streamlit.app`; paste it into the
**Live demo** line near the top of this README.

> The deploy install is intentionally lightweight (TF-IDF + k-means fallbacks), which
> keeps it within the free tier's resource limits. To run it on real arXiv data,
> ingest locally and point the deployment at a populated cache, or run it locally with
> `requirements-full.txt`.

---

## Configuration

- `config/settings.yaml` — arXiv categories, date window, caps, rate limits, embedding
  model, clustering algorithm, velocity/citation/divergence thresholds, output paths.
- `config/taxonomy.yaml` — the safety taxonomy, capability topics (each with seed
  phrases), and the **capability↔safety pairings** that drive the divergence layer.
  Edit these to change what the tool tracks.
- `config/context.md` — the **living societal-context file for the Foresight Gap tab**,
  which *you maintain*. Paste the current real-world state across the STEEP/PESTLE-plus
  domains and keep it updated week to week; the weekly synthesis crosses it with the
  research signals. Examples in it are illustrative of the format only, never exhaustive.

---

## Project structure

```
config/                 settings.yaml + taxonomy.yaml + context.md (all tunables)
signal_lag/
  config.py             YAML -> dataclasses
  models.py             Paper / Author
  cli.py                ingest | enrich | analyze | signals | dashboard
  ingest/               arxiv_client, openalex_client, store (SQLite), pipeline
  analysis/             embeddings, taxonomy, cluster, velocity, citations,
                        divergence, authors, signals, runner, llm, foresight
  dashboard/app.py      Streamlit dashboard
  fixtures/             sample_papers.json (offline dataset)
scripts/                generate_fixtures, refresh_snapshot, foresight_preview
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
- **Foresight-Gap risks are hypotheses, not predictions.** They are AI-surfaced
  candidate risks for an analyst to pressure-test — the model widens the aperture, human
  judgment goes on top. Each card flags its own extrapolation beyond the data, and the
  quality depends on how current you keep `config/context.md`. Needs `ANTHROPIC_API_KEY`.
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
- **The Claude analysis layer needs an API key.** The analytical headline, per-tab
  reads, and per-paper notes are written by `claude-opus-4-8` once per refresh, only
  when the repo secret `ANTHROPIC_API_KEY` is set (the refresh workflow passes it
  through). Without it the layer is skipped and the dashboard falls back to its
  data-derived templated text — everything else works unchanged. Claude only interprets
  the real metrics and abstracts; it never invents data.
- OpenReview papers are dated by submission, so they cluster around venue cycles; their
  review scores are captured but not yet surfaced prominently in the UI.

# signal-lag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Patent-landscape-style **foresight on the AI frontier**. It began as foresight on
AI-safety research — treating papers the way patent analysts treat filings (*what* is
worked on, *how fast*, *by whom*, and crucially *what isn't*) — and has grown into a full
**emerging-risk pipeline**: from the research-trend signal, through **harm/misuse vectors**
and a **scored, evergreen risk register**, to **6–24-month scenarios** and a **real-world
incident benchmark**.

The original spine is the **capability-vs-safety divergence**: for paired topics
(e.g. *agentic/autonomy capability* ↔ *agentic monitoring*), it measures whether capability
research is accelerating while the paired safety work stays flat. On top of that, a
Claude-powered **Foresight** layer turns the signal into novel, web-verified, **scored** risk
calls — re-classified by which real-world **misuse** they could enable, developed into
**scenarios**, **benchmarked against actual incidents**, and exportable as analyst-ready
intelligence-estimate and tabletop packs.

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

The live dashboard runs on **real arXiv + Semantic Scholar data**, refreshed **weekly** by a
GitHub Action that pulls fresh papers, re-runs the analysis, and publishes a snapshot
the app reads (see [How the live data works](#how-the-live-data-works)). It only ever
shows a real, published snapshot — never synthetic or demo content. Hosting is free on
[Streamlit Community Cloud](https://share.streamlit.io); to run your own copy, see
[Deploy your own](#deploy-your-own).

---

## What it does

1. **Ingestion** — pulls papers from the arXiv API (`cs.AI`, `cs.LG`, `cs.CL`, `cs.CR`,
   `cs.CY`, `cs.CV`) over a configurable window, **stratified by quarter** for even time
   coverage, and enriches them via **Semantic Scholar** (citation counts, outgoing
   references, author ids + affiliations, TLDRs, influential-citation counts, venue),
   **OpenReview** (venue papers + peer-review scores), and **lab/blog RSS** (a
   capability-leading signal). (OpenAlex is configured but currently unreachable from
   the CI runner; Semantic Scholar supplies its signals instead.) Cached in local SQLite; rate-limited with backoff. All
   sources are fail-soft — one being down just omits its signal.
2. **Topic modeling via embeddings** — embeds every abstract (sentence-transformers
   `all-MiniLM-L6-v2`, with a TF-IDF+SVD fallback), then **discovers emergent topics**
   (HDBSCAN, k-means fallback) and **tags papers against a supervised taxonomy** of 8
   safety + 6 capability topics via cosine similarity to topic centroids — with
   **per-topic thresholds**, a **high/medium/weak confidence tier per tag**, and a
   weekly **LLM-judged precision audit** per topic (Methodology tab) so over-inclusive
   topics are visible and correctable instead of silently inflating volumes.
3. **Velocity analysis** — submission rate per topic per quarter; flags
   acceleration/deceleration inflections and newly-forming clusters (the current
   incomplete quarter is dropped from trend math), plus a **statistical warning layer**:
   CUSUM persistent-shift detection, change-point quarters, capability→safety lead-lag
   cross-correlation per pairing, and trend-forecast ranges with deviation alerts.
4. **Sentiment / confidence layer** — the share of *critical / limitation-focused*
   papers per topic (embedding recall, **LLM-verified precision** on the recent window),
   whether it's **rising** (early confidence-erosion warning), a volume×critique
   **quadrant map**, Wilson confidence intervals, and **false-confidence alerts**
   (capability growing while self-critique falls and paired safety stays flat).
5. **Citation layer** — verifies cross-domain *borrowing* via real references (Semantic
   Scholar): which capability papers actually **cite** core safety work, not just share its
   vocabulary — plus a capability×safety **cross-pollination matrix**, a **bridge-paper**
   detector, and a **safety-impact leaderboard** (the safety papers capability builders
   actually cite). Citation *velocity* (movers/sleepers) is rebuilt from signal-lag's own
   weekly snapshots of per-paper citation totals, since OpenAlex's yearly series is
   unreachable from the CI runner.
6. **Divergence layer (the headline product)** — per configured capability↔safety
   pairing, flags where capability is accelerating but the paired safety topic is flat —
   with volume-balance vs growth-balance made explicit, **high-confidence volumes** next
   to raw, a **confidence-adjusted gap** (growth weighted by each side's self-critique
   posture), cumulative **monitoring-debt curves**, and the **lab-announcement →
   safety-response lag**: how long the paired safety literature takes to measurably
   answer each lab announcement (median lag, unanswered-after-N-weeks).
7. **Research ecosystem** — institutions from Semantic Scholar author affiliations:
   industry-vs-academia split, top and fastest-growing (institution, topic) pairs, plus
   experimental capability→safety **author migration** tracking.
8. **Weekly Claude analysis** — once per refresh, Claude (`claude-opus-4-8`) reads the
   computed metrics + real abstracts and writes the analytical headline, a per-tab read,
   and a *what-it-does / why-it-matters* note per driving paper (baked into the snapshot).
9. **Foresight layer** (Claude) — the emerging-risk engine, all baked into the snapshot:
   - **Foresight Gap** — crosses the research signal with a living societal-context file to
     surface **novel cross-domain risks**, **web-checks each** for novelty + disputes, and
     backfills for quality (methodology §10).
   - **Harm/misuse vectors** — re-classifies the same papers by which real-world **misuse**
     they could enable, on a 0–24-month horizon (§13).
   - **Scored risk register** — every risk scored by severity × likelihood × exposure ×
     trajectory plus confidence / evidence-strength / actionability tie-breakers (a true
     forced ranking), accumulated into an evergreen register with **watchlist statuses**
     (open / strengthening / weakening / dormant), a persisted **counterevidence trail**,
     filters, a per-risk **drill-down dossier**, and a calibration panel (§14). Each risk
     carries epistemically-labeled claims (observed / inferred / speculative),
     **falsification conditions** (upgrade / downgrade / invalidate if), and a **"so
     what" action map** (eval to run, benchmark to watch, mitigation, owner).
   - **Scenarios + exports** — 6–24-month scenarios from the top risks, plus one-tap
     intelligence-estimate and tabletop-exercise packs (§15).
   - **Incident benchmark** — real, web-sourced incidents (credibility-graded on
     AI-involvement / attribution / source quality, cleaned + de-duplicated) crossed
     against the research signal into a leading-vs-lagging 2×2, with an accruing
     early-warning calibration record (§16).
   - **Enablement map** — capability topics → harm vectors → real incidents as a Sankey
     (edges = papers co-tagged to both tracks; association, not proof).
   - **Plain-terms briefing** — the top risks explained for non-specialists (evidence,
     context, the gap, the tool's own skepticism, bottom line), as a first-class view.
10. **Output** — a Streamlit dashboard: 8 tabs led by a self-contained Weekly Summary,
    an always-visible **BLUF sidebar** (each judgment with an analyst *so-what*), a
    per-tab **"since last refresh"** delta panel, and exports: markdown briefs /
    intelligence estimate / tabletop pack / one-page brief, plus **structured CSV/JSON**
    (register, incidents, velocity, lab-lag, citation matrix, full snapshot).
11. **Operations** — tiered LLM usage (a cheap model for high-volume classification, the
    strong model for synthesis/verification), a 21-day **novelty-verification cache**
    keyed by risk fingerprint, and idempotent per-refresh histories (register, benchmark,
    citation counts) committed by the weekly workflow.

---

## Data sources & coverage

signal-lag pulls from several free sources (all config-driven, all fail-soft — a
source being down or rate-limited just omits its signal):

- **arXiv** — the papers themselves (title, abstract, authors, dates), from the
  `cs.AI`, `cs.LG`, `cs.CL`, `cs.CR`, `cs.CY`, `cs.CV` categories (extendable in config).
- **Semantic Scholar** — the primary enrichment source: citation counts, outgoing
  **references** (for citation-flow verification and the cross-pollination matrix),
  stable author ids + **affiliations** (the research-ecosystem view), TLDR summaries,
  **influential**-citation counts, venue, and fields of study.
- **OpenAlex** *(configured, currently dormant)* — unreachable from the CI runner;
  Semantic Scholar supplies its signals instead. Citation *velocity* is rebuilt from
  signal-lag's own weekly snapshots of per-paper totals.
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

**What it does _not_ cover (yet):** further arXiv categories (`cs.RO`, `stat.ML`, …),
and venues outside arXiv + the configured OpenReview conferences (ACL Anthology, PMLR,
journals). So treat it as **high coverage of the AI preprint literature, not "everything
published."** The arXiv categories and OpenReview venues are config-driven (extend them in
`config/settings.yaml`); adding a genuinely new *source* means a small ingestion client
alongside `arxiv_client.py`.

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
  with flat volume) is flagged as eroding confidence. **Hybrid LLM precision:** the
  embedding centroid alone mistakes *academic negation* ("we **overcome** the
  catastrophic failures of prior methods") for criticism, so the recent-window papers it
  flags critical are **re-checked by Claude** and false positives are downgraded before
  the trend is computed (`sentiment.llm_verify`; bounded to that subset, fail-soft).
- **Citation-flow verification:** the cross-silo "borrowing" claim is checked against
  **real citation references** (each paper's outgoing bibliography, by arXiv id, from
  **Semantic Scholar** — OpenAlex is unreachable from the CI runner) — a capability/applied
  paper that *actually cites* a core safety paper, not one that just shares its vocabulary.
  **Positive-only**: a verified citation is strong evidence; *absence is inconclusive*,
  never "they ignore safety work" (`citation_flow.enabled`).
- **Author migration (experimental):** using **Semantic Scholar author IDs**, authors who
  were capability-dominant historically and whose **recent** papers enter a safety/oversight
  topic are flagged as a capability→safety talent flow — a leading indicator. **Noisy by
  construction** (stratified sample + imperfect IDs): it informs the brief, never gates an
  alert (`analysis.author_migration`).
- **Labs-lead signal:** lab/blog posts are embedded and tagged to topics, then shown
  against the paired safety topic's velocity — "labs announce → safety responds on a
  delay → the delay is the risk window."
- **Velocity & inflection:** counts are bucketed by calendar quarter. An inflection
  compares the mean of the last *N* quarters against the prior *N* (default N=2); a
  relative change beyond ±30% is an acceleration/deceleration.
- **Divergence metric:** for each pairing, `gap = capability_growth − safety_growth`.
  A pairing is flagged "lagging" when the gap exceeds the threshold *and* capability
  growth is positive. `volume_ratio` shows how lopsided the absolute volumes are now.
- **Citation heat:** rebuilt from signal-lag's own weekly snapshots of per-paper
  Semantic Scholar citation totals (movers = biggest week-over-week gains). "Sleepers" have a low early
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
4. **Live web brief (current ground truth)** — because fast arXiv data crossed with a
   hand-maintained `context.md` risks anchoring on stale facts, an optional **pre-synthesis
   web search** (`analysis.foresight.live_context`) pulls the *current, dated* status of the
   flagged topics' real-world developments. The synthesis verifies any date/policy claim
   against this live brief and prefers it over the standing context where they conflict —
   complementing, never replacing, the analyst's file. Fail-soft; shown in its own expander.
5. **Synthesis** — Claude returns 2–4 candidate risks, each with a fixed six-part
   structure: **risk statement · derived-from (citing the actual digest signals, so it's
   traceable) · why it's under-discussed · mechanism · leading indicator · calibration ·
   extrapolation** (an honest flag of what goes beyond the data). It's instructed to
   ground every claim in the provided signals and to refuse to restate well-known AI risks.
   It also gets the **citation-verified borrowings** (use as evidence, not vocabulary;
   absence inconclusive) and the **experimental author-migration** signal (soft
   corroboration only).

The synthesis is tuned to the tool's real strength:
- **Research-trend anchor (the proprietary edge).** Each risk leads with a signal only this
  tool has — a safety subfield whose velocity is *decelerating* or whose *critical share is
  rising* — then crosses it with the societal context. The research signal makes it novel;
  the societal cross makes it a real-world risk.
- **Cross-silo seams.** It prioritizes risks where two distinct expert communities each
  track one half and nobody connects them, and names which community sees which half.
- **Framing inversions.** It rewards risks that invert a trend everyone treats as simply
  good or bad (e.g. transparency regulation freezing in an unsound standard).
- **Calibrated on contested ground.** When a risk leans on a disputed or inferential claim
  — including over-reading the tool's own trend metric as causation — confidence is lowered
  explicitly, never laundered into a confident claim.

**Novelty verification.** After synthesis, each candidate is run through a web search
(Claude's server-side web search) that looks for **both confirming and disputing** coverage,
returning a *prior-coverage check*, a verified novelty rating (*genuinely unsurfaced /
partially anticipated / already widely discussed*), disputing sources, and a recalibrated
confidence. Already-discussed risks are tucked into a collapsed expander (flagged, not
hidden). **Quality over quantity:** if too few candidates survive verification as fresh,
the synthesis automatically re-runs for *different* seams (up to `max_rounds`) and verifies
those too — so a week where several turn out already-covered still surfaces genuine ones.
Verifications run in parallel and are baked into the snapshot (cached — never at page load).
This is the calibrated posture: generate candidate risks, *then check them against current
coverage before surfacing them* — distinguishing a genuine seam from something that just
isn't in the index yet.

**These are AI-surfaced candidate hypotheses for an analyst to pressure-test — not
predictions.** The model widens the aperture; human judgment goes on top. The tab shows
the digest, context, framework, and per-risk prior-coverage check, so the reasoning is fully
transparent. Config lives under `analysis.foresight` in `settings.yaml` (enable, number of
risks, context-file path, `verify_novelty`, web-search tool version). Like the rest of the
Claude layer it needs the `ANTHROPIC_API_KEY` repo secret; without it the tab shows an
honest "unavailable" message.

### 11. "This week" lens + History

The quarterly view is a slow-moving baseline (3 years of quarterly trends). Two layers add
a timely and a longitudinal dimension:

- **"This week" toggle (Summary + Foresight Gap).** Alongside the overall view, a toggle
  analyzes *only* the papers submitted in the last `window_days` (default 7): topic counts
  (safety/capability), a focused Claude "what landed this week" summary, notable papers, and
  a full **web-verified** this-week Foresight Gap. It's anchored on the quarterly research-
  trend signal (as backdrop) then crossed with *this week's* papers + the societal context.
  The quarterly charts are untouched. An extra recent-window arXiv pull
  (`recent_topup_days`) guarantees the 7-day set is complete even if a category exceeds the
  quarterly cap.
- **History tab (📜).** Every refresh appends a **compact** briefing — headline, top
  foresight gaps (overall + this week), rising sentiment, and what-changed counts — to
  `data/history.json` (idempotent per date). The tab renders a metrics-over-time chart
  (safety-lag alerts per refresh) plus a reverse-chronological list. Records are compact by
  design (no abstracts/raw data) so the file stays lean; full per-week data is recoverable
  from the git history of `data/snapshot.json`.

Config lives under `analysis.weekly` in `settings.yaml` (`enabled`, `window_days`,
`recent_topup_days`, and lighter `max_risks`/`min_surfaced`/`max_rounds` for the this-week
foresight). Fail-soft like the rest of the Claude layer.

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

Run where arXiv and Semantic Scholar are reachable:

```bash
python -m signal_lag.cli ingest             # pull live arXiv + Semantic Scholar enrichment
python -m signal_lag.cli analyze
python -m signal_lag.cli dashboard
```

`ingest` is idempotent (upserts by arXiv id), so re-running extends the cache rather
than duplicating. Use `--no-enrich` to skip enrichment, and `enrich` later to backfill:

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

- **📋 Weekly Summary** — the self-contained briefing, in inverted-pyramid order:
  the **Claude-written analytical headline** (what the widest gap means and why it
  matters) with the driving papers, then **the week's best (web-verified novel) foresight
  gaps**, then *what changed since the last refresh* (new safety-lag alerts, newly
  accelerating topics, new citation sleepers), then a plain-language read of every other
  tab so you needn't open them, plus the labs-announce→safety-responds view. The former
  *Signals* tab is folded in here (full ranked list + downloadable brief).
- **⚖️ Divergence** — for every capability↔safety pair, the recent growth rate of each
  side as horizontal bars (long capability + short safety = safety lagging). The table
  separates **volume balance** from **growth balance**, shows **high-confidence volumes**
  next to raw, ±1σ bands on growth, and a **confidence-adjusted gap**; below it,
  cumulative **monitoring-debt curves** and the **🛰️ lab-announcement → safety-response
  lag** (median weeks to a measurable safety-literature response, per capability, with
  unanswered-after-N-weeks). This is the core product.
- **📈 Velocity** — papers per quarter per topic over time, an inflection table, a
  **this-week momentum vs expected** table (Poisson z-scores), **🧪 statistical
  detectors** (CUSUM shifts, change-point quarters, capability→safety lead-lag
  correlation, forecast-range deviations), **and the strategic map** (recent volume ×
  growth: *emerging* / *hot* / *cooling* / *white-space*). Momentum (attention, not
  success).
- **🔬 Sentiment** — the **negative-signal layer**: the share of *critical /
  limitation-focused* papers per topic (embedding recall, LLM-verified precision on the
  recent window) with Wilson 95% intervals, whether it's **rising**, 🟣
  **false-confidence alerts** (capability growing + self-critique falling + paired
  safety flat), and a **volume × critique quadrant map** (growing-&-straining /
  growing-&-confident / contracting-&-critical / fading).
- **🔮 Foresight** — a second Claude pass that anchors on the tool's own research-trend
  signal and crosses it with broader societal forces to surface **novel, not-yet-in-the-news
  risks** living in the *seam between domains*. Each candidate is then **web-checked against
  current coverage** and given a verified novelty rating (genuinely unsurfaced / partially
  anticipated / already widely discussed) with a *prior-coverage check* and disputing
  sources — already-discussed ones are flagged and demoted, not hidden. Each is a candidate
  hypothesis (not a prediction) naming which communities see which half of the problem, with
  mechanism, leading indicator, calibration, and an explicit extrapolation line. See
  methodology section 10. Exportable as a brief. A **⚠️ Harm vectors** toggle on this tab adds
  the **dual-use lens**: the same frontier papers re-classified by which real-world **misuse**
  they could enable (cyber-offense, bio/chem uplift, influence ops, scams, agentic misuse,
  surveillance, model-weight exfiltration, jailbreak/guardrail-evasion, child safety,
  harassment), each with its own velocity — *"which harms is the literature quietly making
  easier, and how fast?"* over a **0–24 month** horizon. Accelerating harm vectors feed the
  synthesis to frame **capability→harm enablement** risks. A foresight signal over research,
  **not** on-platform abuse telemetry. Harm taxonomy: `config/taxonomy.yaml` (`harm_topics`).
  Each risk also carries **epistemically-labeled claims** (🔵 observed · 🟠 inferred · ⚪
  speculative), **falsification conditions** (what would upgrade / downgrade / invalidate
  it), and a **"so what" action map**. A **🕸️ enablement map** (Sankey) connects
  capability topics → harm vectors → real incidents.
  A **📋 Risk register** view scores every surfaced risk on **severity × likelihood × exposure
  × trajectory** (priority = severity × likelihood, 1–25) with confidence /
  evidence-strength / actionability **tie-breakers** (a true forced ranking; stale risks
  downgraded), accumulates them into an evergreen `data/risk_register.json` with
  **watchlist statuses** (open / strengthening / weakening / dormant), a persisted
  **counterevidence trail**, triage **filters**, a per-risk **drill-down dossier**, and a
  **calibration panel** — rendered as a **priority matrix** + ranked table. A **🧩 Plain
  terms** view explains the top risks for non-specialists. A **🎬 Scenarios** view develops
  **6–24 month scenarios** from the top risks (drivers, leading indicators, branch points,
  mitigations, ICD-203 estimative likelihood), and two one-tap **exports** turn it all into
  analyst-ready products: a **Strategic Intelligence Estimate** and a **Tabletop-exercise pack**.
  A **🌐 Incidents** view adds the **all-source** layer: real, already-occurred AI-misuse
  **incidents** (web-sourced from the AI Incident Database / OECD / news, verifiable + dated)
  tagged to the harm vectors and **benchmarked** against the research signal into a
  leading-vs-lagging 2×2 (🎯 foresight lead · 🔴 materializing · 🟠 active/known · ⚪ quiet).
- **🔍 Sources** — the receipts: actual arXiv papers behind each topic (each with a
  *what-it-does / why-it-matters* note and a 🟢/🟡/⚪ **tag-confidence badge**, with a
  hide-weak-matches filter), **📈 citation velocity** from signal-lag's own weekly count
  snapshots (movers + sleepers), the **🔗 citation cross-pollination** section
  (capability×safety heatmap, bridge papers, safety-impact leaderboard), and the **🏛️
  research ecosystem** (industry-vs-academia split, top + fastest-growing institutions,
  from Semantic Scholar affiliations).
- **📖 Methodology** — how every layer works, on the live data behind the current
  snapshot — including the weekly **🧪 tagging-precision audit** (per-topic LLM-judged
  precision; the basis for the per-topic thresholds in `config/settings.yaml`) and
  **📤 structured CSV/JSON exports**.
- **📜 History** — a running record of past weekly briefings (headline, top foresight gaps,
  rising sentiment, what-changed) plus a metrics-over-time chart — the longitudinal view a
  single snapshot can't show.

The **Summary, Divergence, Velocity, Sentiment, and Foresight Gap** tabs carry a
**🆕 This week ⟷ 📊 Quarterly** toggle (defaulting to *This week*): alongside the 3-year
quarterly view, "This week" analyzes *only* the papers from the last 7 days — topic counts,
the this-week capability/safety attention split, this-week critical-share, a focused Claude
read, notable papers, and a web-verified this-week Foresight Gap. The quarterly trend math
(velocity/divergence/quadrant) is unchanged and clearly labeled *this quarter* — Quadrant
stays quarterly-only (a 7-day volume×growth map isn't meaningful).

Two cross-tab layers ride on top: a **🎯 BLUF sidebar** (widest safety lag, top register
risk, top new risk, biggest weekly anomaly, harm vector to watch, median lab-response lag —
each with an analyst *so-what*, plus one-tap downloads including a **one-page brief**), and
a **🔄 "since last refresh"** delta panel at the top of each tab (new/resolved alerts,
newly accelerating topics, new warnings, new/rotated risks, new incidents).

---

## How the live data works

The dashboard never pulls data at page-load time — that would be slow and
network-dependent. Instead:

1. **`.github/workflows/refresh.yml`** runs weekly (Mondays 06:00 UTC) and on demand.
2. It pulls real arXiv + Semantic Scholar data, runs the full analysis (including the optional
   Claude analysis layer when `ANTHROPIC_API_KEY` is set), and writes
   **`data/snapshot.json`** via `scripts/refresh_snapshot.py`.
3. It commits that snapshot; Streamlit Community Cloud redeploys on the push.
4. The app reads the snapshot — fast — and the **Weekly Summary** tab diffs it against
   the previous snapshot to show what changed.

Trigger it manually anytime from the repo's **Actions → Weekly data refresh → Run
workflow** (you can tune the per-category and enrichment caps there). To produce a
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
- arXiv covers preprints only; Semantic Scholar rarely indexes brand-new papers, so
  recent-quarter citation/reference signals are noisier than older ones, and
  affiliations are author-reported and sparse (the ecosystem view is indicative, not a
  census).
- **Topic tags are imperfect and now measured.** The weekly LLM-judged audit
  (Methodology tab) estimates per-topic precision; several topics proved over-inclusive
  at the original global threshold — volumes inflated by borderline semantic matches —
  which is why per-topic thresholds, tag-confidence tiers, and high-confidence volumes
  exist. Precision estimates use ~30 samples/topic, so single-week numbers carry ±0.1+
  noise; watch the trend, not one reading.
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

---

## License

**[MIT](LICENSE)** — free to use, copy, modify, and redistribute (including commercially),
with attribution. Fork it, deploy your own, retune the taxonomy/context, build on it.

The **code** is MIT-licensed. The **data** it surfaces comes from third parties under
their own terms — arXiv, OpenAlex, Semantic Scholar, OpenReview, and the labs' blogs —
so respect those sources' licenses and rate limits when you run your own copy.

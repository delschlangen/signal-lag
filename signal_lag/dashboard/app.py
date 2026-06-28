"""Streamlit dashboard for signal-lag.

Reads a precomputed snapshot (``data/snapshot.json``) produced weekly from real
arXiv + OpenAlex (+ OpenReview, lab blogs) data by the refresh GitHub Action.

It only ever renders **real** data: if no live snapshot is present, it shows an
honest "data not available yet" message rather than any synthetic/demo content.

Run: streamlit run signal_lag/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signal_lag.snapshot import diff_snapshots, load_snapshot  # noqa: E402

st.set_page_config(page_title="signal-lag", layout="wide", page_icon="📡")

SNAPSHOT = Path(__file__).resolve().parent.parent.parent / "data" / "snapshot.json"
PREV_SNAPSHOT = SNAPSHOT.with_name("snapshot_prev.json")


# TTL + a cache key derived from the snapshot file's mtime so a refreshed
# snapshot is always picked up, even if the app container isn't redeployed.
@st.cache_data(ttl=1800, show_spinner="Loading foresight snapshot...")
def _load(_cache_key: float):
    return load_snapshot(SNAPSHOT), load_snapshot(PREV_SNAPSHOT)


def lbl(snap, key):
    return snap["label_map"].get(key, key)


_key = SNAPSHOT.stat().st_mtime if SNAPSHOT.exists() else 0.0
snap, prev = _load(_key)

st.title("📡 signal-lag — AI safety research foresight")
st.caption(
    "Patent-landscape-style foresight: topic velocity, sentiment, citation dynamics, "
    "and the gap where safety attention lags capability."
)

# Only ever show real data. No synthetic/demo fallback, ever.
if snap is None or snap.get("meta", {}).get("mode") != "live":
    st.error(
        "**Live data isn't available yet.** The weekly refresh hasn't published a "
        "snapshot to this branch. This dashboard only ever shows real arXiv / OpenAlex / "
        "OpenReview / lab data — no synthetic or demo content is shown. "
        "Run **Actions → Weekly data refresh → Run workflow** (or wait for the Monday run)."
    )
    st.stop()

meta = snap["meta"]
st.success(
    f"🟢 **Live data** · refreshed **{meta['refreshed_at']}** · {meta['n_papers']:,} papers "
    f"({meta['date_start']} → {meta['date_end']}) · {', '.join(meta['categories'])} · weekly."
)

# Analyst's note — front and centre, not buried.
st.info(
    "🧭 **Analyst's note — read me.** signal-lag measures *research attention*, not "
    "*research success*. A spike can mean a breakthrough **or** a field thrashing against "
    "a wall — so treat this as a **triage instrument** that shows *where to investigate*, "
    "not *what to conclude*. The Sentiment tab helps tell those two cases apart.",
    icon="🧭",
)

(tab_summary, tab_div, tab_vel, tab_sentiment, tab_quad,
 tab_sources, tab_signals, tab_method) = st.tabs(
    ["📋 Weekly Summary", "⚖️ Divergence", "📈 Velocity", "🔬 Sentiment",
     "🧭 Quadrant", "🔍 Sources", "🚨 Signals", "📖 Methodology"]
)


def topic_links(snap, topic_key, n=3):
    """Markdown bullet list of recent papers for a topic."""
    items = snap["sources"].get(topic_key, [])[:n]
    return "".join(
        f"\n  - [{p['title']}]({p['url']}) · {p['published']}"
        + (f" · {p['cited_by_count']} cites" if p.get("cited_by_count") else "")
        for p in items
    )


def week_note(text: str):
    """Render a short 'this week / what you're looking at' note atop a tab."""
    st.caption(f"📅 **This week** — {text}")


def paper_signal(p: dict, topic_label: str) -> str:
    """A short, data-derived 'what it signals' line for a paper (no LLM, no fabrication)."""
    bits = []
    if p.get("influential_citations"):
        bits.append(f"{p['influential_citations']} influential citations (real downstream uptake)")
    elif p.get("cited_by_count"):
        bits.append(f"{p['cited_by_count']} citations so far")
    if p.get("venue"):
        bits.append(f"peer-reviewed at {p['venue']}")
    if p.get("source") == "openreview":
        bits.append("from a venue submission")
    why = "; ".join(bits) if bits else "recent work in this area"
    return f"Signals activity in **{topic_label}** — {why}."


# ============================================================ Weekly Summary
with tab_summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", f"{meta['n_papers']:,}")
    c2.metric("Window", f"{meta['date_start'][:7]} → {meta['date_end'][:7]}")
    c3.metric("Safety-lag alerts", f"{meta['n_flagged']} of {meta['n_pairings']}")
    c4.metric("Topics tracked", meta["topics_tracked"])
    srcs = meta.get("source_counts") or {}
    bits = [f"{k}: {v:,}" for k, v in srcs.items()]
    if meta.get("n_posts"):
        bits.append(f"blog posts: {meta['n_posts']}")
    if meta.get("s2_enriched") is not None:
        bits.append(f"Semantic Scholar enriched: {meta['s2_enriched']:,}")
    if bits:
        st.caption(" · ".join(bits))

    st.subheader("What changed since last refresh")
    d = diff_snapshots(snap, prev)
    if d["first_run"]:
        st.info("First snapshot — no prior refresh to compare against yet. "
                "Week-over-week changes will appear here from the next refresh.")
    else:
        st.caption(f"Compared against the snapshot from {d['prev_date']}.")
        if d["new_alerts"]:
            st.markdown("**🚨 New safety-lag alerts this week:**")
            for a in d["new_alerts"]:
                st.markdown(
                    f"- **{lbl(snap, a['capability_topic'])}** is now outpacing "
                    f"**{lbl(snap, a['safety_topic'])}** "
                    f"(capability {a['cap_growth']*100:+.0f}% vs safety {a['saf_growth']*100:+.0f}%)."
                )
        if d["new_accelerations"]:
            st.markdown("**📈 Newly accelerating topics:**")
            for a in d["new_accelerations"]:
                st.markdown(f"- {lbl(snap, a['topic_key'])} (+{a['change']*100:.0f}%)")
        if d["new_sleepers"]:
            st.markdown("**💤 New citation sleepers:**")
            for s in d["new_sleepers"][:5]:
                st.markdown(f"- [{s['title']}]({s['url']}) — {s['cited_by_count']} cites")
        if not (d["new_alerts"] or d["new_accelerations"] or d["new_sleepers"]):
            st.write("No major new signals since the last refresh.")

    st.divider()
    st.subheader("Headline: where safety attention is lagging")
    alerts = [x for x in snap["divergence"] if x["lagging"]]
    if not alerts:
        st.write("No capability/safety divergences cross the alert threshold this period.")
    for a in alerts:
        with st.container(border=True):
            st.markdown(
                f"**{lbl(snap, a['capability_topic'])} → {lbl(snap, a['safety_topic'])}**  \n"
                f"Capability velocity **{a['cap_growth']*100:+.0f}%** vs safety "
                f"**{a['saf_growth']*100:+.0f}%** "
                + (f"· capability runs ~{a['volume_ratio']:.1f}× the safety volume."
                   if a.get("volume_ratio") else "")
            )
            st.markdown(
                "Recent capability papers driving this:"
                + topic_links(snap, a["capability_topic"], 3)
            )

    lab = snap.get("lab_activity") or []
    if lab:
        st.divider()
        st.subheader("🏢 Labs announce → safety research responds (the lag)")
        st.caption(
            "Labs announce new capability directions in blog posts/system cards *before* "
            "the safety literature responds. That lead time is the risk window this tool "
            "is built to measure."
        )
        # Map capability topic -> its pairing row (for the paired safety response level).
        cap_pair = {d["capability_topic"]: d for d in snap.get("divergence", [])}
        # Count recent announcements per topic.
        from collections import Counter
        ann = Counter(p.get("topic") for p in lab if p.get("topic"))
        shown = 0
        for cap_key, d in cap_pair.items():
            n_ann = ann.get(cap_key, 0)
            if n_ann == 0:
                continue
            shown += 1
            st.markdown(
                f"- **{lbl(snap, cap_key)}** — {n_ann} recent lab announcement(s) → "
                f"paired safety response **{lbl(snap, d['safety_topic'])}** running at "
                f"~{d['saf_recent']:.0f} papers/qtr "
                f"(growth {d['saf_growth']*100:+.0f}%)."
            )
        if not shown:
            st.caption("No recent lab announcements mapped to a tracked capability topic.")

        with st.expander(f"Recent lab posts ({len(lab)})"):
            for post in lab[:20]:
                when = f" · {post['published']}" if post.get("published") else ""
                tp = f" · _{lbl(snap, post['topic'])}_" if post.get("topic") else ""
                title = post.get("title") or "(untitled)"
                if post.get("url"):
                    st.markdown(f"- **{post['source']}**: [{title}]({post['url']}){when}{tp}")
                else:
                    st.markdown(f"- **{post['source']}**: {title}{when}{tp}")

    st.divider()
    st.download_button("⬇️ Download full markdown brief", data=snap["brief"],
                       file_name="foresight_brief.md", mime="text/markdown")

# ================================================================ Divergence
with tab_div:
    st.subheader("Capability vs. safety velocity gap")
    _nflag = meta.get("n_flagged", 0)
    week_note(
        "recent growth of each **capability** topic vs its paired **safety** topic. "
        "A long blue bar (capability) next to a short orange bar (safety) = safety lagging. "
        f"**{_nflag} of {meta.get('n_pairings','?')} pairings** cross the safety-lag threshold "
        "this week — those are where capability is pulling ahead of the safety response."
    )
    st.caption("A long blue bar with a short orange bar = safety lagging.")
    div = pd.DataFrame(snap["divergence"]).sort_values("gap")
    if not div.empty:
        names = [n.replace(" vs. ", "<br>vs. ") for n in div["pairing"]]
        fig = go.Figure()
        fig.add_bar(name="Capability growth", y=names, x=div["cap_growth"],
                    orientation="h", marker_color="#4c8bf5")
        fig.add_bar(name="Safety growth", y=names, x=div["saf_growth"],
                    orientation="h", marker_color="#ffa94d")
        fig.update_layout(
            barmode="group", xaxis_title="Recent growth rate (Δ vs prior periods)",
            height=140 + 95 * len(div), template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=20, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        show = div[["pairing", "cap_growth", "saf_growth", "gap", "volume_ratio", "lagging"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No pairings configured.")

# ================================================================== Velocity
with tab_vel:
    st.subheader("Topic submission velocity (papers per quarter)")
    week_note(
        "how many papers per quarter each topic is producing, and whether that rate is "
        "**accelerating or decelerating** (inflection table below). This is *attention*, "
        "not success — pair it with the Sentiment tab to read what a spike means."
    )
    ts = pd.DataFrame(snap["timeseries"])
    if not ts.empty:
        ts["topic"] = ts["topic_key"].map(lambda k: lbl(snap, k))
        ts = ts.sort_values("period")
        totals = ts.groupby("topic")["count"].sum().sort_values(ascending=False)
        default = list(totals.head(6).index)
        chosen = st.multiselect("Topics to plot", list(totals.index), default=default)
        sel = ts[ts["topic"].isin(chosen)]
        fig = px.line(sel, x="period", y="count", color="topic", markers=True,
                      height=520, template="plotly_dark")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      title=None), xaxis_title=None,
                          yaxis_title="papers / quarter", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeseries data.")
    st.markdown("**Velocity inflections**")
    st.dataframe(pd.DataFrame(snap["inflections"]), use_container_width=True, hide_index=True)

# ================================================================== Sentiment
with tab_sentiment:
    st.subheader("Confidence / negative-signal tracker")
    sent = snap.get("sentiment") or {}
    rising = [(k, v) for k, v in sent.items() if v.get("rising")]
    week_note(
        "the **share of critical / limitation-focused papers** within each topic. "
        "Volume tells you how *much* a field is working; this tells you whether it may be "
        "**hitting a wall**. A rising critical share — especially when volume is flat — is an "
        "early warning that confidence in an approach is eroding before it hits headlines. "
        + (f"**{len(rising)} topic(s) flagged this week.**" if rising else
           "No topics cross the rising-critical threshold this week.")
    )
    if sent:
        rows = []
        for k, v in sent.items():
            rows.append({
                "topic": lbl(snap, k),
                "critical_share": round(v.get("critical_share", 0) * 100, 1),
                "recent_share": round(v.get("recent_share", 0) * 100, 1),
                "prior_share": round(v.get("prior_share", 0) * 100, 1),
                "trend_pts": round(v.get("trend", 0) * 100, 1),
                "n_recent": v.get("n_recent", 0),
                "rising_⚠": "⚠️" if v.get("rising") else "",
            })
        sdf = pd.DataFrame(rows).sort_values("trend_pts", ascending=False)
        fig = px.bar(sdf, x="trend_pts", y="topic", orientation="h",
                     color="trend_pts", color_continuous_scale="Reds",
                     labels={"trend_pts": "Critical-share change (pts, recent vs prior)",
                             "topic": ""},
                     height=120 + 26 * len(sdf), template="plotly_dark")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=30), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        if rising:
            st.markdown("**⚠️ Eroding-confidence warnings:**")
            for k, v in rising:
                st.markdown(
                    f"- **{lbl(snap, k)}** — critical share {v['prior_share']*100:.0f}% → "
                    f"{v['recent_share']*100:.0f}% (+{v['trend']*100:.0f} pts, {v['n_recent']} recent papers)"
                )
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.caption("Critical detection is embedding-based (cosine similarity to negative/"
                   "limitation seed phrases), not keyword matching — and it's a proxy, "
                   "so use it to decide where to read, not as a verdict.")
    else:
        st.info("No sentiment data in this snapshot.")

# ================================================================== Quadrant
with tab_quad:
    st.subheader("Emerging / hot / cooling / white-space")
    week_note(
        "a strategic map of every topic by **recent volume** (x) vs **growth** (y): "
        "*emerging* (small but surging — worth watching), *hot* (big and growing), "
        "*cooling* (shrinking), *white-space* (quiet — potential gaps)."
    )
    st.caption("X = recent volume (papers/quarter), Y = growth rate. Hover for topic names.")
    quad = pd.DataFrame(snap["quadrant"])
    if not quad.empty:
        quad["topic"] = quad["topic_key"].map(lambda k: lbl(snap, k))
        fig = px.scatter(
            quad, x="recent_mean", y="change", color="quadrant", size="recent_mean",
            size_max=26, hover_name="topic",
            labels={"recent_mean": "Recent volume (papers/quarter)", "change": "Growth rate"},
            height=560, template="plotly_dark",
        )
        # Label only the standout points to avoid clutter.
        notable = quad[quad["quadrant"].isin(["emerging", "hot", "cooling"])]
        for _, r in notable.iterrows():
            fig.add_annotation(x=r["recent_mean"], y=r["change"], text=r["topic"],
                               showarrow=False, yshift=14, font=dict(size=10))
        fig.add_hline(y=0.3, line_dash="dot", line_color="gray")
        fig.add_vline(x=5, line_dash="dot", line_color="gray")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quadrant data.")
    if snap["new_clusters"]:
        st.markdown("**Newly forming clusters (emergent, unsupervised):**")
        for c in snap["new_clusters"]:
            st.write(f"- {c}")

# =================================================================== Sources
with tab_sources:
    st.subheader("Source papers")
    week_note(
        "the actual papers behind each topic — with a short description and a "
        "**what-it-signals** line so you can see *why* each one is in the picture. "
        "Everything links out to the real source."
    )
    keys = list(snap["label_map"].keys())
    pick = st.selectbox("Topic", keys, format_func=lambda k: lbl(snap, k))
    topic_label = lbl(snap, pick)
    for p in snap["sources"].get(pick, []):
        bits = [p["published"]]
        if p.get("venue"):
            bits.append(p["venue"])
        if p.get("cited_by_count"):
            bits.append(f"{p['cited_by_count']} cites")
        if p.get("influential_citations"):
            bits.append(f"{p['influential_citations']} influential")
        with st.container(border=True):
            st.markdown(f"**[{p['title']}]({p['url']})**  \n" + " · ".join(bits))
            desc = p.get("tldr") or p.get("abstract")
            if desc:
                st.markdown(f"*{desc.rstrip('.')}.*")
            st.markdown(f"**What it signals:** {paper_signal(p, topic_label)}")
    if not snap["sources"].get(pick):
        st.write("No tagged papers for this topic in the current snapshot.")

    st.divider()
    ccol, scol = st.columns(2)
    def _cite_line(r):
        extra = f" · {r['influential_citations']} influential" if r.get("influential_citations") else ""
        ven = f" · {r['venue']}" if r.get("venue") else ""
        return f"- [{r['title']}]({r['url']}) — {r['cited_by_count']} cites{extra}{ven}"

    with ccol:
        st.markdown("**🔥 Rapid citation growth**")
        for r in snap["citations"].get("rapid_growth", [])[:10]:
            st.markdown(_cite_line(r))
    with scol:
        st.markdown("**💤 Sleepers (early-heat)**")
        for r in snap["citations"].get("sleepers", [])[:10]:
            st.markdown(_cite_line(r))

# =================================================================== Signals
with tab_signals:
    st.subheader("All signals (BLUF)")
    sev = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    for s in snap["signals"]:
        st.markdown(f"{sev.get(s['severity'], '⚪')} **{s['headline']}**  \n{s['detail']}")
    with st.expander("Preview full brief"):
        st.markdown(snap["brief"])

# =============================================================== Methodology
with tab_method:
    st.subheader("How signal-lag works")
    cats = ", ".join(meta.get("categories", []) or [])
    srcs = meta.get("source_counts") or {}
    src_line = ", ".join(f"{k}: {v:,}" for k, v in srcs.items()) if srcs else "—"
    st.markdown(
        f"""
This dashboard does **patent-landscape-style foresight** on AI-safety research:
it tracks *what* is being worked on, *how fast*, *by whom*, and — most importantly —
**where safety research lags capability research**.

*This snapshot: {meta.get('n_papers', '?'):,} papers ({src_line}); embeddings via
`{meta.get('backend', '?')}`; window {meta.get('date_start','?')} → {meta.get('date_end','?')}.*

### 1. Data sources
All free, all **fail-soft** (a source being down just omits its signal):
- **arXiv** — papers (title, abstract, authors, dates) from `{cats}`.
- **OpenAlex** — citation counts, year-by-year citation series, author institutions.
- **Semantic Scholar** — TLDRs, *influential*-citation counts, venue, fields (needs an API key).
- **OpenReview** — venue papers + peer-review scores, added as papers.
- **Lab/blog RSS** — posts from major labs as a *capability-leading* signal (kept separate from paper velocity).

### 2. Sampling
arXiv publishes hundreds of papers/day, so ingestion is **stratified by quarter**:
up to *N* papers per category per quarter, giving even coverage across the whole
window. The **current incomplete quarter is dropped** from trend math so a
mid-quarter refresh doesn't look like a slowdown.

### 3. Topic categorization (embeddings, not keywords)
Two complementary layers:
- **Supervised tagging** — each taxonomy topic has seed phrases; these are embedded
  and averaged into a **centroid**. Each paper's abstract is embedded into the same
  space, and tagged to a topic when **cosine similarity** clears a threshold. Semantic,
  so it catches papers that don't use the exact words.
- **Unsupervised clustering** — HDBSCAN over the embeddings surfaces *emergent* topics
  not in the taxonomy (falls back to k-means when it finds too few clusters).

### 4. Velocity
Papers per **quarter** per topic. An **inflection** compares the mean of the last
*N* quarters to the prior *N*; a relative change beyond ±30% counts as
acceleration/deceleration. New clusters appearing in recent quarters are flagged
as emerging.

### 5. Divergence (the headline)
For each configured **capability ↔ safety pairing**,
`gap = capability_growth − safety_growth`. A pairing is flagged **"safety lagging"**
when the gap clears a threshold, capability growth is positive, *and* capability has
enough recent volume (a floor that suppresses noisy tiny-count topics).

### 6. Citation dynamics
From OpenAlex yearly counts: **rapid recent growth**, and **sleepers** — papers that
were quiet early but are now accruing most of their citations (early-heat signals).

### 7. Author / institution flow
Which labs are growing activity in which subfields over time — a talent-flow
leading indicator.

### 8. Signals & brief
Computed metrics are templated into ranked **BLUF findings**, exportable as a
markdown brief (see the Signals tab).

### Caveats
- High coverage of the **AI preprint literature**, not every publisher.
- Velocity tracks each topic's *share* of activity (stratified sample), not raw totals.
- Quality depends on the embedding backend and the taxonomy seed phrases — all
  config-driven in `config/taxonomy.yaml` and `config/settings.yaml`.

Full details: [github.com/delschlangen/signal-lag](https://github.com/delschlangen/signal-lag).
"""
    )

st.caption(
    f"Embedding backend: {meta['backend']} · snapshot v{meta.get('version', 1)} · "
    "signal-lag · [github.com/delschlangen/signal-lag](https://github.com/delschlangen/signal-lag)"
)

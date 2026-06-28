"""Streamlit dashboard for signal-lag.

Reads a precomputed snapshot (``data/snapshot.json``) produced weekly from real
arXiv + OpenAlex data by the refresh GitHub Action. If no snapshot exists yet
(e.g. first deploy, or local dev), it falls back to building one from the bundled
synthetic fixtures so the app always renders.

Tabs: Weekly Summary (default) · Divergence · Velocity · Quadrant · Sources · Signals.

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

from signal_lag.config import load_all  # noqa: E402
from signal_lag.snapshot import (  # noqa: E402
    build_snapshot,
    diff_snapshots,
    load_snapshot,
)

st.set_page_config(page_title="signal-lag", layout="wide", page_icon="📡")

SNAPSHOT = Path(__file__).resolve().parent.parent.parent / "data" / "snapshot.json"
PREV_SNAPSHOT = SNAPSHOT.with_name("snapshot_prev.json")


@st.cache_data(show_spinner="Loading foresight snapshot...")
def _load():
    snap = load_snapshot(SNAPSHOT)
    if snap is None:
        # No published snapshot yet -> build a demo one from fixtures in-memory.
        from signal_lag.ingest.pipeline import ingest

        settings, taxonomy = load_all()
        ingest(settings, use_fixtures=True, enrich=False)
        snap = build_snapshot(settings, taxonomy, mode="fixtures")
    prev = load_snapshot(PREV_SNAPSHOT)
    return snap, prev


def lbl(snap, key):
    return snap["label_map"].get(key, key)


snap, prev = _load()
meta = snap["meta"]
live = meta.get("mode") == "live"

# ----------------------------------------------------------------- header
st.title("📡 signal-lag — AI safety research foresight")
st.caption(
    "Patent-landscape-style foresight: topic velocity, citation dynamics, and the "
    "gap where safety attention lags capability."
)

if live:
    st.success(
        f"🟢 **Live arXiv + OpenAlex data** · refreshed **{meta['refreshed_at']}** · "
        f"{meta['n_papers']:,} papers ({meta['date_start']} → {meta['date_end']}) · "
        f"categories {', '.join(meta['categories'])} · auto-refreshes weekly."
    )
else:
    st.warning(
        "🟡 **Demo data (synthetic).** A live snapshot from real arXiv + OpenAlex data "
        "appears automatically once the weekly refresh job has run.",
        icon="🧪",
    )

tabs = st.tabs(
    ["📋 Weekly Summary", "⚖️ Divergence", "📈 Velocity", "🧭 Quadrant",
     "🔍 Sources", "🚨 Signals"]
)


def topic_links(snap, topic_key, n=3):
    """Markdown bullet list of recent papers for a topic."""
    items = snap["sources"].get(topic_key, [])[:n]
    return "".join(
        f"\n  - [{p['title']}]({p['url']}) · {p['published']}"
        + (f" · {p['cited_by_count']} cites" if p.get("cited_by_count") else "")
        for p in items
    )


# ============================================================ Weekly Summary
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", f"{meta['n_papers']:,}")
    c2.metric("Window", f"{meta['date_start'][:7]} → {meta['date_end'][:7]}")
    c3.metric("Safety-lag alerts", f"{meta['n_flagged']} of {meta['n_pairings']}")
    c4.metric("Topics tracked", meta["topics_tracked"])

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
        st.subheader("🏢 Recent lab activity (capability-leading signal)")
        st.caption("Posts from major lab blogs — capability news often precedes papers.")
        for post in lab[:12]:
            when = f" · {post['published']}" if post.get("published") else ""
            title = post.get("title") or "(untitled)"
            if post.get("url"):
                st.markdown(f"- **{post['source']}**: [{title}]({post['url']}){when}")
            else:
                st.markdown(f"- **{post['source']}**: {title}{when}")

    st.divider()
    st.download_button("⬇️ Download full markdown brief", data=snap["brief"],
                       file_name="foresight_brief.md", mime="text/markdown")

# ================================================================ Divergence
with tabs[1]:
    st.subheader("Capability vs. safety velocity gap")
    st.caption("Recent growth rate of each capability topic and its paired safety topic. "
               "A long blue bar with a short orange bar = safety lagging.")
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
with tabs[2]:
    st.subheader("Topic submission velocity (papers per quarter)")
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

# ================================================================== Quadrant
with tabs[3]:
    st.subheader("Emerging / hot / cooling / white-space")
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
with tabs[4]:
    st.subheader("Source papers")
    st.caption("Representative recent papers behind each tracked topic. Click through to arXiv.")
    keys = list(snap["label_map"].keys())
    pick = st.selectbox("Topic", keys, format_func=lambda k: lbl(snap, k))
    for p in snap["sources"].get(pick, []):
        bits = [p["published"]]
        if p.get("venue"):
            bits.append(p["venue"])
        if p.get("cited_by_count"):
            bits.append(f"{p['cited_by_count']} cites")
        if p.get("influential_citations"):
            bits.append(f"{p['influential_citations']} influential")
        st.markdown(f"- [{p['title']}]({p['url']}) · " + " · ".join(bits))
        if p.get("tldr"):
            st.caption(f"  TL;DR: {p['tldr']}")
    if not snap["sources"].get(pick):
        st.write("No tagged papers for this topic in the current snapshot.")

    st.divider()
    ccol, scol = st.columns(2)
    with ccol:
        st.markdown("**🔥 Rapid citation growth**")
        for r in snap["citations"].get("rapid_growth", [])[:10]:
            st.markdown(f"- [{r['title']}]({r['url']}) — {r['cited_by_count']} cites")
    with scol:
        st.markdown("**💤 Sleepers (early-heat)**")
        for r in snap["citations"].get("sleepers", [])[:10]:
            st.markdown(f"- [{r['title']}]({r['url']}) — {r['cited_by_count']} cites")

# =================================================================== Signals
with tabs[5]:
    st.subheader("All signals (BLUF)")
    sev = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    for s in snap["signals"]:
        st.markdown(f"{sev.get(s['severity'], '⚪')} **{s['headline']}**  \n{s['detail']}")
    with st.expander("Preview full brief"):
        st.markdown(snap["brief"])

st.caption(
    f"Embedding backend: {meta['backend']} · snapshot v{meta.get('version', 1)} · "
    "signal-lag · [github.com/delschlangen/signal-lag](https://github.com/delschlangen/signal-lag)"
)

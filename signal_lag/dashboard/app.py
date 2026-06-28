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

from signal_lag.glossary import CAPABILITY_KEYS, GLOSSARY, SAFETY_KEYS  # noqa: E402
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


def get_analysis(snap):
    return snap.get("analysis") or {}


def tab_analysis(snap, key):
    """The LLM's read of a tab, or None if the analysis layer didn't run."""
    return (get_analysis(snap).get("tabs") or {}).get(key)


def paper_notes(snap):
    """arxiv_id -> {summary, why_it_matters} from the LLM analysis (may be empty)."""
    return {p.get("arxiv_id"): p for p in get_analysis(snap).get("papers", [])}


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
    "**Analyst note:** signal-lag measures *research attention*, not *research success*. "
    "A spike can mean a breakthrough **or** a field thrashing against a wall — so treat "
    "this as a **triage instrument** that shows *where to investigate*, not *what to "
    "conclude*. The Sentiment tab helps tell those two cases apart.",
    icon="🧭",
)

if get_analysis(snap):
    st.caption("🧠 This week's figures were analyzed by **Claude** — look for *Claude's read* "
               "on each tab and the narrative headline below.")

(tab_summary, tab_div, tab_vel, tab_sentiment, tab_quad,
 tab_foresight, tab_sources, tab_method) = st.tabs(
    ["📋 Weekly Summary", "⚖️ Divergence", "📈 Velocity", "🔬 Sentiment",
     "🧭 Quadrant", "🔮 Foresight Gap", "🔍 Sources", "📖 Methodology"]
)


def topic_links(snap, topic_key, n=3):
    """Markdown bullet list of recent papers for a topic, each with a short blurb."""
    items = snap["sources"].get(topic_key, [])[:n]
    out = ""
    for p in items:
        cites = f" · {p['cited_by_count']} cites" if p.get("cited_by_count") else ""
        out += f"\n  - [{p['title']}]({p['url']}) · {p['published']}{cites}"
        d = short_desc(p, 160)
        if d:
            out += f"  \n    _{d}_"
    return out


def week_note(what: str, finding: str | None = None):
    """One line on what the chart is, then a data-driven read of THIS week."""
    st.caption(what)
    if finding:
        st.markdown(f"**📅 This week:** {finding}")


def _fmt_pct(x):
    return f"{x*100:+.0f}%"


def divergence_finding(snap) -> str:
    div = snap.get("divergence", [])
    if not div:
        return "No pairings configured."
    flagged = [d for d in div if d.get("lagging")]
    widest = max(div, key=lambda d: d.get("gap", 0))
    parts = []
    if flagged:
        names = ", ".join(f"**{lbl(snap, d['capability_topic'])}** over "
                          f"**{lbl(snap, d['safety_topic'])}**" for d in flagged[:3])
        parts.append(f"{len(flagged)} pairing(s) show safety lagging — {names}.")
    else:
        parts.append("No pairing crosses the safety-lag threshold — safety is broadly keeping pace.")
    parts.append(
        f"Widest gap: {lbl(snap, widest['capability_topic'])} "
        f"({_fmt_pct(widest['cap_growth'])}/qtr) vs {lbl(snap, widest['safety_topic'])} "
        f"({_fmt_pct(widest['saf_growth'])})."
    )
    # where safety leads
    lead = min(div, key=lambda d: d.get("gap", 0))
    if lead.get("gap", 0) < 0:
        parts.append(f"Safety is *ahead* in {lbl(snap, lead['safety_topic'])} "
                     f"({_fmt_pct(lead['saf_growth'])} vs {_fmt_pct(lead['cap_growth'])}).")
    return " ".join(parts)


def velocity_finding(snap) -> str:
    inf = snap.get("inflections", [])
    if not inf:
        return "No velocity data yet."
    by_change = sorted(inf, key=lambda i: i.get("change", 0), reverse=True)
    by_vol = sorted(inf, key=lambda i: i.get("recent_mean", 0), reverse=True)
    top = by_change[0]
    bot = by_change[-1]
    parts = [
        f"Fastest-accelerating: **{lbl(snap, top['topic_key'])}** "
        f"({_fmt_pct(top['change'])}, ~{top['recent_mean']:.0f}/qtr)."
    ]
    if bot.get("change", 0) < -0.05:
        parts.append(f"Steepest pullback: **{lbl(snap, bot['topic_key'])}** ({_fmt_pct(bot['change'])}).")
    parts.append(f"Largest field by volume: **{lbl(snap, by_vol[0]['topic_key'])}** "
                 f"(~{by_vol[0]['recent_mean']:.0f}/qtr).")
    return " ".join(parts)


def sentiment_finding(snap) -> str:
    sent = snap.get("sentiment", {})
    if not sent:
        return "No sentiment data yet."
    rising = [(k, v) for k, v in sent.items() if v.get("rising")]
    hi = max(sent.items(), key=lambda kv: kv[1].get("recent_share", 0))
    parts = []
    if rising:
        names = ", ".join(f"**{lbl(snap, k)}** (+{v['trend']*100:.0f} pts → "
                          f"{v['recent_share']*100:.0f}% critical)" for k, v in rising[:3])
        parts.append(f"Rising critical share (possible confidence erosion): {names}.")
    else:
        parts.append("No topic shows a rising critical share this week.")
    parts.append(f"Most critical overall: **{lbl(snap, hi[0])}** "
                 f"({hi[1].get('recent_share',0)*100:.0f}% of recent papers).")
    return " ".join(parts)


def quadrant_finding(snap) -> str:
    quad = snap.get("quadrant", [])
    if not quad:
        return "No quadrant data yet."
    groups = {}
    for q in quad:
        groups.setdefault(q.get("quadrant", "?"), []).append(lbl(snap, q["topic_key"]))
    order = ["emerging", "hot", "cooling", "white-space"]
    parts = []
    for g in order:
        if groups.get(g):
            parts.append(f"**{g}**: " + ", ".join(groups[g][:4]))
    return " · ".join(parts) if parts else "All topics established."


def short_desc(item, limit=200) -> str | None:
    """A short human summary for a paper/post item from real fields."""
    text = item.get("tldr") or item.get("abstract") or item.get("summary")
    if not text:
        return None
    text = " ".join(text.split())
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


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


def render_paper_card(p, topic_label):
    """A paper as a bordered card: link + date + metrics, summary, why it matters.

    Prefers the LLM's per-paper summary + 'why it matters' when the weekly Claude
    analysis ran; otherwise falls back to the abstract + a data-derived signal line.
    """
    bits = [p.get("published", "")]
    if p.get("venue"):
        bits.append(p["venue"])
    if p.get("cited_by_count"):
        bits.append(f"{p['cited_by_count']} cites")
    if p.get("influential_citations"):
        bits.append(f"{p['influential_citations']} influential")
    note = paper_notes(snap).get(p.get("arxiv_id")) or {}
    with st.container(border=True):
        st.markdown(f"**[{p['title']}]({p['url']})**  \n" + " · ".join(b for b in bits if b))
        summary = note.get("summary") or short_desc(p, 240)
        if summary:
            st.markdown(f"*{summary}*")
        why = note.get("why_it_matters") or paper_signal(p, topic_label)
        st.markdown(f"**Why it matters:** {why}")


def citation_finding(snap) -> str:
    c = snap.get("citations", {})
    rg, sl = c.get("rapid_growth", []), c.get("sleepers", [])
    parts = []
    if rg:
        parts.append(f"Fastest-rising by citations: **{rg[0]['title']}** "
                     f"({rg[0]['cited_by_count']} cites).")
    if sl:
        parts.append(f"A previously-quiet 'sleeper' now heating up: **{sl[0]['title']}**.")
    return " ".join(parts) if parts else "No standout citation movers this week."


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
    st.subheader("📌 Headline")
    hl = get_analysis(snap).get("headline") or {}
    alerts = sorted([x for x in snap["divergence"] if x["lagging"]],
                    key=lambda a: a["gap"], reverse=True)
    if alerts:
        a = alerts[0]
        cap, saf = lbl(snap, a["capability_topic"]), lbl(snap, a["safety_topic"])
        ratio = (f" — capability is running about **{a['volume_ratio']:.1f}× the volume** of "
                 f"the safety work" if a.get("volume_ratio") else "")
        meaning = f" {hl['meaning']}" if hl.get("meaning") else ""
        cap_focus = f" (focused on {hl['capability_focus']})" if hl.get("capability_focus") else ""
        saf_focus = f", which covers {hl['safety_focus']}" if hl.get("safety_focus") else ""
        st.markdown(
            f"**The biggest safety gap this week is {cap} vs {saf}.**{meaning} Capability "
            f"research there{cap_focus} is growing **{a['cap_growth']*100:+.0f}%/quarter** while "
            f"the paired safety topic ({saf}{saf_focus}) is at **{a['saf_growth']*100:+.0f}%**{ratio}. "
            + (f"In all, **{len(alerts)} of {meta['n_pairings']} pairings** show safety lagging. "
               if len(alerts) > 1 else "")
        )
        if hl.get("why_it_matters"):
            st.markdown(f"**Why this matters:** {hl['why_it_matters']}")
        st.markdown("**The papers driving the capability side:**")
        for p in snap["sources"].get(a["capability_topic"], [])[:3]:
            render_paper_card(p, cap)
    else:
        if hl.get("meaning") or hl.get("why_it_matters"):
            st.markdown(
                "**No capability/safety pairing crosses the safety-lag threshold this week.** "
                + (hl.get("meaning") or ""))
            if hl.get("why_it_matters"):
                st.markdown(f"**Why this matters:** {hl['why_it_matters']}")
        else:
            sig = (snap.get("signals") or [None])[0]
            if sig:
                st.markdown(f"**{sig['headline']}.** {sig['detail']}")
            else:
                st.write("No capability/safety divergence crosses the alert threshold this "
                         "week — safety research is broadly keeping pace.")

    st.divider()
    st.subheader("🗞️ This week across the board")
    if get_analysis(snap).get("tabs"):
        st.caption("Claude's analytical read of each section, grounded in this week's data — "
                   "open a tab only if you want the underlying charts.")
    else:
        st.caption("Plain-language read of every section — open a tab only if you want the detail.")

    def board(label, fallback, key):
        st.markdown(f"**{label} —** {tab_analysis(snap, key) or fallback}")

    board("⚖️ Capability vs. safety", divergence_finding(snap), "divergence")
    board("📈 Velocity", velocity_finding(snap), "velocity")
    board("🔬 Sentiment / confidence", sentiment_finding(snap), "sentiment")
    board("🧭 Landscape", quadrant_finding(snap), "quadrant")
    board("🔥 Citations", citation_finding(snap), "citations")

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

        st.markdown("**Latest lab announcements:**")
        for post in lab[:5]:
            when = f" · {post['published']}" if post.get("published") else ""
            tp = f" · _{lbl(snap, post['topic'])}_" if post.get("topic") else ""
            title = post.get("title") or "(untitled)"
            link = f"[{title}]({post['url']})" if post.get("url") else title
            st.markdown(f"- **{post['source']}**: {link}{when}{tp}")
            dsum = short_desc(post, 200)
            if dsum:
                st.caption(dsum)

        with st.expander(f"All recent lab posts ({len(lab)})"):
            for post in lab[:20]:
                when = f" · {post['published']}" if post.get("published") else ""
                tp = f" · _{lbl(snap, post['topic'])}_" if post.get("topic") else ""
                title = post.get("title") or "(untitled)"
                if post.get("url"):
                    st.markdown(f"- **{post['source']}**: [{title}]({post['url']}){when}{tp}")
                else:
                    st.markdown(f"- **{post['source']}**: {title}{when}{tp}")
                d = short_desc(post, 200)
                if d:
                    st.caption(d)

    st.divider()
    with st.expander("🚨 All signals (full ranked list)"):
        sev = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for s in snap["signals"]:
            st.markdown(f"{sev.get(s['severity'], '⚪')} **{s['headline']}**  \n{s['detail']}")
    with st.expander("📄 Full markdown brief"):
        st.markdown(snap["brief"])
    st.download_button("⬇️ Download full markdown brief", data=snap["brief"],
                       file_name="foresight_brief.md", mime="text/markdown")

# ================================================================ Divergence
with tab_div:
    st.subheader("Capability vs. safety velocity gap")
    week_note(
        "Recent growth of each **capability** topic vs its paired **safety** topic "
        "(long blue + short orange = safety lagging).",
        divergence_finding(snap),
    )
    if tab_analysis(snap, "divergence"):
        st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'divergence')}")
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
        "Papers per quarter per topic, and whether each rate is accelerating or "
        "decelerating (table below). Attention, not success — cross-read with Sentiment.",
        velocity_finding(snap),
    )
    if tab_analysis(snap, "velocity"):
        st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'velocity')}")
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
        "Share of **critical / limitation-focused** papers per topic. A rising share "
        "(especially when volume is flat) is an early warning that confidence is eroding.",
        sentiment_finding(snap),
    )
    if tab_analysis(snap, "sentiment"):
        st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'sentiment')}")
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
        "Strategic map: **recent volume** (x) vs **growth** (y) — emerging, hot, "
        "cooling, white-space.",
        quadrant_finding(snap),
    )
    if tab_analysis(snap, "quadrant"):
        st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'quadrant')}")
    st.caption("Hover points for topic names.")
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


# ============================================================== Foresight Gap
def _arxiv_lookup(snap):
    """arxiv_id -> (title, url) from sources + citation movers, for linking risks."""
    out = {}
    for lst in (snap.get("sources") or {}).values():
        for p in lst:
            if p.get("arxiv_id"):
                out[p["arxiv_id"]] = (p.get("title", p["arxiv_id"]), p.get("url"))
    for bucket in ("rapid_growth", "sleepers"):
        for r in (snap.get("citations") or {}).get(bucket, []):
            if r.get("arxiv_id"):
                out[r["arxiv_id"]] = (r.get("title", r["arxiv_id"]), r.get("url"))
    return out


NOVELTY_BADGE = {
    "genuinely_unsurfaced": ("🟢", "Genuinely unsurfaced"),
    "partially_anticipated": ("🟡", "Partially anticipated"),
    "already_widely_discussed": ("🔴", "Already widely discussed"),
}


def novelty_label(v) -> str:
    if not v:
        return "⚪ Not verified"
    emoji, label = NOVELTY_BADGE.get(v.get("novelty_rating"), ("⚪", "Unrated"))
    act = v.get("recommended_action")
    return f"{emoji} {label}" + (f" · action: {act}" if act else "")


def foresight_brief_md(fg) -> str:
    """A self-contained markdown brief of the foresight-gap synthesis."""
    lines = [f"# signal-lag — Foresight Gap brief ({meta['refreshed_at']})", ""]
    lines.append("_AI-surfaced candidate risks for an analyst to pressure-test — not "
                 "predictions. Each crosses this week's research-trend signals with broader "
                 "societal forces, then is web-checked against current coverage._\n")
    for i, r in enumerate(fg.get("risks", []), 1):
        v = r.get("verification")
        lines.append(f"## {i}. {r.get('risk','')}")
        lines.append(f"- **Novelty (verified):** {novelty_label(v)}")
        if r.get("research_anchor") and r["research_anchor"].lower() != "none":
            lines.append(f"- **Research-trend anchor:** {r['research_anchor']}")
        if r.get("domains_crossed"):
            lines.append(f"- **Seam (domains crossed):** {', '.join(r['domains_crossed'])}")
        if r.get("communities"):
            lines.append(f"- **Which communities see which half:** {r['communities']}")
        if r.get("framing_inversion") and r["framing_inversion"].lower() not in ("n/a", "na", ""):
            lines.append(f"- **Framing inversion:** {r['framing_inversion']}")
        lines.append(f"- **Derived from:** {r.get('derived_from','')}")
        lines.append(f"- **Why under-discussed:** {r.get('why_underdiscussed','')}")
        lines.append(f"- **Mechanism:** {r.get('mechanism','')}")
        lines.append(f"- **Leading indicator:** {r.get('leading_indicator','')}")
        lines.append(f"- **Calibration:** {r.get('calibration','')}")
        lines.append(f"- **Extrapolation beyond the data:** {r.get('extrapolation','')}")
        if v:
            lines.append(f"- **Prior-coverage check:** {v.get('prior_coverage','')}")
            lines.append(f"- **Disputed claims:** {v.get('disputed_claims','')}")
            lines.append(f"- **Recalibrated:** {v.get('recalibrated_calibration','')}")
            for s in (v.get("sources") or [])[:6]:
                lines.append(f"    - [{s.get('title','')}]({s.get('url','')})")
        lines.append("")
    return "\n".join(lines)


with tab_foresight:
    st.subheader("🔮 Foresight Gap — novel risks in the seam")
    fg = get_analysis(snap).get("foresight_gap")
    # Humility framing — these are candidate hypotheses, not predictions.
    st.info(
        "**These are AI-surfaced *candidate* risks for an analyst to pressure-test — "
        "not predictions.** Each is anchored on *this week's* research-trend signal (a "
        "safety subfield decelerating or going critical — the tool's proprietary edge), "
        "crossed with broader societal forces to find the **seam between domains** that no "
        "single community is tracking — **then web-checked against current coverage** so a "
        "genuine seam is distinguished from something that just isn't in the index yet. "
        "Claude widens the aperture; **human judgment goes on top.**",
        icon="🔮",
    )

    if not fg:
        st.warning(
            "**Foresight synthesis unavailable for this snapshot.** Re-run the weekly "
            "refresh to populate it.",
            icon="🔌",
        )
    else:
        look = _arxiv_lookup(snap)
        risks = fg.get("risks", [])
        if fg.get("verified"):
            st.caption("✅ Each candidate was web-checked against current coverage and "
                       "ranked by verified novelty (genuinely unsurfaced → already "
                       "discussed). Already-discussed risks are kept but flagged, not hidden.")
        st.markdown(f"**{len(risks)} candidate risk(s)** this week.")

        demoted_header_shown = False
        for i, r in enumerate(risks, 1):
            v = r.get("verification")
            rating = (v or {}).get("novelty_rating")
            # Visual divider when we cross into the demoted (already-discussed) group.
            if rating == "already_widely_discussed" and not demoted_header_shown:
                st.divider()
                st.markdown("#### 🔴 Shown for transparency — already widely discussed")
                st.caption("The verifier found these (or close versions) are already public. "
                           "Kept visible so you can see what was filtered and why — not novel.")
                demoted_header_shown = True
            with st.container(border=True):
                st.markdown(f"**{novelty_label(v)}**")
                st.markdown(f"### {i}. {r.get('risk','')}")
                if r.get("research_anchor") and str(r["research_anchor"]).lower() != "none":
                    st.markdown(f"**📊 Research-trend anchor:** {r['research_anchor']}")
                if r.get("domains_crossed"):
                    st.markdown("**🔗 Seam (domains crossed):** "
                                + " × ".join(r["domains_crossed"]))
                if r.get("communities"):
                    st.markdown(f"**👥 Which communities see which half:** {r['communities']}")
                fi = r.get("framing_inversion")
                if fi and str(fi).lower() not in ("n/a", "na", ""):
                    st.markdown(f"**🔄 Framing inversion:** {fi}")
                st.markdown(f"**📐 Derived from:** {r.get('derived_from','')}")
                ids = [a for a in (r.get("source_arxiv_ids") or []) if a in look]
                if ids:
                    links = []
                    for a in ids:
                        title, url = look[a]
                        links.append(f"[{title}]({url})" if url else title)
                    st.markdown("**📄 Source papers:** " + " · ".join(links))
                if r.get("source_topics"):
                    st.markdown("**🏷️ Source topics:** " + ", ".join(r["source_topics"]))
                st.markdown(f"**🕳️ Why it's under-discussed:** {r.get('why_underdiscussed','')}")
                st.markdown(f"**⚙️ Mechanism:** {r.get('mechanism','')}")
                st.markdown(f"**📡 Leading indicator:** {r.get('leading_indicator','')}")
                st.markdown(f"**🎯 Calibration:** {r.get('calibration','')}")
                st.markdown(f"**⚠️ Extrapolation beyond the data:** {r.get('extrapolation','')}")
                # Prior-coverage check (the verification pass).
                if v:
                    with st.expander("🔎 Prior-coverage check (web-verified)", expanded=(rating != "genuinely_unsurfaced")):
                        st.markdown(f"**What already exists:** {v.get('prior_coverage','')}")
                        st.markdown(f"**Disputed / contested claims:** {v.get('disputed_claims','')}")
                        st.markdown(f"**Recalibrated:** {v.get('recalibrated_calibration','')}")
                        srcs = v.get("sources") or []
                        if srcs:
                            st.markdown("**Coverage found:**")
                            for s in srcs[:8]:
                                t, u = s.get("title", ""), s.get("url", "")
                                st.markdown(f"- [{t}]({u})" if u else f"- {t}")
                elif fg.get("verified"):
                    st.caption("⚪ Prior-coverage check could not be completed for this risk.")

        st.download_button(
            "⬇️ Download foresight-gap brief", data=foresight_brief_md(fg),
            file_name="foresight_gap_brief.md", mime="text/markdown",
        )

        # Transparency: show exactly what fed the synthesis.
        st.divider()
        digest = fg.get("digest") or {}
        with st.expander("🔬 The signal digest that fed this synthesis"):
            st.caption("The strongest signals signal-lag computed this week — the raw "
                       "material the synthesis reasons over. Nothing here is invented.")
            st.json(digest)
        with st.expander("🌐 Societal context & scanning framework used"):
            st.markdown("**Scanning framework** (the domains the synthesis must reason "
                        "across so it never tunnels on technology alone):")
            for dom, desc in (fg.get("framework") or {}).items():
                st.markdown(f"- **{dom}** — {desc}")
            st.divider()
            ctx = fg.get("context") or ""
            st.markdown("**Societal context** (the living, analyst-maintained "
                        "`config/context.md` — the current real-world state crossed "
                        "against the signals):")
            if ctx.strip():
                st.markdown(ctx)
            else:
                st.caption("No societal context was provided this week — the synthesis "
                           "reasoned across the scanning framework and general knowledge "
                           "only. Fill in `config/context.md` to sharpen it.")


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
        render_paper_card(p, topic_label)
    if not snap["sources"].get(pick):
        st.write("No tagged papers for this topic in the current snapshot.")

    st.divider()
    if tab_analysis(snap, "citations"):
        st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'citations')}")
    ccol, scol = st.columns(2)
    def _cite_line(r):
        extra = f" · {r['influential_citations']} influential" if r.get("influential_citations") else ""
        ven = f" · {r['venue']}" if r.get("venue") else ""
        return f"- [{r['title']}]({r['url']}) — {r['cited_by_count']} cites{extra}{ven}"

    def _render_cites(bucket):
        for r in snap["citations"].get(bucket, [])[:10]:
            st.markdown(_cite_line(r))
            d = short_desc(r, 160)
            if d:
                st.caption(d)

    with ccol:
        st.markdown("**🔥 Rapid citation growth**")
        _render_cites("rapid_growth")
    with scol:
        st.markdown("**💤 Sleepers (early-heat)**")
        _render_cites("sleepers")

# =============================================================== Methodology
def render_glossary(snap):
    """Plain-language definition + read-more link for every tracked category."""
    def block(title, keys, blurb):
        st.markdown(f"#### {title}")
        st.caption(blurb)
        for k in keys:
            entry = GLOSSARY.get(k)
            if not entry:
                continue
            definition, link = entry
            st.markdown(f"- **{lbl(snap, k)}** — {definition} [↗ read more]({link})")

    block("🛡️ Safety topics", SAFETY_KEYS,
          "What we'd need to *trust* increasingly capable systems.")
    block("⚡ Capability topics", CAPABILITY_KEYS,
          "What makes systems more powerful — the side that tends to move first.")
    st.caption("Each safety topic is paired with the capability it's meant to keep pace "
               "with; the **Divergence** tab measures the gap between the two.")


with tab_method:
    st.subheader("How signal-lag works")

    with st.expander("📖 The categories we track — what each term means", expanded=True):
        st.markdown("Plain-language definitions of every capability and safety concept "
                    "this tool tracks, so the rest of the dashboard is legible even if "
                    "you don't live in AI-safety jargon.")
        render_glossary(snap)

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
- **Semantic Scholar** — TLDRs, *influential*-citation counts, venue, fields.
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
markdown brief.

### 9. Weekly Claude analysis
Once per refresh, the computed metrics **plus the real paper abstracts** are sent
to **Claude** (`claude-opus-4-8`), which returns a genuine analytical read: what the
widest capability↔safety gap actually *means* and why it matters, a per-tab
interpretation of the week, and a one-line *what-it-does* / *why-it-matters* for
each driving paper. This is computed once and baked into the snapshot (no model calls
at page-load). Claude only interprets the real metrics and abstracts; it does not
invent data.

### 10. Foresight Gap synthesis + novelty verification (the 🔮 tab)
A **second Claude pass** (`claude-opus-4-8`, same API and fail-soft architecture as
section 9) crosses this week's strongest signals with broader societal forces to surface
**novel, not-yet-in-the-news risks** — the kind that live in the *seam between domains*.

It assembles a **signal digest** (flagged divergences, velocity inflections, rising
critical-share / eroding-confidence flags, quadrant emerging/white-space, citation
movers, new clusters, lab activity, **and what changed week-over-week**), combines it
with a fixed **STEEP/PESTLE-plus scanning framework** (Social, Technological, Economic,
Environmental, Political, Legal/Regulatory, Security/Geopolitical, Demographic) and a
**living, analyst-maintained context file** (`config/context.md`, the current real-world
state — *you* keep it updated; examples in it are illustrative, never an exhaustive list).

The synthesis is tuned to the tool's actual edge: it **anchors each risk on the tool's
proprietary research-trend signal** (a safety subfield decelerating or going critical —
something no outside commentator has), then **crosses it with the societal context** to
find a **cross-silo seam** where two expert communities each see only half the problem
(named per risk). It **rewards framing inversions** of trends everyone treats as simply
good or bad, and **lowers confidence when a risk leans on a contested or inferential
claim** — including over-reading the tool's own trend metric as causation.

Then a **novelty-verification pass** web-searches each candidate (Claude's server-side
web search) for **both confirming and disputing** coverage, and returns a *prior-coverage
check* with a verified novelty rating — *genuinely unsurfaced / partially anticipated /
already widely discussed* — plus disputing sources and a recalibrated confidence.
Already-discussed risks are **demoted and flagged, not hidden**. The searches are baked
into the snapshot (run once per refresh, cached — never at page load). This is the
calibrated posture: the tool generates candidate risks **and then checks them against
current coverage before surfacing them**, distinguishing a genuine seam from something
that simply isn't in its index yet.

**These are candidate hypotheses to pressure-test, not predictions** — the model widens
the aperture; human judgment goes on top.

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

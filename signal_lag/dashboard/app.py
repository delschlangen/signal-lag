"""Streamlit dashboard for signal-lag.

Views:
  - Topic-velocity time series
  - Capability-vs-safety divergence (the headline)
  - Emerging / cooling / white-space quadrant
  - Signals panel with markdown-brief download

Run: streamlit run signal_lag/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run` to import the package without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signal_lag.analysis.runner import run_analysis  # noqa: E402
from signal_lag.config import load_all  # noqa: E402

st.set_page_config(page_title="signal-lag", layout="wide")


@st.cache_data(show_spinner="Running analysis...")
def _analyze():
    settings, taxonomy = load_all()
    results = run_analysis(settings, taxonomy)
    label_map = {t.key: t.label for t in taxonomy.all_topics}
    return results, label_map


def _label(label_map, key):
    return label_map.get(key, key)


st.title("📡 signal-lag — AI safety research foresight")
st.caption(
    "Patent-landscape-style foresight: topic velocity, citation dynamics, and the "
    "gap where safety attention lags capability."
)

try:
    results, label_map = _analyze()
except Exception as e:
    st.error(f"Could not run analysis: {e}\n\nRun `python -m signal_lag.cli ingest "
             "--use-fixtures` first to populate the cache.")
    st.stop()

meta = results["meta"]
c1, c2, c3 = st.columns(3)
c1.metric("Papers analyzed", meta["n_papers"])
c2.metric("Embedding backend", meta["backend"])
c3.metric(
    "Divergences flagged",
    f"{sum(1 for d in results['divergence'] if d['lagging'])}/{len(results['divergence'])}",
)

tab_div, tab_vel, tab_quad, tab_sig = st.tabs(
    ["⚖️ Divergence", "📈 Velocity", "🧭 Quadrant", "🚨 Signals"]
)

# --------------------------------------------------------------- Divergence
with tab_div:
    st.subheader("Capability vs. safety velocity gap")
    div = pd.DataFrame(results["divergence"])
    if not div.empty:
        fig = go.Figure()
        fig.add_bar(name="Capability growth", x=div["pairing"], y=div["cap_growth"])
        fig.add_bar(name="Safety growth", x=div["pairing"], y=div["saf_growth"])
        fig.update_layout(barmode="group", yaxis_title="Recent growth rate",
                          xaxis_tickangle=-25, height=460, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        show = div[[
            "pairing", "cap_growth", "saf_growth", "gap", "volume_ratio", "lagging"
        ]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No pairings configured.")

# ----------------------------------------------------------------- Velocity
with tab_vel:
    st.subheader("Topic submission velocity (per quarter)")
    ts = results["taxonomy_timeseries"]
    if not ts.empty:
        plot_df = ts.copy()
        plot_df["topic"] = plot_df["topic_key"].map(lambda k: _label(label_map, k))
        plot_df["period"] = plot_df["period"].astype(str)
        topics = sorted(plot_df["topic"].unique())
        chosen = st.multiselect("Topics", topics, default=topics[: min(6, len(topics))])
        sel = plot_df[plot_df["topic"].isin(chosen)]
        fig = px.line(sel, x="period", y="count", color="topic", markers=True, height=480)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeseries yet.")

    st.markdown("**Inflections**")
    st.dataframe(pd.DataFrame(results["inflections"]), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- Quadrant
with tab_quad:
    st.subheader("Emerging / hot / cooling / white-space")
    quad = pd.DataFrame(results["quadrant"])
    if not quad.empty:
        quad["topic"] = quad["topic_key"].map(lambda k: _label(label_map, k))
        fig = px.scatter(
            quad, x="recent_mean", y="change", color="quadrant", text="topic",
            labels={"recent_mean": "Recent volume (papers/quarter)", "change": "Growth rate"},
            height=560,
        )
        fig.update_traces(textposition="top center")
        fig.add_hline(y=0.3, line_dash="dot", line_color="gray")
        fig.add_vline(x=5, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(quad[["topic", "recent_mean", "change", "quadrant"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No quadrant data.")

    if results["new_clusters"]:
        st.markdown("**Newly forming clusters (emergent, unsupervised):**")
        for c in results["new_clusters"]:
            st.write(f"- {c}")

# ------------------------------------------------------------------ Signals
with tab_sig:
    st.subheader("Signals (BLUF)")
    sev_color = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    for s in results["signals"]:
        st.markdown(f"{sev_color.get(s['severity'], '⚪')} **{s['headline']}**  \n{s['detail']}")
    st.divider()
    st.download_button(
        "⬇️ Download markdown brief",
        data=results["brief"],
        file_name="foresight_brief.md",
        mime="text/markdown",
    )
    with st.expander("Preview brief"):
        st.markdown(results["brief"])

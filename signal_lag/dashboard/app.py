"""Streamlit dashboard for signal-lag.

Reads a precomputed snapshot (``data/snapshot.json``) produced weekly from real
arXiv + OpenAlex (+ OpenReview, lab blogs) data by the refresh GitHub Action.

It only ever renders **real** data: if no live snapshot is present, it shows an
honest "data not available yet" message rather than any synthetic/demo content.

Run: streamlit run signal_lag/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from signal_lag.analysis import alerts  # noqa: E402
from signal_lag.glossary import CAPABILITY_KEYS, GLOSSARY, SAFETY_KEYS  # noqa: E402
from signal_lag.snapshot import (  # noqa: E402
    arxiv_url, diff_snapshots, load_snapshot, register_is_stale, register_newest_date,
    sort_register)

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


def surfaced_risks(fg, limit=3):
    """The best (web-verified novel/partial) risks from a foresight_gap block."""
    if not fg or not fg.get("risks"):
        return []
    surfaced = [
        r for r in fg["risks"]
        if (r.get("verification") or {}).get("novelty_rating")
        in ("genuinely_unsurfaced", "partially_anticipated")
    ]
    chosen = surfaced or fg["risks"]  # unverified snapshots: show top risks
    return chosen[:limit]


def surfaced_foresight(snap, limit=3):
    return surfaced_risks(get_analysis(snap).get("foresight_gap"), limit)


_PLAIN_SECTIONS = (("The technical evidence", "technical_evidence"),
                   ("The real-world context", "societal_evidence"),
                   ("The gap (synthesis)", "the_gap"),
                   ("The tool's own skepticism", "skepticism"),
                   ("Bottom line", "bottom_line"))


def explained_risks(snap):
    """Overall foresight risks that carry a plain-language walkthrough."""
    fg = get_analysis(snap).get("foresight_gap") or {}
    return [r for r in (fg.get("risks") or []) if r.get("plain_explanation")]


def provenance_line(meta: dict) -> str:
    """Compact one-line provenance/trust metadata from the snapshot meta (#21)."""
    n_papers = meta.get("n_papers") or 0
    n_tagged = meta.get("n_tagged")
    s2 = meta.get("s2_enriched") or 0
    pct = round(100 * s2 / n_papers) if n_papers else 0
    parts = []
    if n_tagged:
        parts.append(f"🏷️ {n_tagged:,} of {n_papers:,} papers tagged")
    elif n_papers:
        parts.append(f"📦 {n_papers:,} papers")
    if n_papers:
        parts.append(f"🔬 {pct}% Semantic-Scholar enriched")
    if meta.get("topics_tracked"):
        parts.append(f"🧭 {meta['topics_tracked']} topics · {meta.get('n_pairings', 0)} pairings")
    cats = meta.get("categories") or []
    if cats:
        parts.append("📚 " + ", ".join(cats))
    if meta.get("refreshed_at"):
        parts.append(f"🗓️ refreshed {meta['refreshed_at']}")
    return " · ".join(parts)


def plain_language_brief_md(snap) -> str:
    """Downloadable markdown of the plain-language 'how the tool reasoned' walkthroughs."""
    date = snap.get("meta", {}).get("refreshed_at", "")
    lines = [f"# signal-lag — Risks in plain terms ({date})", "",
             "Plain-language walkthroughs of the top foresight risks: the technical evidence, "
             "the real-world context, the gap, the tool's own skepticism, and a bottom line.", ""]
    risks = explained_risks(snap)
    if not risks:
        lines.append("_No plain-language explanations in this snapshot._")
    for i, r in enumerate(risks, 1):
        pe = r["plain_explanation"]
        lines.append(f"## {i}. {r.get('risk','')}")
        for label, key in _PLAIN_SECTIONS:
            if pe.get(key):
                lines.append(f"- **{label}:** {pe[key]}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------- structured data exports (#41)
def register_csv(register) -> str:
    """Flatten the evergreen risk register to CSV for analyst reuse."""
    rows = []
    for e in register or []:
        lt = e.get("latest") or {}
        rows.append({
            "id": e.get("id"), "risk": e.get("risk"),
            "first_seen": e.get("first_seen"), "last_seen": e.get("last_seen"),
            "n_appearances": e.get("n_appearances"),
            "priority": lt.get("priority"), "severity": lt.get("severity"),
            "likelihood": lt.get("likelihood"), "exposure": lt.get("exposure"),
            "trajectory": lt.get("trajectory"),
            "leading_indicator": lt.get("leading_indicator"),
        })
    return pd.DataFrame(rows).to_csv(index=False)


def incidents_csv(snap) -> str:
    """Incident benchmark records to CSV (with the credibility fields)."""
    recs = (snap.get("incidents") or {}).get("records") or []
    cols = ["date", "title", "harm_key", "confidence", "severity", "affected_sector",
            "ai_involvement_confidence", "attribution_confidence", "source_quality",
            "deployer", "summary", "source_url"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in recs]).to_csv(index=False)


def velocity_csv(snap) -> str:
    """Per-topic velocity inflections to CSV."""
    rows = [{"topic": lbl(snap, r.get("topic_key")), **{k: r.get(k) for k in
            ("recent_mean", "prior_mean", "change", "direction")}}
            for r in (snap.get("inflections") or [])]
    return pd.DataFrame(rows).to_csv(index=False)


def lab_lag_csv(snap) -> str:
    """Per-announcement lab→safety-response lag to CSV."""
    posts = (snap.get("lab_lag") or {}).get("posts") or []
    cols = ["published", "lab", "announcement", "capability", "safety",
            "days_to_first", "weeks_to_measurable", "status", "baseline_per_week"]
    return pd.DataFrame([{c: p.get(c) for c in cols} for p in posts]).to_csv(index=False)


def citation_matrix_csv(snap) -> str:
    """Capability×safety citation matrix (long form) to CSV."""
    rows = []
    cg = snap.get("citation_graph") or {}
    for direction, key in (("cap→saf", "matrix_cap_to_saf"), ("saf→cap", "matrix_saf_to_cap")):
        for src, tgt_counts in (cg.get(key) or {}).items():
            for tgt, n in tgt_counts.items():
                rows.append({"direction": direction, "from": src, "to": tgt, "citations": n})
    return pd.DataFrame(rows).to_csv(index=False)


def compact_brief_md(snap) -> str:
    """One-page weekly brief (#38): top-3s + recommended actions, print-friendly markdown.

    Only called at render time (from the sidebar), so it can safely use globals defined
    later in the script (_DELTAS, risk_register).
    """
    date = snap.get("meta", {}).get("refreshed_at", "")
    lines = [f"# signal-lag — one-page brief ({date})", ""]

    d = _DELTAS or {}
    changes = []
    for p in (d.get("divergence") or {}).get("new_lagging", []):
        changes.append(f"🚨 New safety-lag pairing: {p}")
    for k in (d.get("velocity") or {}).get("new_accelerating", []):
        changes.append(f"📈 Newly accelerating: {lbl(snap, k)}")
    for r in (d.get("incidents") or {}).get("new", []):
        changes.append(f"🌐 New incident: {r['title']} ({r['date']})")
    for r in (d.get("foresight") or {}).get("new_risks", []):
        changes.append(f"🆕 New risk: {r[:100]}")
    lines.append("## Top changes this refresh")
    lines += [f"- {c}" for c in changes[:3]] or ["- (no week-over-week movement recorded)"]

    reg = sort_register(risk_register)[:3]
    lines += ["", "## Top risks (register, forced ranking)"]
    for i, e in enumerate(reg, 1):
        lt = e.get("latest") or {}
        lines.append(f"{i}. **[P{lt.get('priority')}]** {e.get('risk')} "
                     f"(*{estimative(lt.get('likelihood'))}*, {lt.get('trajectory', '')})")

    lines += ["", "## Indicators to watch"]
    inds = [(e.get("latest") or {}).get("leading_indicator") for e in reg]
    lines += [f"- {i}" for i in inds if i][:3] or ["- (none recorded)"]

    lines += ["", "## Notable papers"]
    notable = (weekly_block(snap).get("summary") or {}).get("notable") or []
    look = _arxiv_lookup(snap)
    n_added = 0
    for r in notable:
        aid = r.get("arxiv_id")
        if aid in look and n_added < 3:
            title, url = look[aid]
            lines.append(f"- [{title}]({url}) — {r.get('why_it_matters', '')}")
            n_added += 1
    if not n_added:
        lines.append("- (no this-week notables in this snapshot)")

    lines += ["", "## Recommended actions"]
    fg = get_analysis(snap).get("foresight_gap") or {}
    acts = []
    for r in sorted(fg.get("risks") or [], key=lambda r: r.get("priority") or 0, reverse=True):
        am = r.get("action_map") or {}
        for k in ("eval_to_run", "benchmark_to_monitor", "mitigation"):
            if am.get(k) and len(acts) < 4:
                acts.append(am[k])
    lines += [f"- {a}" for a in acts] or ["- (populates when risks carry action maps)"]

    lines += ["", "---", "_Full detail: the signal-lag dashboard (Divergence · Velocity · "
              "Sentiment · Foresight · Sources). Research-attention signal, not deployment "
              "telemetry — candidate hypotheses to pressure-test._"]
    return "\n".join(lines)


def weekly_block(snap):
    return snap.get("weekly") or {}


def view_toggle(key: str, available: bool) -> str:
    """Quarterly/This-week toggle (defaults to **This week** when available).

    Returns "weekly" or "overall". Renders nothing (returns "overall") when there's
    no weekly data to switch to.
    """
    if not available:
        return "overall"
    choice = st.radio(
        "View", ["🆕 This week", "📊 Quarterly (overall)"],
        horizontal=True, label_visibility="collapsed", key=key,
    )
    return "weekly" if choice.startswith("🆕") else "overall"


_key = SNAPSHOT.stat().st_mtime if SNAPSHOT.exists() else 0.0
snap, prev = _load(_key)

st.title("📡 signal-lag — AI emerging-risk foresight")
st.caption(
    "Strategic foresight on the AI frontier: from the research-trend signal "
    "(topic velocity, sentiment, capability-vs-safety divergence) to harm/misuse vectors, "
    "a scored risk register, 6–24-month scenarios, and a real-world incident benchmark — on "
    "real arXiv data, refreshed weekly."
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
st.caption("🔎 " + provenance_line(meta) + "  ·  _all figures trace to this snapshot; "
           "no page-load API calls_")

# Per-tab week-over-week deltas (#39), computed once from the previous snapshot.
_DELTAS = alerts.tab_deltas(snap, prev)


def delta_panel(tab_key: str):
    """Compact 'what changed since last refresh' panel at the top of a tab (#39)."""
    d = _DELTAS.get(tab_key) or {}
    if not d or not any(v for v in d.values()):
        return
    L = lambda k: lbl(snap, k)  # noqa: E731
    lines = []
    if tab_key == "divergence":
        if d.get("new_lagging"):
            lines.append("🚨 **New safety-lag pairings:** " + ", ".join(d["new_lagging"]))
        if d.get("resolved"):
            lines.append("✅ **No longer lagging:** " + ", ".join(d["resolved"]))
    elif tab_key == "velocity":
        if d.get("new_accelerating"):
            lines.append("📈 **Newly accelerating:** " + ", ".join(L(k) for k in d["new_accelerating"]))
        if d.get("new_decelerating"):
            lines.append("📉 **Newly decelerating:** " + ", ".join(L(k) for k in d["new_decelerating"]))
    elif tab_key == "sentiment":
        if d.get("new_rising"):
            lines.append("⚠️ **New eroding-confidence warnings:** " + ", ".join(L(k) for k in d["new_rising"]))
        if d.get("cleared"):
            lines.append("✅ **Warnings cleared:** " + ", ".join(L(k) for k in d["cleared"]))
        if d.get("biggest_shifts"):
            lines.append("↕️ **Biggest critical-share shifts:** " + ", ".join(
                f"{L(s['topic_key'])} ({s['shift_pts']:+.0f} pts)" for s in d["biggest_shifts"]))
    elif tab_key == "foresight":
        if d.get("new_risks"):
            lines.append(f"🆕 **New risks this refresh ({len(d['new_risks'])}):** "
                         + " · ".join(r[:80] for r in d["new_risks"][:3]))
        if d.get("dropped_risks"):
            lines.append(f"🗑️ **Rotated out:** {len(d['dropped_risks'])} risk(s) from last refresh")
    elif tab_key == "incidents":
        if d.get("new"):
            lines.append("🆕 **New incidents since last refresh:** " + " · ".join(
                f"{r['title']} ({r['date']})" for r in d["new"][:4]))
    if lines:
        with st.container(border=True):
            st.caption(f"🔄 **Since last refresh** ({_DELTAS.get('prev_date', '')}):")
            for ln in lines:
                st.markdown(ln)

with st.expander("ℹ️ New here? How to read this dashboard", expanded=False):
    # Analyst's note — the core framing for reading everything below.
    st.info(
        "**Analyst note:** signal-lag measures *research attention*, not *research success*. "
        "A spike can mean a breakthrough **or** a field thrashing against a wall — so treat "
        "this as a **triage instrument** that shows *where to investigate*, not *what to "
        "conclude*. The Sentiment tab helps tell those two cases apart.",
        icon="🧭",
    )
    st.markdown(
        "**Start on 📋 Weekly Summary** — it digests every other tab in plain language, "
        "so you only open the others for detail.\n\n"
        "**The tabs**\n"
        "- **📋 Weekly Summary** — the briefing: what changed, the headline gap, a read of "
        "every tab, and this week's best foresight risks.\n"
        "- **⚖️ Divergence** — where capability research is outpacing the paired safety work.\n"
        "- **📈 Velocity** — how fast each topic is moving (accelerating / cooling), plus "
        "the **strategic map** (volume × growth: emerging / hot / cooling / white-space).\n"
        "- **🔬 Sentiment** — share of *critical / limitation-focused* papers; a rising share "
        "is an early confidence-erosion warning.\n"
        "- **🔮 Foresight** — five views: novel **cross-domain risks** (web-checked for "
        "novelty), **⚠️ Harm vectors** (dual-use misuse lens, 0–24 mo), a scored **📋 Risk "
        "register** (severity × likelihood × exposure × trajectory), **🎬 Scenarios** "
        "(how the top risks could evolve, 6–24 mo), and **🌐 Incidents** (real-world incidents "
        "crossed against the research signal: leading vs lagging) — plus downloadable "
        "intelligence-estimate and tabletop-exercise packs.\n"
        "- **🔍 Sources** — the actual papers behind every topic, all linked.\n"
        "- **📖 Methodology** — how it all works + a glossary of every term.\n\n"
        "**Symbols you'll see**\n"
        "- 🟢 genuinely unsurfaced · 🟡 partially anticipated · 🔴 already widely discussed "
        "(the verified novelty of a foresight risk)\n"
        "- 🧠 *Claude's read* — an AI-written analytical interpretation of that tab's data\n"
        "- ⚠️ eroding confidence · 🚨 new safety-lag alert this week"
        + ("\n\n_This week's figures and risks were analyzed by **Claude** (computed once "
           "into the snapshot, not run live)._" if get_analysis(snap) else "")
    )

(tab_summary, tab_div, tab_vel, tab_sentiment,
 tab_foresight, tab_sources, tab_method, tab_history) = st.tabs(
    ["📋 Weekly Summary", "⚖️ Divergence", "📈 Velocity", "🔬 Sentiment",
     "🔮 Foresight", "🔍 Sources", "📖 Methodology", "📜 History"]
)


HISTORY_PATH = SNAPSHOT.with_name("history.json")


@st.cache_data(ttl=1800)
def _load_history(_cache_key: float):
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


history = _load_history(HISTORY_PATH.stat().st_mtime if HISTORY_PATH.exists() else 0.0)

REGISTER_PATH = SNAPSHOT.with_name("risk_register.json")


@st.cache_data(ttl=1800)
def _load_register(_cache_key: float):
    try:
        return json.loads(REGISTER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


risk_register = _load_register(REGISTER_PATH.stat().st_mtime if REGISTER_PATH.exists() else 0.0)

BENCH_HISTORY_PATH = SNAPSHOT.with_name("benchmark_history.json")


@st.cache_data(ttl=1800)
def _load_bench_history(_cache_key: float):
    try:
        return json.loads(BENCH_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


bench_history = _load_bench_history(
    BENCH_HISTORY_PATH.stat().st_mtime if BENCH_HISTORY_PATH.exists() else 0.0)


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
    """One line on what the chart is, then a data-driven read of the current quarter."""
    st.caption(what)
    if finding:
        st.markdown(f"**📊 This quarter:** {finding}")


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
        parts.append("No topic shows a rising critical share this quarter.")
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
    return " ".join(parts) if parts else "No standout citation movers this quarter."


# ============================================================ Weekly Summary
def render_overall_summary():
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

    # Inverted-pyramid briefing order: the single biggest gap, then the novel
    # foresight risks, then this week's deltas, then the per-tab digest.

    # ---- 1. Headline: the single most important thing ----
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
            f"**The biggest safety gap this quarter is {cap} vs {saf}.**{meaning} Capability "
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
                "**No capability/safety pairing crosses the safety-lag threshold this quarter.** "
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

    # ---- 2. Best foresight gaps: the novel forward-looking content ----
    top_fore = surfaced_foresight(snap, limit=3)
    if top_fore:
        st.divider()
        st.subheader("🔮 Foresight Gap — top novel risks")
        st.caption("The strongest cross-domain risks from the latest analysis — anchored on "
                   "our research-trend signals and web-checked for novelty. These are "
                   "candidate hypotheses to pressure-test; full six-part analysis and "
                   "prior-coverage checks are in the **🔮 Foresight Gap** tab.")
        for r in top_fore:
            with st.container(border=True):
                st.markdown(f"**{novelty_label(r.get('verification'))}** — {r.get('risk','')}")
                meta_bits = []
                if r.get("domains_crossed"):
                    meta_bits.append("🔗 " + " × ".join(r["domains_crossed"]))
                if r.get("research_anchor") and str(r["research_anchor"]).lower() != "none":
                    meta_bits.append("📊 " + r["research_anchor"])
                if meta_bits:
                    st.caption("  ·  ".join(meta_bits))
                if r.get("communities"):
                    st.markdown(f"_Who sees which half:_ {r['communities']}")

    # ---- 3. What changed since last refresh (week-over-week deltas) ----
    st.divider()
    st.subheader("🆕 What changed since last refresh")
    d = diff_snapshots(snap, prev)
    if d["first_run"]:
        st.info("First snapshot — no prior refresh to compare against yet. "
                "Week-over-week changes will appear here from the next refresh.")
    else:
        st.caption(f"Compared against the snapshot from {d['prev_date']}.")
        if d["new_alerts"]:
            st.markdown("**🚨 New safety-lag alerts:**")
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

    # ---- 4. Per-tab digest: a read of every other tab ----
    st.divider()
    st.subheader("🗞️ This quarter — across the board")
    if get_analysis(snap).get("tabs"):
        st.caption("Claude's analytical read of each section, grounded in this quarter's data — "
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


def render_weekly_summary():
    w = weekly_block(snap)
    st.caption(
        f"Just the **{w.get('n_papers', 0)} papers** submitted in the last "
        f"{w.get('window_days', 7)} days (since {w.get('cutoff', '')}). The quarterly "
        "charts/trends are unaffected — switch back to **Quarterly** for the long view."
    )
    summ = w.get("summary") or {}
    if summ.get("summary"):
        st.subheader("📝 This week in AI-safety research")
        st.markdown(summ["summary"])
        if summ.get("themes"):
            st.markdown("**Themes:** " + " · ".join(summ["themes"]))
    else:
        st.info("This-week Claude summary isn't available in this snapshot.", icon="🔌")

    counts = w.get("counts_by_topic") or {}
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**🛡️ Safety topics this week**")
        sc = counts.get("safety") or {}
        if sc:
            for k, v in list(sc.items())[:8]:
                st.markdown(f"- {k}: **{v}**")
        else:
            st.caption("No safety-tagged papers this week.")
    with cc2:
        st.markdown("**⚡ Capability topics this week**")
        cap = counts.get("capability") or {}
        if cap:
            for k, v in list(cap.items())[:8]:
                st.markdown(f"- {k}: **{v}**")
        else:
            st.caption("No capability-tagged papers this week.")

    top = surfaced_risks(w.get("foresight_gap"), limit=3)
    if top:
        st.divider()
        st.subheader("🔮 This week's novel risks")
        st.caption("From the this-week foresight pass — full detail in the **🔮 Foresight "
                   "Gap** tab (switch it to *This week*).")
        for r in top:
            with st.container(border=True):
                st.markdown(f"**{novelty_label(r.get('verification'))}** — {r.get('risk', '')}")
                if r.get("domains_crossed"):
                    st.caption("🔗 " + " × ".join(r["domains_crossed"]))

    notable = w.get("notable_papers") or []
    if notable:
        st.divider()
        st.subheader("📄 Notable papers this week")
        notes = {n.get("arxiv_id"): n for n in (summ.get("notable") or [])}
        for p in notable[:8]:
            with st.container(border=True):
                bits = [p.get("published", "")]
                if p.get("venue"):
                    bits.append(p["venue"])
                if p.get("cited_by_count"):
                    bits.append(f"{p['cited_by_count']} cites")
                st.markdown(f"**[{p['title']}]({p['url']})**  \n" + " · ".join(b for b in bits if b))
                desc = short_desc(p, 240)
                if desc:
                    st.markdown(f"*{desc}*")
                why = (notes.get(p.get("arxiv_id")) or {}).get("why_it_matters")
                if why:
                    st.markdown(f"**Why it matters this week:** {why}")


with tab_summary:
    if view_toggle("summary_view", bool(weekly_block(snap))) == "weekly":
        render_weekly_summary()
    else:
        render_overall_summary()

    # 🧩 Plain-language risk briefs — the "how the tool reasoned" walkthroughs from the
    # (quarterly) foresight pass, surfaced here as a button + expanders under either toggle.
    _explained = explained_risks(snap)
    if _explained:
        st.divider()
        st.subheader("🧩 Risks in plain terms")
        st.caption("Plain-language walkthroughs of the top risks — the technical evidence, "
                   "the real-world context, the gap, the tool's own skepticism, and a bottom "
                   "line (what's observed vs projected). Same content as the 🔮 Foresight tab "
                   "and the downloadable intelligence estimate.")
        st.download_button(
            "🧩 Download risks in plain terms (markdown)",
            data=plain_language_brief_md(snap),
            file_name="signal_lag_plain_language_risks.md", mime="text/markdown",
            width="stretch",
        )
        for i, r in enumerate(_explained, 1):
            pe = r["plain_explanation"]
            with st.expander(f"🧩 {i}. {(r.get('risk') or '')[:90]}"):
                for label, key in _PLAIN_SECTIONS:
                    if pe.get(key):
                        st.markdown(f"**{label}:** {pe[key]}")


# ---- This-week chart views (raw counts from the last window_days, not quarterly) ----
def render_weekly_velocity():
    w = weekly_block(snap)
    cbk = w.get("counts_by_key") or {}
    st.caption(f"Papers submitted in the last {w.get('window_days', 7)} days, per topic — "
               "a raw count of what landed this week, not the quarterly velocity trend.")
    if not cbk:
        st.info("No this-week paper counts in this snapshot.")
        return
    df = pd.DataFrame([{"topic": lbl(snap, k), "papers": v} for k, v in cbk.items()])
    df = df.sort_values("papers")
    fig = px.bar(df, x="papers", y="topic", orientation="h", template="plotly_dark",
                 height=120 + 24 * len(df))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="papers this week", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")


def render_weekly_divergence():
    w = weekly_block(snap)
    cbk = w.get("counts_by_key") or {}
    st.caption(f"This week's attention split per pairing — papers from the last "
               f"{w.get('window_days', 7)} days on the capability vs the safety side "
               "(a raw count, not the quarterly growth gap).")
    rows = [{"pairing": d["pairing"].replace(" vs. ", "<br>vs. "),
             "cap": cbk.get(d["capability_topic"], 0), "saf": cbk.get(d["safety_topic"], 0)}
            for d in snap.get("divergence", [])]
    if not rows:
        st.info("No pairings configured.")
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(name="Capability papers", y=df["pairing"], x=df["cap"],
                orientation="h", marker_color="#4c8bf5")
    fig.add_bar(name="Safety papers", y=df["pairing"], x=df["saf"],
                orientation="h", marker_color="#ffa94d")
    fig.update_layout(barmode="group", template="plotly_dark", height=140 + 95 * len(df),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=10, r=20, t=10, b=30), xaxis_title="papers this week")
    st.plotly_chart(fig, width="stretch")


def render_weekly_sentiment():
    w = weekly_block(snap)
    ws = w.get("sentiment") or {}
    st.caption(f"Critical / limitation-focused share among this week's papers (last "
               f"{w.get('window_days', 7)} days; topics with ≥3 papers) — a this-week "
               "snapshot, not the quarterly trend.")
    if not ws:
        st.info("Not enough this-week papers per topic for a critical-share read.")
        return
    df = pd.DataFrame([{"topic": lbl(snap, k), "critical_%": round(v["critical_share"] * 100, 1),
                        "n papers": v["n"]} for k, v in ws.items()])
    df = df.sort_values("critical_%")
    fig = px.bar(df, x="critical_%", y="topic", orientation="h", color="critical_%",
                 color_continuous_scale="Reds", template="plotly_dark",
                 height=120 + 24 * len(df))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                      xaxis_title="% critical (this week)", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.sort_values("critical_%", ascending=False), width="stretch", hide_index=True)


# ================================================================ Divergence
def render_divergence_overall():
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
        st.plotly_chart(fig, width="stretch")
        def _vol_balance(x):
            if pd.isna(x):
                return "—"
            if x >= 1.15:
                return f"cap-heavy {x:.2f}×"
            if x <= 0.87:
                return f"safety-heavy {1/x:.2f}×"
            return f"even {x:.2f}×"

        def _growth_balance(cap, saf):
            g = (cap - saf) * 100
            side = "capability" if g > 0 else ("safety" if g < 0 else "—")
            return f"{side} +{abs(g):.0f} pts" if side != "—" else "even"

        def _g_with_ci(growth, recent):
            """Growth% with a ±1σ Poisson band (#22); '—' band when a side is too thin."""
            prior = recent / (1 + growth) if growth > -0.99 else 0
            unc = alerts.growth_uncertainty(recent, prior)
            band = f" ±{unc*100:.0f}" if unc is not None else ""
            return f"{growth*100:+.0f}%{band}"

        adj = {a["pairing"]: a for a in alerts.confidence_adjusted_divergence(snap)}
        disp = pd.DataFrame({
            "Pairing": div["pairing"],
            "Cap papers/qtr": div["cap_recent"].map(lambda x: f"{x:.0f}"),
            "Saf papers/qtr": div["saf_recent"].map(lambda x: f"{x:.0f}"),
            "Volume balance": div["volume_ratio"].map(_vol_balance),
            "Capability growth/qtr": [
                _g_with_ci(g, r) for g, r in zip(div["cap_growth"], div["cap_recent"])],
            "Safety growth/qtr": [
                _g_with_ci(g, r) for g, r in zip(div["saf_growth"], div["saf_recent"])],
            "Growth balance": [
                _growth_balance(c, s) for c, s in zip(div["cap_growth"], div["saf_growth"])],
            "Adj. gap": div["pairing"].map(
                lambda p: f"{adj[p]['adjusted_gap']*100:+.0f} pts" if p in adj else "—"),
            "Lag status": div["lagging"].map(
                lambda b: "⚠️ safety lagging" if b else "keeping pace"),
        })
        st.dataframe(disp, width="stretch", hide_index=True)
        # Confidence-adjusted divergence (#12): why the adjusted gap differs from raw.
        notable_adj = [a for a in adj.values()
                       if abs((a["adjusted_gap"] or 0) - (a["raw_gap"] or 0)) >= 0.03]
        if notable_adj:
            with st.expander("🎛️ Confidence-adjusted gap — where and why it differs from raw"):
                st.caption("Growth is weighted by each side's **confidence posture** "
                           "(1 − recent critical share): capability growth with little "
                           "self-critique reads as deployment-grade momentum (stronger "
                           "warning); safety growth that is itself highly critical is weaker "
                           "reassurance. Shown next to the raw gap, never replacing it.")
                for a in notable_adj:
                    st.markdown(
                        f"- **{a['pairing']}** — raw {a['raw_gap']*100:+.0f} → adjusted "
                        f"{a['adjusted_gap']*100:+.0f} pts (cap conf {a['cap_confidence']}, "
                        f"saf conf {a['saf_confidence']}): {a['reason']}")
        st.caption("**Volume balance** = which side has more papers now (cap÷saf ratio). "
                   "**Growth balance** = which side is accelerating faster (gap = capability "
                   "growth − safety growth). A pairing is flagged *safety lagging* when the "
                   "growth gap clears the threshold, capability growth is positive, and "
                   "capability has enough recent volume.")
        st.info("**Reading it:** safety can have *more papers overall* (safety-heavy volume "
                "balance) yet still be **lagging** if capability is accelerating while safety "
                "is flat or shrinking — the two columns answer different questions (how big vs. "
                "how fast).", icon="🧭")

        # --- Monitoring-debt curves (#3): cumulative capability-minus-safety backlog. ---
        debt = alerts.monitoring_debt(snap)
        debt = [d for d in debt if any(d["debt"])]
        if debt:
            st.divider()
            st.markdown("#### 📉 Monitoring-debt curves — accumulated backlog per pairing")
            st.caption("Cumulative Σ(capability − safety) papers per quarter. A one-quarter "
                       "gap is noise; a **rising** curve is persistent structural imbalance "
                       "(capability consistently out-producing its paired safety topic). The "
                       "*slope* is the signal — topics start at different baselines.")
            fig = go.Figure()
            for d in debt:
                fig.add_trace(go.Scatter(
                    x=d["periods"], y=d["debt"], mode="lines+markers", name=d["pairing"]))
            fig.update_layout(
                height=460, template="plotly_dark", xaxis_title=None,
                yaxis_title="cumulative capability − safety (papers)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None),
                margin=dict(t=10))
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig, width="stretch")
            worsening = [d["pairing"] for d in debt if d["rising"] and d["latest"] > 0]
            if worsening:
                st.markdown("**Debt still rising this quarter:** " + ", ".join(worsening))
    else:
        st.info("No pairings configured.")


def render_lab_lag():
    """🛰️ Lab-announcement → safety-response lag (#2) — the tool's namesake, quantified."""
    ll = snap.get("lab_lag") or {}
    st.divider()
    st.markdown("#### 🛰️ Lab-announcement → safety-response lag")
    if not ll.get("available"):
        st.caption("Not enough lab announcements tagged to a paired capability topic (with an "
                   "elapsed response window) to measure response lag in this snapshot.")
        return
    st.caption("For each lab/blog announcement tagged to a **capability** topic in a pairing, "
               "how long the paired **safety** research takes to answer in the arXiv literature "
               "— the first safety paper afterwards, and the first week safety volume rises "
               "measurably above its pre-announcement baseline. Announcements whose window "
               "hasn't elapsed are **pending**, not unanswered.")
    c1, c2, c3 = st.columns(3)
    mw = ll.get("median_weeks_to_measurable")
    c1.metric("Median safety-response lag", f"{mw} wk" if mw is not None else "—")
    md = ll.get("median_days_to_first")
    c2.metric("Median days to first safety paper", f"{md:.0f}" if md is not None else "—")
    c3.metric("Announcements measured", f"{ll.get('n_posts_considered', 0)}",
              help=f"{ll.get('n_window_elapsed', 0)} have a fully-elapsed response window")
    un = ll.get("unanswered") or {}
    if ll.get("n_window_elapsed"):
        st.markdown(
            f"**Unanswered after** — 4 wk: **{un.get('4', 0)}** · 8 wk: **{un.get('8', 0)}** · "
            f"12 wk: **{un.get('12', 0)}**  _(of {ll.get('n_window_elapsed')} elapsed-window "
            "announcements)_")
    byc = ll.get("by_capability") or []
    if byc:
        st.markdown("**Median response lag by capability topic:**")
        st.dataframe(pd.DataFrame([
            {"Capability topic": b["capability"], "Announcements": b["n"],
             "Median days→first": b["median_days_to_first"],
             "Median weeks→measurable": b["median_weeks_to_measurable"],
             "Unanswered": b["n_unanswered"]}
            for b in byc
        ]), width="stretch", hide_index=True)
    posts = ll.get("posts") or []
    if posts:
        _st_emoji = {"responded": "✅", "no measurable response": "⚠️", "pending": "⏳"}
        with st.expander(f"Per-announcement detail ({len(posts)})"):
            st.dataframe(pd.DataFrame([
                {"": _st_emoji.get(p["status"], ""), "Published": p["published"],
                 "Lab": p["lab"], "Announcement": p["announcement"],
                 "Capability": p["capability"], "Paired safety": p["safety"],
                 "Days→first": p["days_to_first"], "Weeks→measurable": p["weeks_to_measurable"],
                 "Status": p["status"]}
                for p in posts
            ]), width="stretch", hide_index=True)


with tab_div:
    st.subheader("⚖️ Capability vs. safety")
    delta_panel("divergence")
    if view_toggle("div_view", bool(weekly_block(snap).get("counts_by_key"))) == "weekly":
        render_weekly_divergence()
    else:
        render_divergence_overall()
    render_lab_lag()

# ================================================================== Velocity
def render_velocity_overall():
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
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No timeseries data.")
    st.markdown("**Velocity inflections** — which topics sped up or slowed down")
    infl = pd.DataFrame(snap["inflections"])
    if not infl.empty:
        infl = infl.sort_values("change", ascending=False)
        disp = pd.DataFrame({
            "Topic": infl["topic_key"].map(lambda k: lbl(snap, k)),
            "Recent /qtr": infl["recent_mean"].round(0).astype(int),
            "Prior /qtr": infl["prior_mean"].round(0).astype(int),
            "Change": (infl["change"] * 100).map(lambda x: f"{x:+.0f}%"),
            "Direction": infl["direction"].map(
                {"acceleration": "📈 accelerating", "deceleration": "📉 decelerating"}
            ).fillna("→ steady"),
        })
        st.dataframe(disp, width="stretch", hide_index=True)
    else:
        st.caption("No inflection data.")

    # --- Momentum vs. expected (#14): this week's volume against the quarterly baseline. ---
    wd = weekly_block(snap).get("window_days", 7)
    mom = alerts.weekly_momentum(snap, window_days=wd)
    if mom:
        st.divider()
        st.markdown(f"#### 📊 This week vs. expected — momentum (last {wd} days)")
        st.caption("Each topic's actual count this week against the count expected from its "
                   "recent quarterly baseline (scaled to the window). **z** is a Poisson "
                   "deviation (√expected) — |z| ≳ 2 is an anomalous spike/lull, not ordinary "
                   "weekly noise.")
        mdf = pd.DataFrame([
            {"Topic": lbl(snap, m["topic_key"]), "This week": m["actual"],
             "Expected": m["expected"], "Δ%": f"{m['pct']:+.0f}%", "z": f"{m['z']:+.1f}",
             "": ("🔥" if m["z"] >= 2 else ("❄️" if m["z"] <= -2 else ""))}
            for m in mom
        ])
        st.dataframe(mdf, width="stretch", hide_index=True)

    # --- Strategic map (formerly the Quadrant tab) — the same velocity data plotted as
    # recent volume × growth: emerging / hot / cooling / white-space. Quarterly only.
    st.divider()
    st.markdown("#### 🧭 Strategic map — emerging / hot / cooling / white-space")
    st.caption("The same topics plotted by **recent volume** (x) vs **growth** (y). "
               "Hover points for names.")
    if tab_analysis(snap, "quadrant"):
        st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'quadrant')}")
    quad = pd.DataFrame(snap["quadrant"])
    if not quad.empty:
        quad["topic"] = quad["topic_key"].map(lambda k: lbl(snap, k))
        fig = px.scatter(
            quad, x="recent_mean", y="change", color="quadrant", size="recent_mean",
            size_max=26, hover_name="topic",
            labels={"recent_mean": "Recent volume (papers/quarter)", "change": "Growth rate"},
            height=560, template="plotly_dark",
        )
        notable = quad[quad["quadrant"].isin(["emerging", "hot", "cooling"])]
        for _, r in notable.iterrows():
            fig.add_annotation(x=r["recent_mean"], y=r["change"], text=r["topic"],
                               showarrow=False, yshift=14, font=dict(size=10))
        fig.add_hline(y=0.3, line_dash="dot", line_color="gray")
        fig.add_vline(x=5, line_dash="dot", line_color="gray")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No quadrant data.")
    if snap["new_clusters"]:
        st.markdown("**Newly forming clusters (emergent, unsupervised):**")
        for c in snap["new_clusters"]:
            st.write(f"- {c}")


with tab_vel:
    st.subheader("📈 Topic velocity & strategic map")
    delta_panel("velocity")
    if view_toggle("vel_view", bool(weekly_block(snap).get("counts_by_key"))) == "weekly":
        render_weekly_velocity()
    else:
        render_velocity_overall()

# ================================================================== Sentiment
def render_sentiment_overall():
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
            lo, hi = alerts.wilson_interval(v.get("recent_share", 0), v.get("n_recent", 0))
            rows.append({
                "topic": lbl(snap, k),
                "critical_share": round(v.get("critical_share", 0) * 100, 1),
                "recent_share": round(v.get("recent_share", 0) * 100, 1),
                "recent_95CI": f"{lo*100:.0f}–{hi*100:.0f}%",
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
                     height=140 + 34 * len(sdf), template="plotly_dark")
        # Larger fonts + taller bars render crisper on mobile (the small default text
        # upscales blurrily on high-DPI phone screens).
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=30), coloraxis_showscale=False,
                          font=dict(size=14), bargap=0.25)
        st.plotly_chart(fig, width="stretch")
        if rising:
            st.markdown("**⚠️ Eroding-confidence warnings:**")
            for k, v in rising:
                st.markdown(
                    f"- **{lbl(snap, k)}** — critical share {v['prior_share']*100:.0f}% → "
                    f"{v['recent_share']*100:.0f}% (+{v['trend']*100:.0f} pts, {v['n_recent']} recent papers)"
                )
        # --- False-confidence alerts (#13): rising capability + falling self-critique. ---
        fc = alerts.false_confidence_alerts(snap)
        if fc:
            st.markdown("**🟣 Possible false-confidence signals** — *investigate, not confirmed*:")
            st.caption("A capability topic growing while its **critical share is falling** and "
                       "the paired safety topic is flat/shrinking. Falling criticism in a "
                       "fast-growing field can be genuine resolution — or premature deployment "
                       "confidence outrunning scrutiny. Distinct from safety-lag and "
                       "sentiment-erosion.")
            for a in fc:
                lab = " · 🛰️ recent lab activity" if a.get("lab_active") else ""
                st.markdown(
                    f"- **{lbl(snap, a['capability_topic'])}** growing "
                    f"{a['cap_growth']*100:+.0f}%/qtr, critical share {a['critical_trend']*100:+.0f} "
                    f"pts, paired safety **{lbl(snap, a['safety_topic'])}** "
                    f"{a['saf_growth']*100:+.0f}%/qtr{lab}")
        st.dataframe(sdf, width="stretch", hide_index=True)
        st.caption("Critical detection is embedding-based (cosine similarity to negative/"
                   "limitation seed phrases), not keyword matching — and it's a proxy, "
                   "so use it to decide where to read, not as a verdict. **recent_95CI** is "
                   "the Wilson interval on the recent share — wide intervals mean too few "
                   "papers to over-read a shift.")

        # --- Volume × critical-share quadrants (#11): momentum disambiguates sentiment. ---
        qrows = alerts.sentiment_quadrants(snap)
        if qrows:
            st.divider()
            st.markdown("#### 🧭 Sentiment quadrants — volume momentum × self-critique")
            st.caption("The same critical-share number means different things depending on "
                       "momentum. **Growing & straining** = expanding but hitting problems · "
                       "**growing & confident** = expanding with falling self-critique (check "
                       "the false-confidence list above) · **contracting & critical** = "
                       "shrinking into post-mortem · **fading/stabilizing** = cooling off.")
            qdf = pd.DataFrame(qrows)
            qdf["topic"] = qdf["topic_key"].map(lambda k: lbl(snap, k))
            fig = px.scatter(
                qdf, x="vol_change", y="crit_trend", color="quadrant", size="n_recent",
                size_max=28, hover_name="topic", text="topic",
                labels={"vol_change": "Volume change (growth rate)",
                        "crit_trend": "Critical-share trend (pts)"},
                height=520, template="plotly_dark",
                color_discrete_map={"growing & straining": "#ff6b6b",
                                    "growing & confident": "#ffd43b",
                                    "contracting & critical": "#845ef7",
                                    "fading / stabilizing": "#868e96"},
            )
            fig.update_traces(textposition="top center", textfont=dict(size=10))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(margin=dict(t=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No sentiment data in this snapshot.")


with tab_sentiment:
    st.subheader("🔬 Confidence / negative-signal tracker")
    delta_panel("sentiment")
    if view_toggle("sent_view", bool(weekly_block(snap).get("sentiment"))) == "weekly":
        render_weekly_sentiment()
    else:
        render_sentiment_overall()

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


def foresight_brief_md(fg) -> str:
    """A self-contained markdown brief of the foresight-gap synthesis."""
    lines = [f"# signal-lag — Foresight Gap brief ({meta['refreshed_at']})", ""]
    lines.append("_AI-surfaced candidate risks for an analyst to pressure-test — not "
                 "predictions. Each crosses the research-trend signals with broader "
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


def render_foresight_section():
    st.subheader("🔮 Foresight Gap — novel risks in the seam")
    delta_panel("foresight")
    weekly_fg = weekly_block(snap).get("foresight_gap")
    if view_toggle("foresight_view", bool(weekly_fg)) == "weekly":
        fg = weekly_fg
        st.caption("Risks surfaced from **just this week's papers** "
                   f"(last {weekly_block(snap).get('window_days', 7)} days), crossed with the "
                   "societal context. Same web-verification as the overall (quarterly) pass.")
    else:
        fg = get_analysis(snap).get("foresight_gap")
        if weekly_fg:
            st.caption("Risks from the **quarterly** research-trend signals.")
    # Humility framing — these are candidate hypotheses, not predictions.
    st.info(
        "**These are AI-surfaced *candidate* risks for an analyst to pressure-test — "
        "not predictions.** Each is anchored on the latest research-trend signal (a "
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

        def render_risk(r, i):
            v = r.get("verification")
            rating = (v or {}).get("novelty_rating")
            with st.container(border=True):
                st.markdown(f"**{novelty_label(v)}**")
                st.markdown(f"### {i}. {r.get('risk','')}")
                if r.get("priority") is not None:
                    _traj = {"accelerating": "📈", "steady": "→", "decelerating": "📉"}.get(
                        r.get("trajectory"), "")
                    st.markdown(
                        f"**🎚️ Priority {r.get('priority')}/25** · severity {r.get('severity')}/5 "
                        f"· likelihood {r.get('likelihood')}/5 (*{estimative(r.get('likelihood'))}*) "
                        f"· exposure {r.get('exposure')}/5 · trajectory {_traj} {r.get('trajectory','')}")
                    if r.get("confidence") is not None:
                        st.caption(
                            f"confidence {r.get('confidence')}/5 · evidence {r.get('evidence_strength')}/5 "
                            f"· actionability {r.get('actionability')}/5"
                            + (f" — {r['score_rationale']}" if r.get("score_rationale") else ""))
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
                # Epistemic claim labels (#8): observed / inferred / speculative, color-coded.
                claims = r.get("claims") or []
                if claims:
                    _basis_fmt = {"observed": ("🔵", "blue", "Observed"),
                                  "inferred": ("🟠", "orange", "Inferred"),
                                  "speculative": ("⚪", "gray", "Speculative")}
                    st.markdown("**🧬 Key claims by epistemic basis** "
                                "(🔵 measured · 🟠 reasoned from signals · ⚪ forward-looking):")
                    for c in claims:
                        dot, color, word = _basis_fmt.get(c.get("basis"),
                                                          _basis_fmt["speculative"])
                        st.markdown(f"- {dot} :{color}[**{word}:**] {c.get('text','')}")
                com = r.get("change_of_mind") or {}
                if any(com.get(k) for k in ("upgrade_if", "downgrade_if", "invalidate_if")):
                    with st.expander("🔀 What would change my mind (falsification conditions)"):
                        for lbl, key in (("⬆️ Upgrade if", "upgrade_if"),
                                         ("⬇️ Downgrade if", "downgrade_if"),
                                         ("❌ Invalidate if", "invalidate_if")):
                            if com.get(key):
                                st.markdown(f"**{lbl}:** {com[key]}")
                am = r.get("action_map") or {}
                if any(am.values()):
                    with st.expander("🛠️ So what — next actions"):
                        for lbl, key in (("🧪 Eval to run", "eval_to_run"),
                                         ("📊 Benchmark to monitor", "benchmark_to_monitor"),
                                         ("🛡️ Mitigation to consider", "mitigation"),
                                         ("🏛️ Policy question", "policy_question"),
                                         ("👤 Owner community", "owner_community"),
                                         ("🛰️ Data source to watch", "data_source_to_watch")):
                            if am.get(key):
                                st.markdown(f"**{lbl}:** {am[key]}")
                if v:
                    with st.expander("🔎 Prior-coverage check (web-verified)",
                                     expanded=(rating != "genuinely_unsurfaced")):
                        st.markdown(f"**What already exists:** {v.get('prior_coverage','')}")
                        st.markdown(f"**Disputed / contested claims:** {v.get('disputed_claims','')}")
                        st.markdown(f"**Recalibrated:** {v.get('recalibrated_calibration','')}")
                        for s in (v.get("sources") or [])[:8]:
                            t, u = s.get("title", ""), s.get("url", "")
                            st.markdown(f"- [{t}]({u})" if u else f"- {t}")
                elif fg.get("verified"):
                    st.caption("⚪ Prior-coverage check could not be completed for this risk.")
                pe = r.get("plain_explanation")
                if pe:
                    with st.expander("🧩 In plain terms — how the tool reasoned"):
                        for label, key in (("📄 The technical evidence", "technical_evidence"),
                                           ("🌍 The real-world context", "societal_evidence"),
                                           ("🔗 The gap (synthesis)", "the_gap"),
                                           ("🤔 The tool's own skepticism", "skepticism")):
                            if pe.get(key):
                                st.markdown(f"**{label}:** {pe[key]}")
                        if pe.get("bottom_line"):
                            st.markdown(f"**✅ Bottom line:** {pe['bottom_line']}")

        risks = fg.get("risks", [])
        demoted = [r for r in risks
                   if (r.get("verification") or {}).get("novelty_rating") == "already_widely_discussed"]
        surfaced = [r for r in risks if r not in demoted]

        if fg.get("verified"):
            cap = (f"✅ Web-checked against current coverage. **{len(surfaced)} surfaced** "
                   f"(genuinely-novel / partially-anticipated)")
            if demoted:
                cap += f" · {len(demoted)} already widely discussed (tucked below)"
            if fg.get("rounds", 1) > 1:
                cap += f" · {fg['rounds']} synthesis rounds (backfilled for quality)"
            st.caption(cap + ".")

        if surfaced:
            for i, r in enumerate(surfaced, 1):
                render_risk(r, i)
        else:
            st.warning("No risks survived novelty verification as fresh this cycle — all "
                       "candidates were already widely discussed. See below.", icon="🔍")

        if demoted:
            with st.expander(f"🔴 Already widely discussed ({len(demoted)}) — "
                             "shown for transparency, not novel", expanded=False):
                st.caption("The verifier found these (or close versions) are already public. "
                           "Kept so you can see what was considered and filtered out.")
                for j, r in enumerate(demoted, 1):
                    render_risk(r, j)

        st.download_button(
            "⬇️ Download foresight-gap brief", data=foresight_brief_md(fg),
            file_name="foresight_gap_brief.md", mime="text/markdown",
        )

        # Transparency: show exactly what fed the synthesis.
        st.divider()
        digest = fg.get("digest") or {}

        # Live web brief (#3): the dated real-world ground truth the synthesis verified
        # date/policy claims against (complements the static context.md).
        live_ctx = fg.get("live_context")
        if live_ctx:
            with st.expander("🛰️ Live web brief (current real-world status, web-verified)"):
                st.caption("A pre-synthesis web search pulled the CURRENT, dated status of "
                           "the flagged topics' real-world developments, so the synthesis "
                           "checks any date/policy claim against live ground truth instead "
                           "of a possibly-stale context file.")
                st.markdown(live_ctx)

        # Citation-VERIFIED cross-domain borrowing (#2): capability papers that actually
        # cite core safety work (via OpenAlex references), not just shared vocabulary.
        borrowers = (digest.get("citation_verified_borrowing") or [])
        if borrowers:
            with st.expander(f"🔗 Citation-verified borrowing ({len(borrowers)}) — "
                             "capability work that actually cites safety work"):
                st.caption("Verified via OpenAlex outgoing references (not keyword overlap). "
                           "Positive-only: absence from this list is **inconclusive**, never "
                           "evidence that a community ignores safety work.")
                for b in borrowers:
                    cap = ", ".join(b.get("capability_topics") or []) or "—"
                    cites = "; ".join(c for c in (b.get("cites_safety") or []) if c)
                    heat = (f" · {b['cited_by_count']:,} citations"
                            if b.get("cited_by_count") else "")
                    st.markdown(f"- **{b.get('title')}** ({cap}{heat}) → cites "
                                f"{b.get('n_cited_safety')} safety paper(s): {cites}")

        # Author migration (#4, experimental leading indicator).
        amig = digest.get("author_migration_experimental") or {}
        if amig.get("available") and amig.get("n_migrants"):
            with st.expander(f"🧭 Author migration — experimental ({amig.get('n_migrants')} "
                             "capability→safety authors)"):
                st.caption("⚠️ EXPERIMENTAL & noisy: built from a temporally-stratified "
                           "sample with imperfect author IDs. A capability→safety talent "
                           "flow can precede a wave of safety work — it INFORMS the brief, "
                           "never gates an alert.")
                for m in amig.get("examples") or []:
                    topics = ", ".join(m.get("entered_safety_topics") or [])
                    st.markdown(f"- **{m.get('author')}** — entered {topics} "
                                f"(after {m.get('prior_papers')} prior capability papers)")

        with st.expander("🔬 The signal digest that fed this synthesis"):
            st.caption("The strongest signals signal-lag computed for this run — the raw "
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
                st.caption("No societal context was provided for this run — the synthesis "
                           "reasoned across the scanning framework and general knowledge "
                           "only. Fill in `config/context.md` to sharpen it.")


# =============================================================== Harm Foresight
def render_harm_section():
    st.subheader("⚠️ Harm Foresight — the dual-use lens (0–24 months)")
    st.info(
        "A **dual-use foresight lens**: the same frontier papers, re-classified by which "
        "real-world **misuse** they could enable (cyber-offense, bio/chem uplift, influence "
        "ops, scams, agentic misuse, …) and how fast that enabling research is moving. It "
        "answers *'which harms is the literature quietly making easier, and how soon?'* — "
        "**a foresight signal over research, not on-platform abuse telemetry.**",
        icon="⚠️",
    )
    harm = snap.get("harm") or {}
    vectors = harm.get("vectors") or []
    if not vectors:
        st.warning("No harm-vector signal in this snapshot. Re-run the weekly refresh to "
                   "populate it (requires the `harm_topics` taxonomy).", icon="🔌")
    else:
        accel = [v for v in vectors if v.get("direction") == "acceleration"]
        if accel:
            st.markdown("**🚀 Accelerating harm vectors** (enabling research speeding up):")
            for v in accel:
                st.markdown(f"- **{v['label']}** — {v['change_pct']:+.0f}%/qtr, "
                            f"~{v['recent_per_qtr']:.0f} papers/qtr ({v['n_tagged']} tagged)")
        st.divider()
        st.markdown("**All harm vectors** (by momentum):")
        import pandas as _pd
        df = _pd.DataFrame([
            {"Harm vector": v["label"], "Trend %/qtr": v["change_pct"],
             "Recent/qtr": v["recent_per_qtr"], "Papers tagged": v["n_tagged"],
             "Direction": v["direction"]}
            for v in vectors
        ])
        st.dataframe(df, width="stretch", hide_index=True)
        st.divider()
        st.markdown("**What's driving each vector** — representative enabling papers:")
        for v in vectors:
            if not v.get("rep_papers"):
                continue
            with st.expander(f"{v['label']} · {v['change_pct']:+.0f}%/qtr · "
                             f"{v['n_tagged']} papers"):
                for rp in v["rep_papers"]:
                    st.markdown(f"- [{rp['title']}]({rp['url']}) "
                                f"<span style='color:gray'>· {rp['published']}</span>",
                                unsafe_allow_html=True)
                    if rp.get("abstract"):
                        st.caption(rp["abstract"])


# =============================================================== Risk register
def render_register_section():
    st.subheader("📋 Frontier risk register — prioritized")
    st.info(
        "An **evergreen, scored register** of every risk the foresight passes have surfaced. "
        "Ranked by **priority (severity × likelihood)**, then broken out of ties by "
        "**confidence → trajectory → evidence freshness → exposure** so there's a real top-5, "
        "not a pile at Priority 12. Risks not re-seen in the latest refresh are **⏳ downgraded "
        "as stale**. The standing watch-list, not just this week's snapshot.", icon="📋",
    )
    reg = risk_register
    if not reg:
        st.warning("No scored risks yet. The register fills in once a refresh runs with the "
                   "scoring layer (severity / likelihood / exposure / trajectory).", icon="🔌")
        return
    # Forced ranking (#5): a total order (priority, then confidence/trajectory/freshness/
    # exposure), with stale risks penalized — the rank number is printed on the matrix AND as
    # the first table column, so you can match a dot to its row without hovering.
    newest = register_newest_date(reg)
    ranked = sort_register(reg)

    def _disputed(e):
        t = ((e.get("latest") or {}).get("disputed_claims") or "").strip().lower()
        return bool(t) and not t.startswith("none")

    # --- Filters (#32): slice a growing register down to what you're triaging. ---
    with st.expander("🔍 Filter the register", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            f_traj = st.multiselect("Trajectory", ["accelerating", "steady", "decelerating"],
                                    default=[], help="Empty = all")
            f_minp = st.slider("Min priority", 1, 25, 1)
        with f2:
            f_nov = st.multiselect("Novelty", ["genuinely_unsurfaced", "partially_anticipated",
                                               "already_widely_discussed"], default=[])
            f_minc = st.slider("Min confidence", 1, 5, 1)
        with f3:
            f_fresh = st.radio("Freshness", ["All", "Fresh only", "Stale only"], index=0)
            f_disp = st.checkbox("⚖️ Disputed only",
                                 help="Risks where the web-verifier found a contested claim")

    def _keep(e):
        lt = e.get("latest") or {}
        if f_traj and lt.get("trajectory") not in f_traj:
            return False
        if (lt.get("priority") or 0) < f_minp:
            return False
        if f_nov and lt.get("novelty_rating") not in f_nov:
            return False
        if (lt.get("confidence") or 0) < f_minc and lt.get("confidence") is not None:
            return False
        stale = register_is_stale(e, newest)
        if f_fresh == "Fresh only" and stale:
            return False
        if f_fresh == "Stale only" and not stale:
            return False
        if f_disp and not _disputed(e):
            return False
        return True

    filtered = [e for e in ranked if _keep(e)]
    if len(filtered) < len(ranked):
        st.caption(f"Showing **{len(filtered)}** of {len(ranked)} risks after filters.")

    rows = []
    for i, e in enumerate(filtered, 1):
        lt = e.get("latest") or {}
        stale = register_is_stale(e, newest)
        rows.append({
            "#": i, "": ("⏳" if stale else "") + ("⚖️" if _disputed(e) else ""),
            "Priority": lt.get("priority"),
            "Sev": lt.get("severity"), "Lik": lt.get("likelihood"),
            "Estimative": estimative(lt.get("likelihood")), "Exp": lt.get("exposure"),
            "Conf": lt.get("confidence"), "Evid": lt.get("evidence_strength"),
            "Act": lt.get("actionability"),
            "Trajectory": lt.get("trajectory"), "Novelty": lt.get("novelty_rating"),
            "Risk": e.get("risk"), "First seen": e.get("first_seen"),
            "Last seen": e.get("last_seen"), "Seen ×": e.get("n_appearances"),
        })
    rdf = pd.DataFrame(rows)

    st.markdown(
        "**How to read the matrix:** each numbered dot is one risk, placed by **likelihood** "
        "(x → how probable in ~24 mo) and **severity** (y → how bad if it happens). **Bubble "
        "size = exposure** (breadth of impact); **colour = trajectory** (🔴 accelerating · "
        "🟡 steady · 🟢 fading). **Top-right is the danger zone** (likely *and* severe). The "
        "number on each dot is its priority rank — find it in the table below.")

    plot = rdf.dropna(subset=["Sev", "Lik"]).copy()
    if not plot.empty:
        import math
        plot["xj"] = plot["Lik"].astype(float)
        plot["yj"] = plot["Sev"].astype(float)
        # Spread dots that share the exact (likelihood, severity) cell so they don't stack.
        groups: dict = {}
        for idx, r in plot.iterrows():
            groups.setdefault((r["Lik"], r["Sev"]), []).append(idx)
        for (lik, sev), idxs in groups.items():
            if len(idxs) > 1:
                for k, idx in enumerate(idxs):
                    ang = 2 * math.pi * k / len(idxs)
                    plot.at[idx, "xj"] = lik + 0.20 * math.cos(ang)
                    plot.at[idx, "yj"] = sev + 0.20 * math.sin(ang)
        plot["Exposure"] = plot["Exp"].fillna(3)
        plot["label"] = plot["#"].astype(str)
        fig = px.scatter(
            plot, x="xj", y="yj", size="Exposure", color="Trajectory", text="label",
            hover_name="Risk", custom_data=["#"], size_max=30,
            labels={"xj": "Likelihood (1–5)", "yj": "Severity (1–5)"},
            category_orders={"Trajectory": ["accelerating", "steady", "decelerating"]},
            color_discrete_map={"accelerating": "#ff6b6b", "steady": "#ffd43b",
                                "decelerating": "#69db7c"},
            height=400, template="plotly_dark",
        )
        fig.update_traces(textposition="middle center",
                          textfont=dict(size=11, color="black"),
                          hovertemplate="#%{customdata[0]}: %{hovertext}<extra></extra>")
        fig.add_vline(x=3, line_dash="dot", line_color="gray", opacity=0.4)
        fig.add_hline(y=3, line_dash="dot", line_color="gray", opacity=0.4)
        fig.update_xaxes(range=[0.4, 5.6], tickvals=[1, 2, 3, 4, 5])
        fig.update_yaxes(range=[0.4, 5.6], tickvals=[1, 2, 3, 4, 5])
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Ranked register** (forced ranking, highest priority first):")
    st.dataframe(rdf, width="stretch", hide_index=True)
    n_stale = sum(1 for e in reg if register_is_stale(e, newest))
    st.caption(f"{len(reg)} risks tracked across all refreshes"
               + (f" · ⏳ {n_stale} stale (not re-seen in the latest refresh, downgraded)"
                  if n_stale else "")
               + ". Priority = severity × likelihood (1–25); **Conf** = calibrated confidence, "
               "**Evid** = evidence strength, **Act** = actionability (these break priority "
               "ties and set the forced rank). ⚖️ = a dispute was found. First-seen / "
               "last-seen show persistence.")

    # --- Counterevidence (#23): the evidence AGAINST, persisted across refreshes. ---
    disputed_risks = [e for e in filtered if _disputed(e) or e.get("counterevidence")]
    if disputed_risks:
        with st.expander(f"⚖️ Evidence & disputes ({len(disputed_risks)}) — what argues "
                         "AGAINST these risks"):
            st.caption("The web-verifier actively searches for disputing coverage; every "
                       "distinct dispute it has ever found for a risk is kept here (dated), "
                       "so counterevidence accumulates instead of being overwritten. A risk "
                       "that only ever collects confirming evidence isn't calibrated.")
            for e in disputed_risks:
                lt = e.get("latest") or {}
                st.markdown(f"**{e.get('risk','')}**")
                if lt.get("prior_coverage"):
                    st.markdown(f"- *Prior coverage:* {lt['prior_coverage']}")
                for c in (e.get("counterevidence")
                          or ([{"date": e.get("last_seen"),
                                "disputed_claims": lt.get("disputed_claims")}]
                              if _disputed(e) else [])):
                    st.markdown(f"- *Disputed ({c.get('date','')}):* {c.get('disputed_claims','')}")
                for s in (lt.get("sources") or []):
                    t, u = s.get("title", ""), s.get("url", "")
                    st.markdown(f"  - [{t}]({u})" if u else f"  - {t}")

    # --- Forecast-validation scaffold (#9): what the register's own history shows so far. ---
    cal = alerts.register_calibration(reg)
    if cal.get("n"):
        with st.expander("📏 Calibration & validation — is the register accountable to itself?"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risks re-seen ≥2×", cal.get("n_reseen", 0),
                      help="Persistence: a risk that keeps re-surfacing is a strengthening signal")
            c2.metric("Priority upgraded", cal.get("n_upgraded", 0),
                      help="First-seen vs latest history point")
            c3.metric("Priority downgraded", cal.get("n_downgraded", 0))
            c4.metric("Ever disputed", cal.get("n_ever_disputed", 0),
                      help="The web-verifier found counterevidence at least once")
            st.caption(
                f"Score history spans **{cal.get('n_refreshes', 0)} refresh(es)** "
                f"({cal.get('first_date', '—')} → {cal.get('last_date', '—')}). Full forecast "
                "validation — hit rate, false-positive rate, time-to-confirmation, Brier-style "
                "likelihood calibration — needs materialized/invalidated outcomes over several "
                "quarters of history. The tracking is in place now; those metrics will appear "
                "honestly as the register ages, rather than being simulated today.")


# =================================================================== Scenarios
# ICD-203-style estimative-probability vocabulary (Step 4): a 1-5 likelihood maps to a
# calibrated word so the output reads like an intelligence product, not a number soup.
ESTIMATIVE = {1: "very unlikely", 2: "unlikely", 3: "roughly even chance",
              4: "likely", 5: "very likely"}


def estimative(lik) -> str:
    return ESTIMATIVE.get(lik, "—")


def render_scenarios_section():
    st.subheader("🎬 Scenarios — how the top risks could evolve (6–24 mo)")
    st.info(
        "**Scenario analysis**, not prediction: a few plausible ways the highest-priority "
        "risks could unfold over the next 6–24 months — each with its **drivers**, the "
        "**early indicators** to watch, the **branch points** where the future forks, and "
        "**candidate mitigations**. Use them to pre-position, then pressure-test.", icon="🎬",
    )
    fg = get_analysis(snap).get("foresight_gap") or {}
    scenarios = fg.get("scenarios") or []
    if not scenarios:
        st.warning("No scenarios in this snapshot. They populate on a refresh with the "
                   "scenario pass enabled (`analysis.foresight.scenarios`).", icon="🔌")
        return
    for i, sc in enumerate(scenarios, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {sc.get('title','')}")
            st.markdown(f"**🕐 Horizon:** {sc.get('horizon','—')}  ·  "
                        f"**📊 Estimative likelihood:** *{sc.get('estimative_likelihood','—')}*")
            if sc.get("narrative"):
                st.markdown(sc["narrative"])
            def _bullets(label, key):
                items = sc.get(key) or []
                if items:
                    st.markdown(f"**{label}**")
                    for it in items:
                        st.markdown(f"- {it}")
            _bullets("⚙️ Drivers", "drivers")
            _bullets("📡 Leading indicators to watch", "leading_indicators")
            _bullets("🔀 Branch points", "branch_points")
            _bullets("🛡️ Candidate mitigations", "candidate_mitigations")
            if sc.get("linked_risks"):
                st.caption("Builds on: " + " · ".join(sc["linked_risks"]))


def intelligence_estimate_md(snap) -> str:
    """Templated Strategic Intelligence Estimate (Step 4) — no LLM, built from the snapshot."""
    meta = snap.get("meta", {})
    fg = get_analysis(snap).get("foresight_gap") or {}
    reg = sort_register(risk_register)
    lines = [f"# Strategic Intelligence Estimate — signal-lag ({meta.get('refreshed_at','')})", ""]
    lines.append("## Bottom line up front")
    if reg:
        top = reg[0]; lt = top.get("latest") or {}
        lines.append(f"- Highest-priority risk (**P{lt.get('priority')}/25**, "
                     f"*{estimative(lt.get('likelihood'))}*): {top.get('risk')}")
    lines.append(f"- Tracking **{len(reg)} risks** across all refreshes; "
                 f"**{len(fg.get('scenarios') or [])} active scenarios** (6–24 mo).")
    lines += ["", "## Key risks (ranked by priority)"]
    for i, e in enumerate(reg[:8], 1):
        lt = e.get("latest") or {}
        lines.append(
            f"{i}. **[P{lt.get('priority')}]** {e.get('risk')}  \n"
            f"   severity {lt.get('severity')}/5 · likelihood {lt.get('likelihood')}/5 "
            f"(*{estimative(lt.get('likelihood'))}*) · exposure {lt.get('exposure')}/5 · "
            f"trajectory {lt.get('trajectory')}")
        if lt.get("leading_indicator"):
            lines.append(f"   - Leading indicator: {lt['leading_indicator']}")
    scen = fg.get("scenarios") or []
    if scen:
        lines += ["", "## Scenarios (6–24 months)"]
        for i, sc in enumerate(scen, 1):
            lines.append(f"### {i}. {sc.get('title','')} — {sc.get('horizon','')} "
                         f"(*{sc.get('estimative_likelihood','')}*)")
            if sc.get("narrative"):
                lines.append(sc["narrative"])
            if sc.get("leading_indicators"):
                lines.append("- **Watch for:** " + "; ".join(sc["leading_indicators"]))
            if sc.get("candidate_mitigations"):
                lines.append("- **Mitigations:** " + "; ".join(sc["candidate_mitigations"]))
    # Plain-language briefings for the risks that have them (the top-N explained risks).
    explained = [r for r in (fg.get("risks") or []) if r.get("plain_explanation")]
    if explained:
        lines += ["", "## In plain terms — how each top risk was reasoned"]
        for i, r in enumerate(explained, 1):
            pe = r["plain_explanation"]
            lines.append(f"### {i}. {r.get('risk','')}")
            for label, key in (("The technical evidence", "technical_evidence"),
                               ("The real-world context", "societal_evidence"),
                               ("The gap (synthesis)", "the_gap"),
                               ("The tool's own skepticism", "skepticism"),
                               ("Bottom line", "bottom_line")):
                if pe.get(key):
                    lines.append(f"- **{label}:** {pe[key]}")
    # Falsification conditions + next actions (from the surfaced risks that carry them).
    surfaced = [r for r in (fg.get("risks") or [])
                if (r.get("verification") or {}).get("novelty_rating") != "already_widely_discussed"]
    fa = [r for r in surfaced if r.get("change_of_mind") or r.get("action_map")]
    if fa:
        lines += ["", "## What would change our mind & next actions"]
        for i, r in enumerate(fa, 1):
            lines.append(f"### {i}. {r.get('risk','')}")
            com = r.get("change_of_mind") or {}
            for label, key in (("Upgrade if", "upgrade_if"), ("Downgrade if", "downgrade_if"),
                               ("Invalidate if", "invalidate_if")):
                if com.get(key):
                    lines.append(f"- **{label}:** {com[key]}")
            am = r.get("action_map") or {}
            for label, key in (("Eval to run", "eval_to_run"),
                               ("Benchmark to monitor", "benchmark_to_monitor"),
                               ("Mitigation", "mitigation"), ("Policy question", "policy_question"),
                               ("Owner community", "owner_community"),
                               ("Data source to watch", "data_source_to_watch")):
                if am.get(key):
                    lines.append(f"- **{label}:** {am[key]}")
    lines += ["", "## Confidence & caveats",
              "- Scores are AI-assigned and calibrated, not actuarial; likelihood is lowered "
              "where a risk leans on a contested or inferential claim.",
              "- signal-lag measures research *attention*/enablement, not deployed abuse — "
              "these are candidate hypotheses to pressure-test, not forecasts."]
    return "\n".join(lines)


def tabletop_pack_md(snap) -> str:
    """Templated tabletop-exercise pack (Step 4) from the top scenario / top risk."""
    fg = get_analysis(snap).get("foresight_gap") or {}
    scen = fg.get("scenarios") or []
    reg = sort_register(risk_register)
    base = scen[0] if scen else None
    title = (base or {}).get("title") if base else (reg[0].get("risk") if reg else "AI emerging risk")
    lines = [f"# Tabletop Exercise — {title}", "",
             "**Format:** 60–90 min · 4–6 participants · facilitator-led", "",
             "## Scenario setup"]
    lines.append((base or {}).get("narrative") if base
                 else (reg[0].get("risk") if reg else "(no scenario available)"))
    lines += ["", "## Roles",
              "- **Policy / Global Affairs** — regulatory & external-narrative response",
              "- **Product / Safety** — mitigations on-surface",
              "- **Detection / Intelligence** — what we'd see and when",
              "- **Comms** — public posture", ""]
    lines += ["## Injects (escalating)"]
    inds = (base or {}).get("leading_indicators") or (
        [reg[0].get("latest", {}).get("leading_indicator")] if reg else [])
    for i, ind in enumerate([x for x in inds if x][:4], 1):
        lines.append(f"{i}. **Inject {i}:** {ind} is now observed. What do we do in the next 72h?")
    branches = (base or {}).get("branch_points") or []
    for b in branches[:2]:
        lines.append(f"- **Branch point:** {b}")
    lines += ["", "## Discussion questions",
              "1. What would we detect *first*, and is anyone currently watching that signal?",
              "2. Which mitigation do we pull, who owns it, and what's the lead time?",
              "3. Where do two teams each see half the problem but neither owns the whole?",
              "4. What would make this *worse* faster than expected?",
              "", "## Outputs",
              "- A ranked action list with owners.", "- Gaps in detection/monitoring to close.",
              "- A revised estimate of likelihood/severity after the discussion."]
    return "\n".join(lines)


# =================================================================== Incidents
_QUAD_EMOJI = {"materializing": "🔴", "foresight lead": "🎯",
               "active / known": "🟠", "quiet": "⚪"}


def render_incidents_section():
    st.subheader("🌐 Incidents & benchmark — research lead vs. real-world reality")
    delta_panel("incidents")
    st.info(
        "The **all-source** layer: real, already-occurred AI-misuse **incidents** (gathered "
        "via web search from the AI Incident Database / OECD / news, verifiable + dated) "
        "**crossed against** each harm vector's research momentum. The 2×2: **🎯 foresight "
        "lead** = research accelerating, *no public incidents yet* (the early-warning edge); "
        "**🔴 materializing** = research up *and* incidents appearing; **🟠 active/known** = "
        "incidents but research flat; **⚪ quiet** = neither.", icon="🌐",
    )
    inc = snap.get("incidents") or {}
    bench = inc.get("benchmark") or []
    records = inc.get("records") or []
    if not bench:
        st.warning("No incident benchmark in this snapshot. It populates on a refresh with "
                   "the incidents pass enabled (`analysis.incidents`).", icon="🔌")
        return
    leads = [b for b in bench if b["quadrant"] == "foresight lead"]
    if leads:
        st.markdown("**🎯 Foresight leads** — research accelerating, *no public incidents yet*:")
        for b in leads:
            st.markdown(f"- **{b['label']}** — research {b['research_change_pct']:+.0f}%/qtr "
                        f"over {b['n_research']} papers, 0 incidents")
    st.markdown("**Benchmark — every harm vector:**")
    bdf = pd.DataFrame([
        {"": _QUAD_EMOJI.get(b["quadrant"], ""), "Harm vector": b["label"],
         "Research Δ%/qtr": b["research_change_pct"], "Research papers": b["n_research"],
         "Incidents": b["n_incidents"], "Status": b["quadrant"]}
        for b in bench
    ])
    st.dataframe(bdf, width="stretch", hide_index=True)

    # --- Early-warning calibration scaffold (#28): do foresight leads predict incidents? ---
    trans = alerts.benchmark_transitions(bench_history)
    if trans.get("n_refreshes", 0) >= 1:
        with st.expander("⏱️ Early-warning calibration — do 🎯 foresight leads materialize?"):
            st.caption("Each refresh's benchmark is persisted, so 'research accelerating with "
                       "no incidents yet' episodes can be checked against LATER refreshes for "
                       "incidents appearing — time-to-incident and false-positive rates for "
                       "the harm-vector framework itself.")
            if trans.get("materialized"):
                st.markdown("**Leads that materialized** (incidents appeared after the lead):")
                for m in trans["materialized"]:
                    st.markdown(f"- 🎯→🔴 **{m['label']}** — lead {m['lead_date']} → "
                                f"incidents by {m['incident_date']}")
            if trans.get("open_leads"):
                st.markdown("**Open leads** (research accelerating, still no public incidents):")
                for m in trans["open_leads"]:
                    st.markdown(f"- 🎯 **{m['label']}** — leading since {m['lead_date']}")
            if trans["n_refreshes"] < 3:
                st.caption(f"⏳ Only **{trans['n_refreshes']} refresh(es)** of benchmark "
                           "history so far — time-to-incident and false-positive rates need "
                           "several quarters to be meaningful. They will accrue automatically.")
    st.divider()
    st.markdown(f"**Recent real-world incidents** ({len(records)}):")
    if not records:
        st.caption("No verifiable incidents were surfaced this refresh.")
    st.caption("Public incident data is uneven and attribution is often uncertain, so each "
               "incident is graded by credibility (🟢 high · 🟡 medium · ⚪ low) — the weakest "
               "of AI-involvement confidence, attribution confidence, and source quality. "
               "Lower-confidence reports are tucked below.")
    lm = (snap.get("harm") or {}).get("label_map", {})
    _conf_emoji = {"high": "🟢", "medium": "🟡", "low": "⚪"}
    _sev_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}

    def _render_incident(r):
        label = lm.get(r.get("harm_key"), r.get("harm_key"))
        url = r.get("source_url")
        title = r.get("title", "")
        head = f"[{title}]({url})" if url else title
        badge = _conf_emoji.get(r.get("confidence"), "")
        meta = f"_{r.get('date','')}_ · {label}"
        if r.get("severity"):
            meta += f" · severity {_sev_emoji.get(r['severity'],'')} {r['severity']}"
        if r.get("affected_sector"):
            meta += f" · {r['affected_sector']}"
        st.markdown(f"- {badge} **{head}** · {meta}")
        if r.get("summary"):
            extra = f"  ·  {r['deployer']}" if r.get("deployer") else ""
            conf = (f"  ·  AI-involvement {r.get('ai_involvement_confidence','?')}, "
                    f"attribution {r.get('attribution_confidence','?')}, "
                    f"source {r.get('source_quality','?')}")
            st.caption(r["summary"] + extra + conf)

    higher = [r for r in records if r.get("confidence") != "low"]
    lower = [r for r in records if r.get("confidence") == "low"]
    for r in higher:
        _render_incident(r)
    if lower:
        with st.expander(f"⚪ Lower-confidence reports ({len(lower)}) — alleged / weakly-sourced"):
            for r in lower:
                _render_incident(r)


def render_plain_terms_section():
    st.subheader("🧩 Risks in plain terms — the policy-facing briefing")
    st.info(
        "The top foresight risks explained for a **non-specialist or executive**: what the "
        "research shows, the real-world context, why the gap matters, what the tool itself "
        "doubts, and the bottom line separating **observed** from **projected**. This is the "
        "same reasoning as the risk cards, in plain language.", icon="🧩",
    )
    risks = explained_risks(snap)
    if not risks:
        st.warning("No plain-language explanations in this snapshot. They populate on a "
                   "refresh with explainers enabled (`analysis.foresight.explainers`).",
                   icon="🔌")
        return
    st.download_button("⬇️ Download plain-terms briefing (markdown)",
                       data=plain_language_brief_md(snap),
                       file_name="signal_lag_risks_in_plain_terms.md", mime="text/markdown")
    for i, r in enumerate(risks, 1):
        pe = r["plain_explanation"]
        with st.container(border=True):
            st.markdown(f"### {i}. {r.get('risk','')}")
            for label, key in (("📄 What the research shows", "technical_evidence"),
                               ("🌍 The real-world context", "societal_evidence"),
                               ("🔗 Why the gap matters", "the_gap"),
                               ("🤔 What the tool itself doubts", "skepticism")):
                if pe.get(key):
                    st.markdown(f"**{label}:** {pe[key]}")
            if pe.get("bottom_line"):
                st.success(f"**✅ Bottom line:** {pe['bottom_line']}")


with tab_foresight:
    _fmode = st.radio(
        "Foresight view",
        ["🔮 Cross-domain risks", "⚠️ Harm vectors (dual-use)", "📋 Risk register",
         "🎬 Scenarios", "🌐 Incidents", "🧩 Plain terms"],
        horizontal=True, label_visibility="collapsed",
    )
    if _fmode.startswith("⚠️"):
        render_harm_section()
    elif _fmode.startswith("📋"):
        render_register_section()
    elif _fmode.startswith("🎬"):
        render_scenarios_section()
    elif _fmode.startswith("🌐"):
        render_incidents_section()
    elif _fmode.startswith("🧩"):
        render_plain_terms_section()
    else:
        render_foresight_section()
    # Analyst-ready exports (Step 4) — templated, available under every view.
    if get_analysis(snap).get("foresight_gap") or risk_register:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Intelligence estimate", data=intelligence_estimate_md(snap),
                               file_name="signal_lag_intelligence_estimate.md",
                               mime="text/markdown", width="stretch")
        with c2:
            st.download_button("⬇️ Tabletop pack", data=tabletop_pack_md(snap),
                               file_name="signal_lag_tabletop.md",
                               mime="text/markdown", width="stretch")


# =================================================================== Sources
with tab_sources:
    st.subheader("Source papers")
    # Live-data provenance lives here — the source-of-truth tab.
    st.success(
        f"🟢 **Live data** · refreshed **{meta['refreshed_at']}** · {meta['n_papers']:,} papers "
        f"({meta['date_start']} → {meta['date_end']}) · {', '.join(meta['categories'])} · weekly."
    )
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
    cites = snap.get("citations") or {}
    has_movers = bool(cites.get("rapid_growth") or cites.get("sleepers"))
    if has_movers:
        if tab_analysis(snap, "citations"):
            st.markdown(f"**🧠 Claude's read:** {tab_analysis(snap, 'citations')}")

        def _cite_line(r):
            extra = (f" · {r['influential_citations']} influential"
                     if r.get("influential_citations") else "")
            ven = f" · {r['venue']}" if r.get("venue") else ""
            return f"- [{r['title']}]({r['url']}) — {r['cited_by_count']} cites{extra}{ven}"

        def _render_cites(bucket):
            for r in cites.get(bucket, [])[:10]:
                st.markdown(_cite_line(r))
                d = short_desc(r, 160)
                if d:
                    st.caption(d)

        ccol, scol = st.columns(2)
        with ccol:
            st.markdown("**🔥 Rapid citation growth**")
            _render_cites("rapid_growth")
        with scol:
            st.markdown("**💤 Sleepers (early-heat)**")
            _render_cites("sleepers")
    else:
        st.caption("📊 Citation-velocity movers (rapid-growth / sleepers) need OpenAlex's "
                   "year-by-year citation series, which is currently unreachable from the "
                   "refresh runner — so this section is hidden. Per-paper citation counts and "
                   "the **citation-verified borrowing** in Foresight come from Semantic Scholar.")

    # --- Citation cross-pollination (#16/#17/#18): do the communities actually engage? ---
    cg = snap.get("citation_graph") or {}
    if cg:
        st.divider()
        st.markdown("### 🔗 Citation cross-pollination — do the fields talk to each other?")
        cov = cg.get("coverage") or {}
        st.caption(f"Built from **real outgoing references** (Semantic Scholar), not shared "
                   f"vocabulary. Coverage is partial — {cov.get('pct_with_references', 0)}% of "
                   f"tagged papers ({cov.get('n_with_references', 0):,} of "
                   f"{cov.get('n_tagged', 0):,}) had reference data — so **absence of an edge "
                   "is inconclusive**, never proof two fields ignore each other.")

        m = cg.get("matrix_cap_to_saf") or {}
        if m:
            st.markdown("**Capability → safety citations** (which capability fields cite "
                        "which safety work):")
            caps = sorted(m.keys())
            safs = sorted({s for row in m.values() for s in row})
            z = [[m.get(c, {}).get(s, 0) for s in safs] for c in caps]
            fig = go.Figure(go.Heatmap(
                z=z, x=safs, y=caps, colorscale="Blues",
                text=z, texttemplate="%{text}", showscale=False))
            fig.update_layout(height=120 + 42 * len(caps), template="plotly_dark",
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(tickfont=dict(size=10)),
                              yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig, width="stretch")

        bridges = cg.get("bridge_papers") or []
        if bridges:
            st.markdown("**🌉 Bridge papers** — work connecting the two communities "
                        "(often precedes field convergence):")
            for b in bridges[:10]:
                sides = []
                if b.get("capability_topics"):
                    sides.append("⚡ " + ", ".join(b["capability_topics"]))
                if b.get("safety_topics"):
                    sides.append("🛡️ " + ", ".join(b["safety_topics"]))
                marks = []
                if b.get("dual_tagged"):
                    marks.append("dual-tagged")
                if b.get("n_cross_citations"):
                    marks.append(f"{b['n_cross_citations']} cross-boundary citation(s)")
                url = arxiv_url(b["arxiv_id"]) if b.get("arxiv_id") else None
                head = f"[{b.get('title')}]({url})" if url else b.get("title")
                st.markdown(f"- **{head}** — {' · '.join(sides)}  \n"
                            f"  _{' · '.join(marks)}_")

        impact = cg.get("safety_impact") or []
        if impact:
            st.markdown("**🏆 Safety papers capability builders actually cite** — safety "
                        "*uptake*, not just safety output:")
            idf = pd.DataFrame([
                {"Safety paper": r.get("title"),
                 "Capability citers (in-corpus)": r.get("n_capability_citers"),
                 "Total citations": r.get("cited_by_count"),
                 "Influential": r.get("influential_citations"),
                 "Topics": ", ".join(r.get("safety_topics") or [])}
                for r in impact[:12]
            ])
            st.dataframe(idf, width="stretch", hide_index=True)

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

    with st.expander("📤 Structured data exports — CSV / JSON for your own analysis"):
        st.caption("The same data behind the dashboard, in machine-readable form for memos, "
                   "notebooks, and downstream tooling. Everything is from the current snapshot.")
        date = meta.get("refreshed_at", "")
        c1, c2, c3 = st.columns(3)
        with c1:
            if risk_register:
                st.download_button("⬇️ Risk register (CSV)", data=register_csv(risk_register),
                                   file_name=f"risk_register_{date}.csv", mime="text/csv",
                                   width="stretch")
            st.download_button("⬇️ Topic velocity (CSV)", data=velocity_csv(snap),
                               file_name=f"topic_velocity_{date}.csv", mime="text/csv",
                               width="stretch")
            if (snap.get("lab_lag") or {}).get("posts"):
                st.download_button("⬇️ Lab-response lag (CSV)", data=lab_lag_csv(snap),
                                   file_name=f"lab_response_lag_{date}.csv", mime="text/csv",
                                   width="stretch")
        with c2:
            if (snap.get("incidents") or {}).get("records"):
                st.download_button("⬇️ Incidents (CSV)", data=incidents_csv(snap),
                                   file_name=f"incidents_{date}.csv", mime="text/csv",
                                   width="stretch")
            if snap.get("harm"):
                st.download_button("⬇️ Harm vectors (JSON)",
                                   data=json.dumps(snap["harm"], ensure_ascii=False, indent=2),
                                   file_name=f"harm_vectors_{date}.json",
                                   mime="application/json", width="stretch")
            if snap.get("citation_graph"):
                st.download_button("⬇️ Citation matrix (CSV)", data=citation_matrix_csv(snap),
                                   file_name=f"citation_matrix_{date}.csv", mime="text/csv",
                                   width="stretch")
        with c3:
            st.download_button("⬇️ Full snapshot (JSON)",
                               data=json.dumps(snap, ensure_ascii=False),
                               file_name=f"snapshot_{date}.json", mime="application/json",
                               width="stretch")

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
Already-discussed risks are tucked into a collapsed **"already widely discussed"**
expander (flagged, not hidden). **Quality over quantity:** if too few candidates survive
as fresh, the synthesis re-runs for *different* seams (up to a few rounds) and verifies
those too, so a week where several risks turn out already-covered still surfaces genuine
ones. The searches are baked into the snapshot (run once per refresh, cached — never at
page load). This is the calibrated posture: the tool generates candidate risks **and then
checks them against current coverage before surfacing them**, distinguishing a genuine
seam from something that simply isn't in its index yet.

**These are candidate hypotheses to pressure-test, not predictions** — the model widens
the aperture; human judgment goes on top.

### 11. "This week" lens + History
The quarterly view above is the slow-moving baseline. Two layers make the tool
longitudinal and timely:
- **This week** — a toggle on the **Summary** and **Foresight Gap** tabs analyzes *only*
  the papers submitted in the last `window_days` (default 7): topic counts, a focused
  Claude "what landed this week" read, notable papers, and a full (web-verified) this-week
  Foresight Gap. The quarterly charts are unaffected; an extra recent-window arXiv pull
  guarantees the 7-day set is complete. Anchored on the quarterly research-trend signal,
  then crossed with this week's papers.
- **History (📜)** — each refresh appends a compact briefing (headline, top foresight gaps
  overall + this week, rising sentiment, what-changed) to `data/history.json`, shown as a
  metrics-over-time chart + a reverse-chronological list — the running record a single
  snapshot can't show.

### 12. Signal-fidelity layers (how much to trust each read)
Four upgrades raise how much an analyst can trust the output. All are config-gated and
fail-soft (no API key ⇒ the prior behavior, unchanged):
- **Hybrid sentiment (embedding recall → LLM precision).** The rising-critical-share
  signal starts from a cheap embedding score, but that centroid mistakes *academic
  negation* ("we **overcome** the catastrophic failures of prior methods") for genuine
  criticism. So the recent-window papers the embedding flags critical are **re-checked by
  Claude** ("is this paper's core stance that something fails — or is it constructive?")
  and false positives are downgraded before the trend is computed. Bounded to that subset.
- **Citation-FLOW, not just citation-heat.** The cross-silo "borrowing" story is verified
  against **real citation references** (each paper's outgoing bibliography, by arXiv id, from
  **Semantic Scholar** — OpenAlex is unreachable from the CI runner) — a capability/applied
  paper that *actually cites* a core safety paper, not one that merely shares its vocabulary.
  Surfaced **positive-only**: a verified citation is strong evidence; *absence is
  inconclusive*, never "they ignore it" (the cited work may sit outside our sample).
- **Live web context (fresh ground truth).** Fast arXiv data was being crossed with a
  hand-maintained `context.md` that can be months stale. Before synthesis, **one web
  search** pulls the *current, dated* status of the flagged topics' real-world
  developments, so the synthesis verifies any date/policy claim against live truth
  (complements, never replaces, the analyst's file).
- **Author migration (experimental).** Using Semantic Scholar author IDs, the tool flags
  authors who were capability-dominant historically and whose **recent** papers enter a
  safety/oversight topic — a capability→safety talent flow that can precede a wave of
  safety work. **Clearly labeled experimental and noisy** (stratified sample + imperfect
  author IDs): it informs the brief, never gates an alert.

### 13. Harm Foresight — the dual-use lens (the ⚠️ Harm vectors view in the 🔮 Foresight tab)
The capability/safety taxonomy tracks the *research*; this layer re-classifies the **same
papers** by which real-world **misuse** they could enable — a parallel "harm-vector" taxonomy
(cyber-offense, bio/chem uplift, influence operations, scams & fraud, agentic misuse,
surveillance, model-weight exfiltration, jailbreak/guardrail-evasion, child-safety,
harassment). Each vector gets the same velocity treatment, so the tool answers *"which harms
is the frontier literature quietly making easier, and how fast?"* over a **0–24 month** horizon.
Accelerating harm vectors are fed into the Foresight Gap synthesis, which is instructed to
frame risks as **capability → harm enablement** with a concrete leading indicator and the
defender community that isn't watching that seam. It is a **foresight signal over research, not
on-platform abuse telemetry** — an *enabling* signal, not proof of imminent abuse. The harm
taxonomy lives in `config/taxonomy.yaml` (`harm_topics`) and is fully editable.

### 14. Frontier risk register — prioritized (the 📋 view in 🔮 Foresight)
Every risk the foresight passes surface is **scored** by Claude at synthesis time on four
axes — **severity**, **likelihood** (over ~24 months), **exposure** (breadth), and
**trajectory** (is the enabling signal accelerating / steady / fading) — with **priority =
severity × likelihood** (1–25). Those scores are accumulated into an **evergreen register**
(`data/risk_register.json`): each risk keyed by a stable id with *first-seen / last-seen*, an
appearance count, and a per-refresh **score history**, so you can watch a risk's priority and
trajectory move over time. The 📋 view renders the standard **priority matrix** (likelihood ×
severity, bubble = exposure, colour = trajectory) plus a sortable table — the JD's "evergreen
frontier risk register and prioritization framework (severity, prevalence, exposure,
trajectory)." Scores are deliberately calibrated, not alarmist: likelihood is lowered wherever
a risk leans on a contested or inferential claim.

### 15. Scenarios + intelligence-estimate / tabletop exports (the 🎬 view + ⬇️ buttons)
A further Claude pass takes the **top-priority register risks** and develops a few **6–24
month scenarios** — each with drivers, the early **leading indicators** to watch, the
**branch points** where the future forks, candidate **mitigations**, and an ICD-203-style
**estimative-likelihood** word. These map the possibility space (not predictions) so an
analyst can pre-position. Two **templated, one-tap exports** (no extra model calls) turn the
data into analyst-ready products: a **Strategic Intelligence Estimate** (BLUF + ranked risks
in estimative language + scenarios + confidence caveats) and a **Tabletop-Exercise pack**
(scenario setup, roles, escalating injects built from the leading indicators, discussion
questions). Likelihood scores are shown throughout in **estimative-probability language**
(*very unlikely → very likely*) so the output reads like an intelligence product.

For the **top-N risks**, a further pass writes a **plain-language "🧩 In plain terms"
walkthrough** — five sections that explain *how the tool reasoned*: the **technical evidence**
(the actual papers + trend metric), the **real-world context** it crossed with, the **gap**
(synthesis), the **tool's own skepticism** (where the evidence is contested or a projection),
and a **bottom line** separating what's *observed* from what's *projected*. Shown as an
expander on each risk and folded into the downloadable intelligence estimate.

### 16. Incidents & benchmark — the all-source layer (the 🌐 view)
Everything above is **upstream** (the research signal — leading). This layer adds the
**downstream** half: REAL, already-occurred AI-misuse **incidents**, gathered via Claude's
web search from the **AI Incident Database / OECD AI Incidents Monitor / news** (constrained
to verifiable, dated, sourced incidents), categorized into the same **harm vectors**. Each
vector is then **benchmarked** — research momentum vs incident count — into a leading-vs-
lagging **2×2**: **🎯 foresight lead** (research accelerating, *no public incidents yet* — the
early-warning edge), **🔴 materializing** (research up *and* incidents appearing), **🟠
active/known** (incidents but research flat), **⚪ quiet**. This is the "all-source /
competitive-benchmarking" dimension — though note it's *public external* incidents (which lag
and have uneven coverage), **not** internal platform telemetry.

### Caveats
- High coverage of the **AI preprint literature**, not every publisher.
- Velocity tracks each topic's *share* of activity (stratified sample), not raw totals.
- Quality depends on the embedding backend and the taxonomy seed phrases — all
  config-driven in `config/taxonomy.yaml` and `config/settings.yaml`.

Full details: [github.com/delschlangen/signal-lag](https://github.com/delschlangen/signal-lag).
"""
    )

# =================================================================== History
with tab_history:
    st.subheader("📜 History — weekly briefings over time")
    if not history:
        st.info("No history yet — a compact briefing is saved on each weekly refresh. "
                "Entries appear here from the next refresh onward.", icon="🕰️")
    else:
        st.caption(f"{len(history)} weekly briefing(s) recorded. Newest first.")
        hdf = pd.DataFrame([
            {"date": e.get("date"), "Safety-lag alerts": e.get("n_flagged"),
             "Papers": e.get("n_papers")}
            for e in history if e.get("date")
        ])
        if len(hdf) >= 2:
            hdf = hdf.sort_values("date")
            fig = px.line(hdf, x="date", y="Safety-lag alerts", markers=True,
                          template="plotly_dark", height=260)
            fig.update_layout(yaxis_title="alerts", xaxis_title=None, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Safety-lag alerts per refresh — the longitudinal signal the snapshot "
                       "alone can't show.")

        for e in sorted(history, key=lambda x: x.get("date") or "", reverse=True):
            hl = e.get("headline") or {}
            gap = hl.get("biggest_gap_line", "")
            with st.expander(f"🗓️ {e.get('date', '?')} — {gap}", expanded=False):
                if hl.get("meaning"):
                    st.markdown(hl["meaning"])
                if hl.get("why_it_matters"):
                    st.markdown(f"**Why it matters:** {hl['why_it_matters']}")
                wc = e.get("what_changed") or {}
                st.caption(
                    f"Papers: {e.get('n_papers') or 0:,} · Safety-lag alerts: "
                    f"{e.get('n_flagged', '?')}/{e.get('n_pairings', '?')} · new alerts "
                    f"{wc.get('n_alerts', 0)}, new accel {wc.get('n_accel', 0)}, new sleepers "
                    f"{wc.get('n_sleepers', 0)}"
                )
                if e.get("sentiment_rising"):
                    st.markdown("**⚠️ Rising critical share:** "
                                + ", ".join(e["sentiment_rising"]))
                for label, key in [("🔮 Top foresight gaps (overall)", "overall_foresight"),
                                   ("🆕 Top foresight gaps (this week)", "weekly_foresight")]:
                    rows = e.get(key) or []
                    if rows:
                        st.markdown(f"**{label}:**")
                        for r in rows:
                            nb = NOVELTY_BADGE.get(r.get("novelty_rating"), ("⚪", ""))[0]
                            st.markdown(f"- {nb} {r.get('risk', '')}")
        st.download_button("⬇️ Download full history (JSON)",
                           data=json.dumps(history, indent=1, ensure_ascii=False),
                           file_name="signal_lag_history.json", mime="application/json")

# ============================================================ BLUF sidebar (#31)
# Always-visible executive summary: the main judgments survive tab navigation.
with st.sidebar:
    st.markdown("### 🎯 Bottom line up front")
    st.caption(f"Snapshot {meta.get('refreshed_at', '')}")

    _flagged = sorted([d for d in snap.get("divergence") or [] if d.get("lagging")],
                      key=lambda d: d.get("gap") or 0, reverse=True)
    if _flagged:
        st.markdown(f"**🚨 Widest safety lag:** {_flagged[0].get('pairing')} "
                    f"(gap {_flagged[0].get('gap', 0)*100:+.0f} pts)")

    _reg_ranked = sort_register(risk_register)
    if _reg_ranked:
        _lt = _reg_ranked[0].get("latest") or {}
        st.markdown(f"**📋 Top register risk (P{_lt.get('priority')}):** "
                    f"{(_reg_ranked[0].get('risk') or '')[:120]}")

    _newr = (_DELTAS.get("foresight") or {}).get("new_risks") or []
    if _newr:
        st.markdown(f"**🆕 Top new risk this refresh:** {_newr[0][:120]}")

    _mom = alerts.weekly_momentum(snap, window_days=weekly_block(snap).get("window_days", 7))
    _spike = next((m for m in _mom if abs(m["z"]) >= 2), None)
    if _spike:
        st.markdown(f"**📊 Biggest weekly anomaly:** {lbl(snap, _spike['topic_key'])} "
                    f"({_spike['pct']:+.0f}%, z {_spike['z']:+.1f})")

    _bench = (snap.get("incidents") or {}).get("benchmark") or []
    _hot = next((b for b in _bench if b.get("quadrant") == "materializing"),
                next((b for b in _bench if b.get("quadrant") == "foresight lead"), None))
    if _hot:
        st.markdown(f"**⚠️ Harm vector to watch:** {_hot.get('label')} "
                    f"({_QUAD_EMOJI.get(_hot.get('quadrant'), '')} {_hot.get('quadrant')})")

    _ll = snap.get("lab_lag") or {}
    if _ll.get("available") and _ll.get("median_weeks_to_measurable") is not None:
        st.markdown(f"**🛰️ Median lab→safety response:** "
                    f"{_ll['median_weeks_to_measurable']} wk "
                    f"({_ll.get('n_posts_considered', 0)} announcements)")

    st.divider()
    st.download_button("📄 One-page brief", data=compact_brief_md(snap),
                       file_name="signal_lag_one_page_brief.md",
                       mime="text/markdown", width="stretch", key="sb_brief")
    st.download_button("⬇️ Intelligence estimate", data=intelligence_estimate_md(snap),
                       file_name="signal_lag_intelligence_estimate.md",
                       mime="text/markdown", width="stretch", key="sb_est")
    if explained_risks(snap):
        st.download_button("🧩 Risks in plain terms", data=plain_language_brief_md(snap),
                           file_name="signal_lag_risks_in_plain_terms.md",
                           mime="text/markdown", width="stretch", key="sb_plain")

st.caption(
    f"Embedding backend: {meta['backend']} · snapshot v{meta.get('version', 1)} · "
    "signal-lag · [github.com/delschlangen/signal-lag](https://github.com/delschlangen/signal-lag)"
)

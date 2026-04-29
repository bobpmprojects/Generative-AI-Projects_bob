"""Streamlit app for one-click market intelligence memo generation."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

from agent.cache import IntelCache
from agent.critic import critique_memo
from agent.fetcher import gather_company_intel
from agent.market import get_market_context
from agent.schemas import CritiqueReport, ExecMemo, MarketContextReport, ResearchPlan
from agent.scope import build_plan
from agent.synthesizer import build_exec_memo, revise_memo

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BRIEF = (
    "I'm a PM at a hyperscaler tracking the AI inference platform market — specifically how "
    "Together AI and Fireworks compare to Microsoft Azure AI Foundry, Google Vertex AI, and Amazon Bedrock. "
    "I want enterprise traction, differentiation, and partnership risk over the last six months."
)
PRICING_PER_1K = {"gpt-4o-mini": 0.0003, "gpt-4o": 0.01, "gpt-5.5": 0.02}


def apply_exec_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --navy: #071a2f;
            --blue: #1d4ed8;
            --sky: #e0f2fe;
            --gold: #c99700;
            --ivory: #fbfaf7;
            --line: #d8dee9;
            --muted: #5b6472;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 78, 216, 0.12), transparent 30rem),
                linear-gradient(180deg, #f7f4ed 0%, #ffffff 34%);
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071a2f 0%, #0f2744 100%);
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }
        h1 {
            color: var(--navy);
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        h2, h3 {
            color: var(--navy);
            letter-spacing: -0.02em;
        }
        div[data-testid="stTabs"] button {
            color: var(--navy);
            font-weight: 700;
        }
        .executive-hero {
            background: linear-gradient(135deg, #071a2f 0%, #123c69 58%, #1d4ed8 100%);
            border-radius: 18px;
            padding: 1.35rem 1.6rem;
            margin: 1rem 0 1.3rem 0;
            color: white;
            box-shadow: 0 16px 36px rgba(7, 26, 47, 0.24);
            border-bottom: 4px solid var(--gold);
        }
        .executive-hero .eyebrow {
            color: #fde68a;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .executive-hero .title {
            font-size: 1.65rem;
            line-height: 1.15;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }
        .executive-hero .meta {
            color: #dbeafe;
            font-size: 0.95rem;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.4rem 0 1rem 0;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-top: 4px solid var(--gold);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            box-shadow: 0 8px 24px rgba(7, 26, 47, 0.08);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: var(--navy);
            font-size: 1.25rem;
            font-weight: 800;
        }
        .report-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-left: 7px solid var(--blue);
            border-radius: 16px;
            padding: 1.1rem 1.35rem;
            margin: 0.5rem 0 1rem 0;
            box-shadow: 0 10px 28px rgba(7, 26, 47, 0.10);
        }
        .verdict-pill {
            display: inline-block;
            color: #fff;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.02em;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 16px rgba(7, 26, 47, 0.16);
        }
        .risk-title {
            margin: 0.6rem 0 0.3rem 0;
            font-weight: 700;
            color: var(--navy);
        }
        .footnote {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 1.2rem;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #071a2f, #1d4ed8);
            border: 0;
            color: white;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(29, 78, 216, 0.24);
        }
        div[data-testid="stDownloadButton"] button {
            border-color: var(--blue);
            color: var(--navy);
            font-weight: 700;
        }
        .dark-report {
            background:
                radial-gradient(circle at 15% 0%, rgba(20,184,166,0.14), transparent 26rem),
                linear-gradient(180deg, #071a1f 0%, #0a1117 100%);
            color: #dbeafe;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 18px;
            padding: 1.3rem;
            box-shadow: 0 22px 50px rgba(2, 6, 23, 0.34);
        }
        .dark-report h2, .dark-report h3 {
            color: #f8fafc;
            font-family: Georgia, 'Times New Roman', serif;
            letter-spacing: -0.03em;
        }
        .report-section-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(148, 163, 184, 0.28);
            margin: 1.3rem 0 0.7rem 0;
            padding-bottom: 0.35rem;
            color: #f8fafc;
            font-family: Georgia, 'Times New Roman', serif;
            font-weight: 700;
            font-size: 1.2rem;
        }
        .section-kicker {
            color: #67e8f9;
            font-family: Inter, sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .bluf-box {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(12, 42, 49, 0.96));
            border: 1px solid rgba(34, 211, 238, 0.30);
            border-left: 4px solid #f59e0b;
            border-radius: 14px;
            padding: 1rem;
            font-size: 1.02rem;
            line-height: 1.55;
            color: #f8fafc;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .action-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .dark-card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 12px;
            padding: 0.85rem;
            color: #cbd5e1;
        }
        .dark-card strong {
            color: #f8fafc;
        }
        .accent-bar {
            width: 3px;
            height: 1.4rem;
            background: #22d3ee;
            display: inline-block;
            margin-right: 0.45rem;
            vertical-align: middle;
        }
        .source-link {
            display: block;
            color: #67e8f9 !important;
            border-top: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.45rem 0;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("live_runs", 0)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("cost_usd", 0.0)


def est_cost(model: str, usage: dict[str, Any]) -> float:
    if usage.get("cached"):
        return 0.0
    tokens = float(usage.get("total_tokens", 0))
    return (tokens / 1000.0) * PRICING_PER_1K.get(model, 0.002)


def update_cost(model: str, usage: dict[str, Any]) -> None:
    st.session_state.cost_usd += est_cost(model, usage)


def render_open_questions_callout(critique: CritiqueReport) -> None:
    """Surface follow-ups only; machine verdict labels (e.g. Revise) are omitted here."""
    questions = [str(q).strip() for q in (critique.open_questions_for_followup or []) if str(q).strip()]
    if not questions:
        return
    st.markdown("##### Follow-up questions")
    for item in questions:
        st.markdown(f"- {item}")


def render_report_header(plan: ResearchPlan, critique: CritiqueReport) -> None:
    st.markdown(
        f"""
        <div class="executive-hero">
            <div class="eyebrow">Executive Intelligence Memo</div>
            <div class="title">{plan.sector}</div>
            <div class="meta">
                Decision context: {plan.decision_context.replace("_", " ").title()} ·
                Companies: {len(plan.companies)}
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Coverage</div>
                <div class="metric-value">{len(plan.companies)} companies</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lookback</div>
                <div class="metric-value">{plan.lookback_days} days</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Auto QC score</div>
                <div class="metric-value">{critique.confidence_score}/100</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _dominant_sentiment(labels: list[str]) -> str:
    norm: list[str] = []
    for raw in labels:
        k = (raw or "").strip().lower() or "mixed"
        if k not in {"positive", "negative", "mixed", "neutral"}:
            k = "mixed"
        norm.append(k)
    if not norm:
        return "no labeled signals"
    counts: dict[str, int] = {}
    for k in norm:
        counts[k] = counts.get(k, 0) + 1
    top = max(counts, key=counts.get)
    return top


def render_social_reviews_summary(intel: dict[str, Any], memo: ExecMemo) -> None:
    """Names covered plus an executive-style sentiment read (no raw tables)."""
    companies = sorted(intel.keys())
    if not companies:
        st.info("No company intel was returned for this run.")
        return

    st.markdown("### Companies covered")
    st.markdown(", ".join(f"**{escape(c)}**" for c in companies))

    all_social: list[str] = []
    all_review: list[str] = []
    per_company: list[str] = []
    for name in companies:
        bag = intel.get(name) or {}
        social = bag.get("social_sentiment") or []
        reviews = bag.get("customer_review_signals") or []
        s_labels = [str(r.get("sentiment", "") or "") for r in social]
        r_labels = [str(r.get("sentiment", "") or "") for r in reviews]
        all_social.extend(s_labels)
        all_review.extend(r_labels)
        if not social and not reviews:
            per_company.append(f"- **{escape(name)}:** no discrete social or review rows extracted.")
        else:
            s_dom = _dominant_sentiment(s_labels) if social else "—"
            r_dom = _dominant_sentiment(r_labels) if reviews else "—"
            per_company.append(
                f"- **{escape(name)}:** community signals lean **{s_dom}**; "
                f"buyer/review signals lean **{r_dom}**."
            )

    st.markdown("### Per-company sentiment (from extracted signals)")
    st.markdown("\n".join(per_company))

    st.markdown("### Overall read")
    body = (memo.social_customer_sentiment_read or "").strip()
    if body:
        st.markdown(body)
    else:
        st.caption("The memo did not include a synthesized social and customer narrative for this run.")

    if all_social or all_review:
        mix_parts = []
        if all_social:
            mix_parts.append(f"community: mostly **{_dominant_sentiment(all_social)}** ({len(all_social)} signals)")
        if all_review:
            mix_parts.append(f"reviews: mostly **{_dominant_sentiment(all_review)}** ({len(all_review)} signals)")
        st.markdown("### Aggregate signal mix")
        st.caption("Across all companies — " + "; ".join(mix_parts) + ".")


def render_streamlit_sources(title: str, sources: list[Any]) -> None:
    st.markdown(f"### {title}")
    if not sources:
        st.caption("No bibliography entries returned for this run.")
        return
    for idx, src in enumerate(sources, start=1):
        bag = src.model_dump() if hasattr(src, "model_dump") else src
        label = str(bag.get("title") or bag.get("url") or "Source")
        url = str(bag.get("url") or "").strip()
        if url:
            st.markdown(f"{idx}. [{label}]({url})")
        else:
            st.markdown(f"{idx}. {label}")
def _html_multiline_card(text: str, empty_msg: str = "No content returned for this section.") -> str:
    raw = (text or "").strip()
    if not raw:
        return f'<div class="dark-card empty-section">{escape(empty_msg)}</div>'
    return f'<div class="dark-card long-form">{escape(raw).replace(chr(10), "<br>")}</div>'


def _html_company_deep_summaries(memo: ExecMemo) -> str:
    if not memo.company_deep_summaries:
        return '<div class="dark-card empty-section">No per-company deep dives were returned.</div>'
    parts: list[str] = []
    for cs in memo.company_deep_summaries:
        name = escape(str(cs.company_name or "Company"))
        body = escape(str(cs.summary_markdown or "").strip()).replace("\n", "<br>")
        parts.append(
            '<div class="company-dive">'
            f'<div class="report-section-title"><span>{name}</span>'
            '<span class="section-kicker">Deep dive (4–5 paragraphs)</span></div>'
            f'<div class="dark-card long-form">{body}</div>'
            "</div>"
        )
    return "".join(parts)


def _card_grid(items: list[str], cols: int = 2) -> str:
    css_class = "action-grid" if cols == 3 else "insight-grid"
    cards = []
    for idx, item in enumerate(items, start=1):
        cards.append(
            "<div class='dark-card'>"
            f"<strong><span class='accent-bar'></span>{idx:02d}</strong><br>{escape(str(item))}"
            "</div>"
        )
    return f"<div class='{css_class}'>{''.join(cards)}</div>"


def render_premium_memo(memo: ExecMemo, critique: CritiqueReport) -> None:
    risk_cards = _card_grid(critique.top_3_risks_to_recipient or memo.risks_watch_items[:3], cols=3)
    inner = f"""
        <div class="dark-report">
            <div class="section-kicker">Board-Ready Competitive Intelligence</div>
            <h2>{escape(str(memo.sector))} Market Read</h2>
            <div class="bluf-box"><strong>Bottom Line:</strong> {escape(str(memo.bottom_line))}</div>

            <div class="report-section-title"><span>Key Movements</span><span class="section-kicker">What Changed</span></div>
            {_card_grid(memo.key_movements[:6], cols=2)}

            <div class="report-section-title"><span>Competitive Dynamics</span><span class="section-kicker">Strategic Read</span></div>
            <div class="dark-card">{escape(str(memo.competitive_dynamics))}</div>

            <div class="report-section-title"><span>Analyst Reports &amp; Investor Analysis</span><span class="section-kicker">Capital markets &amp; sell-side</span></div>
            {_html_multiline_card(memo.analyst_reports_investor_analysis)}

            <div class="report-section-title"><span>Growth, Revenue &amp; Projections</span><span class="section-kicker">From news &amp; market context</span></div>
            {_html_multiline_card(memo.growth_revenue_projections_from_news)}

            <div class="insight-grid">
                <div>
                    <div class="report-section-title"><span>Investor Sentiment</span></div>
                    <div class="dark-card">{escape(str(memo.investor_sentiment_read))}</div>
                </div>
                <div>
                    <div class="report-section-title"><span>Social & Customer Sentiment</span></div>
                    <div class="dark-card">{escape(str(memo.social_customer_sentiment_read or "See Social & Reviews tab for extracted evidence."))}</div>
                </div>
            </div>

            <div class="report-section-title"><span>Company Deep Dives</span><span class="section-kicker">Full narrative per company</span></div>
            {_html_company_deep_summaries(memo)}

            <div class="report-section-title"><span>Risks</span><span class="section-kicker">Recipient Watchouts</span></div>
            {risk_cards}

            <div class="report-section-title"><span>Recommended Actions</span><span class="section-kicker">Next Moves</span></div>
            {_card_grid(memo.recommended_actions[:6], cols=3)}

            <div class="section-kicker" style="margin-top:1rem;">References open in normal browser tabs below this panel.</div>
        </div>
    """
    doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ margin:0; padding:0; background:transparent; font-family:Inter,system-ui,sans-serif; }}
  .dark-report {{
    background: linear-gradient(180deg, #071a1f 0%, #0a1117 100%);
    color: #dbeafe;
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 18px;
    padding: 1.3rem;
  }}
  .dark-report h2 {{
    color: #f8fafc;
    font-family: Georgia, 'Times New Roman', serif;
    letter-spacing: -0.03em;
    margin-top: 0.2rem;
  }}
  .section-kicker {{
    color: #67e8f9;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }}
  .report-section-title {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(148, 163, 184, 0.28);
    margin: 1.3rem 0 0.7rem 0;
    padding-bottom: 0.35rem;
    color: #f8fafc;
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 700;
    font-size: 1.2rem;
  }}
  .bluf-box {{
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(12, 42, 49, 0.96));
    border: 1px solid rgba(34, 211, 238, 0.30);
    border-left: 4px solid #f59e0b;
    border-radius: 14px;
    padding: 1rem;
    font-size: 1.02rem;
    line-height: 1.55;
    color: #f8fafc;
  }}
  .insight-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.8rem;
  }}
  .action-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
  }}
  .dark-card {{
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 12px;
    padding: 0.85rem;
    color: #cbd5e1;
  }}
  .dark-card strong {{
    color: #f8fafc;
  }}
  .dark-card.long-form {{
    line-height: 1.62;
    font-size: 0.98rem;
  }}
  .dark-card.empty-section {{
    color: #94a3b8;
    font-style: italic;
  }}
  .company-dive {{
    margin-bottom: 1.1rem;
  }}
  .accent-bar {{
    width: 3px;
    height: 1.4rem;
    background: #22d3ee;
    display: inline-block;
    margin-right: 0.45rem;
    vertical-align: middle;
  }}
  .source-link {{
    display: block;
    color: #67e8f9 !important;
    border-top: 1px solid rgba(148, 163, 184, 0.18);
    padding: 0.45rem 0;
    text-decoration: none;
  }}
</style>
</head>
<body>
  {inner}
</body>
</html>
"""
    height = max(780, min(5200, 820 + len(doc) // 3))
    components.html(doc, height=height, scrolling=True)


def load_demo_payload() -> dict[str, Any]:
    return json.loads((BASE_DIR / "samples" / "sample_run.json").read_text(encoding="utf-8"))


def get_config_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        value = ""
    return str(value or os.getenv(name, "")).strip()


def render_live_error(error: Exception) -> None:
    error_name = type(error).__name__
    raw_detail = str(error)
    detail = raw_detail.lower()
    stage = ""
    if raw_detail.startswith("Stage failed: "):
        try:
            stage = raw_detail.split("Stage failed: ", 1)[1].split(" ->", 1)[0]
        except Exception:
            stage = ""
    if "tavily" in error_name.lower() or "invalid api key" in detail:
        st.error("Tavily rejected the API key in Streamlit secrets. Verify TAVILY_API_KEY.")
    elif "openai" in detail and "authentication" in detail:
        st.error("OpenAI rejected the API key in Streamlit secrets. Verify OPENAI_API_KEY.")
    elif "badrequesterror" in detail:
        st.error(
            f"OpenAI rejected the {stage or 'request'}. "
            "If you chose GPT-5.5, confirm it is enabled for your key; otherwise pick GPT-4o. "
            "Synthesis and critique automatically retry on GPT-4o when the primary model fails — check the message below."
        )
    elif "validationerror" in detail:
        st.error(
            f"Model output for stage '{stage}' did not match the schema. "
            "Retry the run; the underlying message is below."
        )
    else:
        st.error(f"Live run failed at stage '{stage or 'unknown'}'.")
    st.code(raw_detail[:1800])


def _stage(name: str, fn):
    try:
        return fn()
    except Exception as exc:
        raise RuntimeError(f"Stage failed: {name} -> {type(exc).__name__}: {exc}") from exc


def run_live(brief: str, openai_key: str, tavily_key: str, synth_model: str, critic_model: str) -> None:
    client = OpenAI(api_key=openai_key)
    cache = IntelCache()
    st.session_state.cost_usd = 0.0
    with st.status("Running Market Intel Agent...", expanded=True) as status:
        st.write("1/5 Parsing research brief into plan...")
        plan, usage = _stage("scoping_plan", lambda: build_plan(client, brief, model="gpt-4o-mini"))
        update_cost("gpt-4o-mini", usage)

        st.write("2/5 Building market context (TAM / growth signals from web evidence)...")
        market_context, m_usage = _stage(
            "market_context",
            lambda: get_market_context(client, cache, tavily_key, plan),
        )
        update_cost("gpt-4o-mini", m_usage)

        st.write("3/5 Gathering company intel, social sentiment, and customer reviews in parallel...")
        done: list[str] = []
        prog = st.empty()

        def on_company(name: str) -> None:
            done.append(name)
            prog.info(f"Gathered: {', '.join(done)}")

        intel, usages = _stage(
            "company_intel_gather",
            lambda: gather_company_intel(client, cache, tavily_key, plan, status_cb=on_company),
        )
        for u in usages:
            update_cost("gpt-4o-mini", u)

        st.write("4/5 Synthesizing executive memo...")
        memo, usage = _stage(
            "memo_synthesis",
            lambda: build_exec_memo(client, plan, intel, market_context, model=synth_model),
        )
        update_cost(synth_model, usage)

        st.write("5/5 Red-teaming memo critique...")
        critique, usage = _stage(
            "memo_critique",
            lambda: critique_memo(client, plan, memo.full_markdown, memo.sources, model=critic_model),
        )
        update_cost(critic_model, usage)

        status.update(label="Run complete", state="complete")

    st.session_state.result = {
        "plan": plan.model_dump(),
        "market_context": market_context.model_dump(),
        "intel": {k: v.model_dump() for k, v in intel.items()},
        "memo": memo.model_dump(),
        "critique": critique.model_dump(),
        "memo_v1": memo.full_markdown,
    }
    st.session_state.live_runs += 1


def main() -> None:
    st.set_page_config(page_title="Market Intel Agent", layout="wide")
    apply_exec_theme()
    init_state()

    st.title("Market Intel Agent — Live Competitive Intelligence")
    st.caption("Describe what you want to know. Get an executive-ready market memo in ~3 minutes.")

    with st.sidebar:
        st.caption("UI v2 · Memo, Follow-ups, Social & Reviews, Sources — if you see Research Plan, restart from repo `app.py`.")
        demo_mode = st.toggle("Demo Mode", value=True)
        model_options = ["gpt-5.5", "gpt-4o", "gpt-4o-mini"]
        synth_model = st.selectbox("Synthesis model", model_options, index=0)
        critic_model = st.selectbox("Critic model", model_options, index=0)
        st.caption(
            "Default: GPT-5.5 for synthesis and critique when your API key supports it; "
            "falls back to GPT-4o on error. Use GPT-4o-mini to reduce cost."
        )
        openai_key = get_config_secret("OPENAI_API_KEY")
        tavily_key = get_config_secret("TAVILY_API_KEY")
        if not demo_mode:
            if openai_key and tavily_key:
                st.success("Live API keys loaded from Streamlit secrets.")
            else:
                st.caption("Missing secrets can be entered here for this session only.")
                if not openai_key:
                    openai_key = st.text_input(
                        "OpenAI API Key",
                        type="password",
                        placeholder="sk-proj-... (from platform.openai.com/api-keys)",
                    )
                if not tavily_key:
                    tavily_key = st.text_input(
                        "Tavily API Key",
                        type="password",
                        placeholder="tvly-... (from app.tavily.com/home)",
                    )
        if st.button("Force refresh"):
            removed = IntelCache().clear_ttl_entries()
            st.success(f"Cleared {removed} TTL cache entries.")
        st.metric("Live cost estimate", f"${st.session_state.cost_usd:.3f}")
        st.write(f"Live runs this session: {st.session_state.live_runs} / 1")

    brief = st.text_area("Research Brief", value=DEFAULT_BRIEF, height=100)
    st.caption(
        "Describe the market or question. Mention specific companies if you have them, or just describe the space — the agent will infer relevant companies."
    )

    disabled = (not demo_mode and st.session_state.live_runs >= 1) or not brief.strip()
    if st.button("Generate Memo", type="primary", disabled=disabled):
        if demo_mode:
            st.session_state.result = load_demo_payload()
        else:
            if not openai_key or not tavily_key:
                st.error("Live mode requires both OpenAI and Tavily keys.")
            else:
                try:
                    run_live(brief, openai_key, tavily_key, synth_model, critic_model)
                except Exception as exc:
                    render_live_error(exc)

    if not demo_mode and st.session_state.live_runs >= 1:
        st.warning("Demo limited to 1 run per session. Refresh to reset, or clone the repo to run unlimited locally.")

    result = st.session_state.result
    if not result:
        st.stop()

    memo = ExecMemo.model_validate(result["memo"])
    critique = CritiqueReport.model_validate(result["critique"])
    plan = ResearchPlan.model_validate(result["plan"])
    memo_text = memo.full_markdown
    if critique.overall_verdict == "reject":
        st.error("⚠️ This memo failed critique — review before sharing.")

    tabs = st.tabs(["📋 Memo", "❓ Follow-ups", "💬 Social & Reviews", "🔗 Sources"])

    with tabs[0]:
        render_report_header(plan, critique)
        render_open_questions_callout(critique)
        memo_with_risks = (
            f"{memo_text}\n\n## Confidence & Risks\n"
            f"- Confidence score: **{critique.confidence_score}/100**\n"
            + "\n".join(f"- Risk: {r}" for r in critique.top_3_risks_to_recipient)
        )
        render_premium_memo(memo, critique)
        if result.get("market_context"):
            mc = MarketContextReport.model_validate(result["market_context"])
            with st.expander("Market context evidence (TAM / growth)"):
                st.markdown(mc.executive_summary or "_No executive summary returned._")
                st.markdown(f"**TAM / spend signals:** {mc.tam_and_spend_signals or '—'}")
                st.markdown(f"**Growth / CAGR signals:** {mc.growth_and_cagr_signals or '—'}")
                if mc.demand_drivers:
                    st.markdown("**Demand drivers:** " + "; ".join(mc.demand_drivers))
                if mc.headwinds:
                    st.markdown("**Headwinds:** " + "; ".join(mc.headwinds))
                if mc.methodology_caveats:
                    st.caption(mc.methodology_caveats)
                render_streamlit_sources("Supporting market sources", mc.supporting_sources)
        render_streamlit_sources("Memo bibliography (clickable)", memo.sources)
        with st.expander("Full cited memo text"):
            st.markdown(memo_with_risks)
        st.download_button("Export as .md", data=memo_with_risks, file_name="market_intel_memo.md")
        if "memo_v1" in result and result.get("memo_v1") != result.get("memo_v2"):
            with st.expander("View v1 memo"):
                st.markdown(result["memo_v1"])

    with tabs[1]:
        st.markdown("### Open questions")
        if critique.open_questions_for_followup:
            for item in critique.open_questions_for_followup:
                st.write(f"- {item}")
        else:
            st.caption("No follow-up questions were returned for this run.")

        revise_disabled = demo_mode or not openai_key
        if st.button("🔄 Revise memo using machine critique", disabled=revise_disabled):
            with st.spinner("Revising memo..."):
                client = OpenAI(api_key=openai_key)
                revised, usage = revise_memo(
                    client,
                    result.get("memo_v2", result["memo"]["full_markdown"]),
                    critique.model_dump(),
                    model=synth_model,
                )
                update_cost(synth_model, usage)
                result["memo_v1"] = result.get("memo_v1", result["memo"]["full_markdown"])
                result["memo_v2"] = revised
                result["memo"]["full_markdown"] = revised
                st.session_state.result = result
                st.success("Memo revised. Check Memo tab for v2.")
        if revise_disabled:
            st.caption("Revision requires live mode with a valid OpenAI key.")

        with st.expander("Detailed machine critique (scores, risks, findings)"):
            st.caption(f"Internal QC confidence: {critique.confidence_score}/100 — not a human verdict.")
            st.write(critique.confidence_rationale)
            st.markdown("**Top risks to recipient**")
            for risk in critique.top_3_risks_to_recipient:
                st.write(f"- {risk}")
            rows = [
                {
                    "Severity": f.severity,
                    "Dimension": f.dimension,
                    "Location": f.location,
                    "Finding": f.finding,
                    "Recommended Fix": f.recommended_fix,
                }
                for f in critique.findings
            ]
            df = pd.DataFrame(rows)
            if not df.empty:
                severity_rank = {"high": 0, "medium": 1, "low": 2}
                df["rank"] = df["Severity"].map(severity_rank).fillna(9)
                df = df.sort_values(by=["rank", "Dimension"]).drop(columns=["rank"])
                styled = df.style.apply(
                    lambda row: [
                        (
                            "background-color: #fee2e2; color: #7f1d1d;"
                            if row["Severity"] == "high"
                            else "background-color: #fef3c7; color: #78350f;"
                            if row["Severity"] == "medium"
                            else "background-color: #f3f4f6; color: #374151;"
                        )
                        for _ in row
                    ],
                    axis=1,
                )
                st.dataframe(styled, use_container_width=True)
            else:
                st.info("No findings rows returned.")
            st.markdown("**Strongest aspects**")
            for item in critique.strongest_aspects:
                st.write(f"- {item}")

    with tabs[2]:
        render_social_reviews_summary(result["intel"], memo)

    with tabs[3]:
        for idx, src in enumerate(memo.sources, start=1):
            data = src.model_dump() if hasattr(src, "model_dump") else src
            title = data.get("title") or data.get("url") or "Source"
            st.markdown(f"{idx}. [{title}]({data.get('url', '#')})")

    st.markdown(
        "<div class='footnote'>Built by Your Name · github.com/your-user/market-intel-agent</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

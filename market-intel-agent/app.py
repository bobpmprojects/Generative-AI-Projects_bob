"""Streamlit app for one-click market intelligence memo generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agent.cache import IntelCache
from agent.critic import critique_memo
from agent.fetcher import gather_company_intel
from agent.schemas import CritiqueReport, ExecMemo, ResearchPlan
from agent.scope import build_plan
from agent.synthesizer import build_exec_memo, revise_memo

load_dotenv()

DEFAULT_BRIEF = (
    "I'm a PM at a hyperscaler tracking the AI inference platform market — specifically how "
    "Together AI, Fireworks, and Baseten are differentiating against Bedrock and Vertex. "
    "Want to know who's winning enterprise share over the last 6 months and which is most likely to IPO."
)
PRICING_PER_1K = {"gpt-4o-mini": 0.0003, "gpt-4o": 0.01}


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


def verdict_color(verdict: str) -> str:
    return {"ship": "green", "revise": "orange", "reject": "red"}.get(verdict, "gray")


def render_badge(verdict: str) -> str:
    colors = {"ship": "#15803d", "revise": "#c99700", "reject": "#b91c1c"}
    bg = colors.get(verdict, "#4b5563")
    return f"<span class='verdict-pill' style='background:{bg};'>Verdict: {verdict.upper()}</span>"


def render_report_header(plan: ResearchPlan, critique: CritiqueReport) -> None:
    st.markdown(
        f"""
        <div class="executive-hero">
            <div class="eyebrow">Executive Intelligence Memo</div>
            <div class="title">{plan.sector}</div>
            <div class="meta">
                Decision context: {plan.decision_context.replace("_", " ").title()} ·
                Confidence: {critique.confidence_score}/100 ·
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
                <div class="metric-label">Critique</div>
                <div class="metric-value">{critique.overall_verdict.title()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_demo_payload() -> dict[str, Any]:
    return json.loads(Path("samples/sample_run.json").read_text(encoding="utf-8"))


def get_config_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        value = ""
    return str(value or os.getenv(name, ""))


def run_live(brief: str, openai_key: str, tavily_key: str, synth_model: str, critic_model: str) -> None:
    client = OpenAI(api_key=openai_key)
    cache = IntelCache()
    st.session_state.cost_usd = 0.0
    with st.status("Running Market Intel Agent...", expanded=True) as status:
        st.write("1/4 Parsing research brief into plan...")
        plan, usage = build_plan(client, brief, model="gpt-4o-mini")
        update_cost("gpt-4o-mini", usage)

        st.write("2/4 Gathering company intel in parallel...")
        done: list[str] = []
        prog = st.empty()

        def on_company(name: str) -> None:
            done.append(name)
            prog.info(f"Gathered: {', '.join(done)}")

        intel, usages = gather_company_intel(client, cache, tavily_key, plan, status_cb=on_company)
        for u in usages:
            update_cost("gpt-4o-mini", u)

        st.write("3/4 Synthesizing executive memo...")
        memo, usage = build_exec_memo(client, plan, intel, model=synth_model)
        update_cost(synth_model, usage)

        st.write("4/4 Red-teaming memo critique...")
        critique, usage = critique_memo(client, plan, memo.full_markdown, memo.sources, model=critic_model)
        update_cost(critic_model, usage)

        status.update(label="Run complete", state="complete")

    st.session_state.result = {
        "plan": plan.model_dump(),
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
        demo_mode = st.toggle("Demo Mode", value=True)
        synth_model = st.selectbox("Synthesis model", ["gpt-4o-mini", "gpt-4o"], index=1)
        critic_model = st.selectbox("Critic model", ["gpt-4o-mini", "gpt-4o"], index=1)
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

    brief = st.text_area("Research Brief", value=DEFAULT_BRIEF, height=180)
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
                run_live(brief, openai_key, tavily_key, synth_model, critic_model)

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

    tabs = st.tabs(["📋 Memo", "🛡️ Critique", "🧭 Research Plan", "🔗 Sources", "🔍 Raw Intel"])

    with tabs[0]:
        render_report_header(plan, critique)
        st.markdown(render_badge(critique.overall_verdict), unsafe_allow_html=True)
        memo_with_risks = (
            f"{memo_text}\n\n## Confidence & Risks\n"
            f"- Confidence score: **{critique.confidence_score}/100**\n"
            + "\n".join(f"- Risk: {r}" for r in critique.top_3_risks_to_recipient)
        )
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.markdown(memo_with_risks)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button("Export as .md", data=memo_with_risks, file_name="market_intel_memo.md")
        if "memo_v1" in result and result.get("memo_v1") != result.get("memo_v2"):
            with st.expander("View v1 memo"):
                st.markdown(result["memo_v1"])

    with tabs[1]:
        st.markdown(render_badge(critique.overall_verdict), unsafe_allow_html=True)
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.subheader(f"Confidence Score: {critique.confidence_score}/100")
        st.caption(critique.confidence_rationale)
        st.markdown("### Top 3 Risks to Recipient")
        for risk in critique.top_3_risks_to_recipient:
            st.write(f"- {risk}")
        st.markdown("</div>", unsafe_allow_html=True)
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
            st.info("No critique findings returned.")
        st.markdown("### Strongest Aspects")
        for item in critique.strongest_aspects:
            st.write(f"- {item}")
        st.markdown("### Open Questions")
        for item in critique.open_questions_for_followup:
            st.write(f"- {item}")

        revise_disabled = demo_mode or not openai_key
        if st.button("🔄 Revise memo with this feedback", disabled=revise_disabled):
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

    with tabs[2]:
        st.json(
            {
                "sector": plan.sector,
                "companies": plan.companies,
                "inferred_companies": plan.inferred_companies,
                "key_questions": [q.model_dump() for q in plan.key_questions],
                "decision_context": plan.decision_context,
                "decision_context_detail": plan.decision_context_detail,
                "confidence": plan.confidence,
            }
        )

    with tabs[3]:
        for idx, src in enumerate(memo.sources, start=1):
            st.markdown(f"{idx}. [{src.get('title', src.get('url', 'Source'))}]({src.get('url', '#')})")

    with tabs[4]:
        for company, intel in result["intel"].items():
            with st.expander(company):
                st.json(intel)

    st.markdown(
        "<div class='footnote'>Built by Your Name · github.com/your-user/market-intel-agent</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

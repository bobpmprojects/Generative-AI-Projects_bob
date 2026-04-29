"""Executive memo synthesis with strict citation policy."""

from __future__ import annotations

import json
from datetime import date

from openai import OpenAI

from .schemas import CompanyIntel, ExecMemo, MarketContextReport, ResearchPlan


def build_exec_memo(
    client: OpenAI,
    plan: ResearchPlan,
    intel_by_company: dict[str, CompanyIntel],
    market_context: MarketContextReport | None = None,
    model: str = "gpt-4o",
) -> tuple[ExecMemo, dict]:
    prompt = """You are a senior partner at a top-tier strategy firm writing a premium executive intelligence report.
MANDATORY RULES:
1) Every factual claim in full_markdown, analyst_reports_investor_analysis, growth_revenue_projections_from_news,
   and each company_deep_summaries[].summary_markdown must use numbered citations like [1] where a fact is asserted.
2) Do not make uncited factual claims in any narrative field.
3) Use only provided intel, plan, and market_context evidence; synthesize across the full payload.
4) Explicitly answer each plan.key_questions item (in full_markdown and/or key_movements / competitive_dynamics).
5) Use polished, board-ready titles and analytical language suitable for a board packet.

STRUCTURED FIELDS (required, in addition to full_markdown):
- analyst_reports_investor_analysis: Multi-paragraph section synthesizing sell-side / industry analyst themes,
  funding rounds, valuation commentary, M&A or strategic alternatives, IPO or liquidity paths, and capital-markets
  tone, grounded in investor_signals and news items (and market_context where relevant). If evidence is thin,
  say so and cite what exists.
- growth_revenue_projections_from_news: Multi-paragraph section on growth rates, revenue impact, scale signals,
  and forward-looking management or press statements ONLY when present in recent_news or market_context.
  Distinguish reported results vs guidance vs analyst estimates. If projections are not in the evidence, state that clearly.
- company_deep_summaries: EXACTLY one entry per company key in intel (same names as intel object keys), in a stable order
  (match plan.companies order for those names, then any remaining intel keys). Each summary_markdown MUST be four to five
  paragraphs integrating: positioning, recent news themes, investor_signals, social_sentiment and customer_review_signals
  (if any), overall_momentum, and strategic implications vs peers. Do not omit a company present in intel.

full_markdown must be a comprehensive, standalone document including at minimum these markdown sections in order:
   - Executive Bottom Line
   - Decision Context and Research Scope
   - Market Size, Spend, and Growth (TAM / CAGR when evidenced; use market_context)
   - Market Structure and Competitive Archetypes
   - Key Movements Since Lookback Window
   - Analyst Reports & Investor Analysis (substantive; may align with analyst_reports_investor_analysis)
   - Growth Rates, Revenue Impact & Forward-Looking Signals (substantive; may align with growth_revenue_projections_from_news)
   - Company Deep Dives (for each company in intel: a ### heading then four to five paragraphs; may align with company_deep_summaries)
   - Investor Sentiment and IPO Readiness (concise executive distillation also reflected in investor_sentiment_read)
   - Social and Customer Sentiment (also reflected in social_customer_sentiment_read)
   - Strategic Risks and Watch Items
   - Recommended Actions
   - Comparison Table (differentiation, enterprise traction proxy, investor signal, social/customer sentiment, risk, action)
   - Key Questions Answered
   - Sources (numbered, markdown hyperlinks when URLs exist)

OTHER FIELDS:
- bottom_line, key_movements, competitive_dynamics, investor_sentiment_read, social_customer_sentiment_read,
  risks_watch_items, recommended_actions: executive distillations consistent with the longer sections.

If evidence is weak anywhere, say so directly; do not invent figures or analyst names not supported by sources.
When market_context is provided, ground TAM/growth statements there and reflect methodology_caveats.
"""
    payload = {
        "date": str(date.today()),
        "plan": plan.model_dump(),
        "intel": {k: v.model_dump() for k, v in intel_by_company.items()},
        "market_context": market_context.model_dump() if market_context else None,
    }
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload)},
    ]
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ExecMemo,
        )
    except Exception:
        if model == "gpt-4o":
            raise
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=ExecMemo,
        )
    usage = completion.usage.model_dump() if completion.usage else {}
    return completion.choices[0].message.parsed, usage


def revise_memo(
    client: OpenAI,
    memo_markdown: str,
    critique_json: dict,
    model: str = "gpt-4o",
) -> tuple[str, dict]:
    prompt = """Revise this memo to address all high/medium critique findings.
Keep citations valid and preserve numbered bibliography mapping.
Add a final section called Revision Notes summarizing fixes.
Never remove citation markers from factual claims.
"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"memo": memo_markdown, "critique": critique_json})},
    ]
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
    except Exception:
        if model == "gpt-4o":
            raise
        completion = client.chat.completions.create(model="gpt-4o", messages=messages)
    usage = completion.usage.model_dump() if completion.usage else {}
    return completion.choices[0].message.content or memo_markdown, usage

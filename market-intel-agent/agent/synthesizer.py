"""Executive memo synthesis with strict citation policy."""

from __future__ import annotations

import json
from datetime import date

from openai import OpenAI

from .schemas import CompanyIntel, ExecMemo, ResearchPlan


def build_exec_memo(
    client: OpenAI,
    plan: ResearchPlan,
    intel_by_company: dict[str, CompanyIntel],
    model: str = "gpt-4o",
) -> tuple[ExecMemo, dict]:
    prompt = """You are a senior partner at a top-tier strategy firm writing a premium executive intelligence report.
MANDATORY RULES:
1) Every factual claim in full_markdown must end with a numbered citation like [1].
2) Do not make any uncited factual claim.
3) Use only provided sources.
4) Explicitly answer each plan.key_questions item.
5) Use polished, board-ready titles and concise analytical language.
6) Include these sections in full_markdown:
   - Executive Bottom Line
   - Decision Context and Research Scope
   - Market Structure and Competitive Archetypes
   - Key Movements Since Lookback Window
   - Company-by-Company Strategic Read
   - Investor Sentiment and IPO Readiness
   - Social and Customer Sentiment
   - Strategic Risks and Watch Items
   - Recommended Actions
   - Comparison Table
   - Key Questions Answered
   - Sources
7) The comparison table must include differentiation, enterprise traction proxy, investor signal,
   social/customer sentiment, risk, and action implication.
8) Sources must be numbered and include markdown hyperlinks when URLs exist.
9) If evidence is weak, say so directly; do not fill gaps with assumptions.
"""
    payload = {
        "date": str(date.today()),
        "plan": plan.model_dump(),
        "intel": {k: v.model_dump() for k, v in intel_by_company.items()},
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

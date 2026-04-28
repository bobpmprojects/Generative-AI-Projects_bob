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
    prompt = """You are a strategy chief-of-staff writing an executive memo.
MANDATORY RULES:
1) Every factual claim in full_markdown must end with a numbered citation like [1].
2) Do not make any uncited factual claim.
3) Use only provided sources.
4) Explicitly answer each plan.key_questions item.
5) Include sections: Bottom Line, Key Movements, Competitive Dynamics, Investor Sentiment Read,
   Confidence & Risks, Recommended Actions, Comparison Table, Sources.
6) In Sources section, include numbered bibliography matching citations.
"""
    payload = {
        "date": str(date.today()),
        "plan": plan.model_dump(),
        "intel": {k: v.model_dump() for k, v in intel_by_company.items()},
    }
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
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
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"memo": memo_markdown, "critique": critique_json})},
        ],
    )
    usage = completion.usage.model_dump() if completion.usage else {}
    return completion.choices[0].message.content or memo_markdown, usage

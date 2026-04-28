"""Red-team critique of memo quality, alignment, and risk."""

from __future__ import annotations

import json

from openai import OpenAI

from .schemas import CritiqueReport, ResearchPlan


def critique_memo(
    client: OpenAI,
    plan: ResearchPlan,
    memo_markdown: str,
    sources: list[dict],
    model: str = "gpt-4o",
) -> tuple[CritiqueReport, dict]:
    prompt = """You are a strict red-team reviewer.
Assess exactly these dimensions:
source_quality, recency, logical_leap, selection_bias, hallucination, strategic_blind_spot, actionability, brief_alignment.
Hard fail citation discipline: if factual claims lack [N], create high severity findings.
Return at least 3 findings with concrete evidence and fixes.
Set overall_verdict to ship/revise/reject.
"""
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"plan": plan.model_dump(), "memo_markdown": memo_markdown, "sources": sources}
                ),
            },
        ],
        response_format=CritiqueReport,
    )
    usage = completion.usage.model_dump() if completion.usage else {}
    return completion.choices[0].message.parsed, usage

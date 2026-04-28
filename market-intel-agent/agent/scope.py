"""Brief parsing into a structured research plan."""

from __future__ import annotations

from openai import OpenAI

from .schemas import ResearchPlan


def build_plan(client: OpenAI, brief: str, model: str = "gpt-4o-mini") -> tuple[ResearchPlan, dict]:
    prompt = """You are a market intelligence scoping analyst.
Parse the user's brief into a practical research plan.
Infer obvious competitor companies if they are strongly implied by the brief.
Keep clarification_needed short and only include blockers.
"""
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": brief},
        ],
        response_format=ResearchPlan,
    )
    usage = completion.usage.model_dump() if completion.usage else {}
    return completion.choices[0].message.parsed, usage

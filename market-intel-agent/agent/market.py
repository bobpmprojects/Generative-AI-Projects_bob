"""Sector-level market context: TAM, growth, and analyst-style sizing quotes from web evidence."""

from __future__ import annotations

from openai import OpenAI

from .cache import IntelCache
from .schemas import MarketContextReport, ResearchPlan
from .search import tavily_search


def get_market_context(
    client: OpenAI,
    cache: IntelCache,
    tavily_key: str,
    plan: ResearchPlan,
    model: str = "gpt-4o-mini",
) -> tuple[MarketContextReport, dict]:
    query = (
        f'{plan.sector} market size TAM CAGR growth forecast {plan.geographic_scope} '
        f"enterprise adoption analyst report"
    )
    cached = cache.get_ttl("market", plan.sector, query, ttl_hours=24)
    if cached:
        return MarketContextReport.model_validate(cached), {"cached": True}
    results = tavily_search(tavily_key, query, max_results=8)
    evidence = "\n".join([f"- {r.title} | {r.url} | {r.content}" for r in results])
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract market sizing and growth evidence. Quote ranges only if explicit in evidence. "
                    "If not explicit, say unknown and explain what is missing. "
                    "Return supporting_sources as distinct URLs from evidence."
                ),
            },
            {"role": "user", "content": f"Sector: {plan.sector}\nScope: {plan.geographic_scope}\nEvidence:\n{evidence}"},
        ],
        response_format=MarketContextReport,
    )
    report = completion.choices[0].message.parsed
    cache.set_ttl("market", plan.sector, query, report.model_dump(), ttl_hours=24)
    usage = completion.usage.model_dump() if completion.usage else {}
    return report, usage

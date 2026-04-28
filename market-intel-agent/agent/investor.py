"""Investor signal extraction from market and fundraising coverage."""

from __future__ import annotations

from openai import OpenAI

from .cache import IntelCache
from .schemas import InvestorSignal, InvestorSignalList
from .search import tavily_search


def get_investor_signals(
    client: OpenAI, cache: IntelCache, tavily_key: str, company: str, model: str = "gpt-4o-mini"
) -> tuple[list[InvestorSignal], dict]:
    query = f"{company} valuation funding investor sentiment IPO rumors"
    cached = cache.get_ttl("investor", company, query, ttl_hours=24)
    if cached:
        return [InvestorSignal.model_validate(item) for item in cached], {"cached": True}
    results = tavily_search(tavily_key, query, max_results=6)
    evidence = "\n".join([f"- {r.title} | {r.url} | {r.content}" for r in results])
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract investor-relevant signals only. Prefer confirmed over speculative statements.",
            },
            {"role": "user", "content": f"Company: {company}\nEvidence:\n{evidence}"},
        ],
        response_format=InvestorSignalList,
    )
    items = completion.choices[0].message.parsed.items
    cache.set_ttl("investor", company, query, [s.model_dump() for s in items], ttl_hours=24)
    usage = completion.usage.model_dump() if completion.usage else {}
    return items, usage

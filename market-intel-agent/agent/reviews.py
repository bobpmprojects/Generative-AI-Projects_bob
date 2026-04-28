"""Customer review and buyer feedback signal extraction."""

from __future__ import annotations

from openai import OpenAI

from .cache import IntelCache
from .schemas import CustomerReviewSignal, CustomerReviewSignalList
from .search import tavily_search


def get_customer_reviews(
    client: OpenAI, cache: IntelCache, tavily_key: str, company: str, model: str = "gpt-4o-mini"
) -> tuple[list[CustomerReviewSignal], dict]:
    query = f'{company} customer reviews G2 Gartner Peer Insights case study complaints "review"'
    cached = cache.get_ttl("reviews", company, query, ttl_hours=24)
    if cached:
        return [CustomerReviewSignal.model_validate(item) for item in cached], {"cached": True}
    results = tavily_search(tavily_key, query, max_results=6)
    evidence = "\n".join([f"- {r.title} | {r.url} | {r.content}" for r in results])
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract buyer/customer review signals only. Distinguish reviews from vendor marketing.",
            },
            {"role": "user", "content": f"Company: {company}\nEvidence:\n{evidence}"},
        ],
        response_format=CustomerReviewSignalList,
    )
    items = completion.choices[0].message.parsed.items
    cache.set_ttl("reviews", company, query, [r.model_dump() for r in items], ttl_hours=24)
    usage = completion.usage.model_dump() if completion.usage else {}
    return items, usage

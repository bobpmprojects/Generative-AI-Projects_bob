"""Recent news retrieval and sentiment labeling for a company."""

from __future__ import annotations

from openai import OpenAI

from .cache import IntelCache
from .schemas import NewsItem, NewsList
from .search import tavily_search


def get_recent_news(
    client: OpenAI,
    cache: IntelCache,
    tavily_key: str,
    company: str,
    lookback_days: int,
    model: str = "gpt-4o-mini",
) -> tuple[list[NewsItem], dict]:
    query = (
        f"{company} news last {lookback_days} days revenue growth earnings guidance "
        f"forecast launches funding partnerships incidents"
    )
    cached = cache.get_ttl("news", company, query, ttl_hours=24)
    if cached:
        return [NewsItem.model_validate(item) for item in cached], {"cached": True}
    results = tavily_search(tavily_key, query, max_results=8)
    evidence = "\n".join([f"- {r.title} | {r.published_date} | {r.url} | {r.content}" for r in results])
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert search evidence to concise factual NewsItem records. "
                    "Use signal_type values like growth, revenue, funding, product, partnership, risk, regulatory, other when fit."
                ),
            },
            {"role": "user", "content": f"Company: {company}\nEvidence:\n{evidence}"},
        ],
        response_format=NewsList,
    )
    items = completion.choices[0].message.parsed.items
    cache.set_ttl("news", company, query, [n.model_dump() for n in items], ttl_hours=24)
    usage = completion.usage.model_dump() if completion.usage else {}
    return items, usage

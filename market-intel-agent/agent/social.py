"""Social sentiment extraction from web-indexed community and social evidence."""

from __future__ import annotations

from openai import OpenAI

from .cache import IntelCache
from .schemas import SocialSentiment, SocialSentimentList
from .search import tavily_search


def get_social_sentiment(
    client: OpenAI, cache: IntelCache, tavily_key: str, company: str, model: str = "gpt-4o-mini"
) -> tuple[list[SocialSentiment], dict]:
    query = f'{company} social sentiment developer community reddit hacker news twitter linkedin "customers"'
    cached = cache.get_ttl("social", company, query, ttl_hours=24)
    if cached:
        return [SocialSentiment.model_validate(item) for item in cached], {"cached": True}
    results = tavily_search(tavily_key, query, max_results=6)
    evidence = "\n".join([f"- {r.title} | {r.url} | {r.content}" for r in results])
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract cautious social/community sentiment signals. Do not infer volume beyond evidence.",
            },
            {"role": "user", "content": f"Company: {company}\nEvidence:\n{evidence}"},
        ],
        response_format=SocialSentimentList,
    )
    items = completion.choices[0].message.parsed.items
    cache.set_ttl("social", company, query, [s.model_dump() for s in items], ttl_hours=24)
    usage = completion.usage.model_dump() if completion.usage else {}
    return items, usage

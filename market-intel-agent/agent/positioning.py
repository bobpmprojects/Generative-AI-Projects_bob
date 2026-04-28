"""Company positioning extraction from website/search evidence."""

from __future__ import annotations

from openai import OpenAI

from .cache import IntelCache
from .schemas import Positioning
from .search import extract_webpage_text, tavily_search


def get_positioning(
    client: OpenAI, cache: IntelCache, tavily_key: str, company: str, model: str = "gpt-4o-mini"
) -> tuple[Positioning, dict]:
    cached = cache.get_positioning(company)
    if cached:
        return cached, {"cached": True}
    results = tavily_search(tavily_key, f"{company} official website platform overview", max_results=4)
    source_url = results[0].url if results else ""
    webpage = extract_webpage_text(source_url) if source_url else ""
    evidence = "\n".join([f"- {r.title} | {r.url} | {r.content}" for r in results[:3]])
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract precise B2B company positioning. Use source_url from evidence only.",
            },
            {
                "role": "user",
                "content": f"Company: {company}\nSource URL: {source_url}\nEvidence:\n{evidence}\nWebpage text:\n{webpage}",
            },
        ],
        response_format=Positioning,
    )
    item = completion.choices[0].message.parsed
    if not item.source_url:
        item.source_url = source_url
    cache.set_positioning(item)
    usage = completion.usage.model_dump() if completion.usage else {}
    return item, usage

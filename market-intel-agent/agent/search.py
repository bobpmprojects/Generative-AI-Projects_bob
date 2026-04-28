"""Search and webpage utility helpers for company intelligence collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    published_date: str


def truncate_text(text: str, max_chars: int = 1500) -> str:
    clean = " ".join(text.split())
    return clean[:max_chars]


def tavily_search(api_key: str, query: str, max_results: int = 6) -> list[SearchResult]:
    client = TavilyClient(api_key=api_key)
    payload: dict[str, Any] = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False,
    )
    results: list[SearchResult] = []
    for row in payload.get("results", []):
        results.append(
            SearchResult(
                title=row.get("title", "Untitled"),
                url=row.get("url", ""),
                content=truncate_text(row.get("content", "")),
                published_date=row.get("published_date", "unknown"),
            )
        )
    return results


def extract_webpage_text(url: str, timeout_sec: int = 8) -> str:
    try:
        response = requests.get(url, timeout=timeout_sec, headers={"User-Agent": "MarketIntelAgent/1.0"})
        response.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return truncate_text(soup.get_text(separator=" ", strip=True))

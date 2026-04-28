"""Parallel company intel gathering orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .cache import IntelCache
from .investor import get_investor_signals
from .news import get_recent_news
from .positioning import get_positioning
from .schemas import CompanyIntel, ResearchPlan


def _momentum(news_positive: int, news_negative: int) -> tuple[str, str]:
    if news_positive >= news_negative + 2:
        return "accelerating", "Recent coverage skewed positive with multiple growth signals."
    if news_negative >= news_positive + 2:
        return "declining", "Recent coverage includes materially more negative than positive signals."
    if news_positive or news_negative:
        return "steady", "Signals are mixed; no decisive directional trend."
    return "unclear", "Insufficient recency-confirmed evidence."


def gather_company_intel(
    client: OpenAI,
    cache: IntelCache,
    tavily_key: str,
    plan: ResearchPlan,
    status_cb: callable | None = None,
) -> tuple[dict[str, CompanyIntel], list[dict]]:
    usages: list[dict] = []
    companies = plan.companies
    output: dict[str, CompanyIntel] = {}

    def run_for_company(company: str) -> tuple[str, CompanyIntel, list[dict]]:
        local_usage: list[dict] = []
        pos, u1 = get_positioning(client, cache, tavily_key, company)
        news, u2 = get_recent_news(client, cache, tavily_key, company, plan.lookback_days)
        inv, u3 = get_investor_signals(client, cache, tavily_key, company)
        local_usage.extend([u1, u2, u3])
        pos_count = sum(1 for n in news if n.sentiment == "positive")
        neg_count = sum(1 for n in news if n.sentiment == "negative")
        momentum, rationale = _momentum(pos_count, neg_count)
        intel = CompanyIntel(
            positioning=pos,
            recent_news=news,
            investor_signals=inv,
            overall_momentum=momentum,
            momentum_rationale=rationale,
        )
        return company, intel, local_usage

    with ThreadPoolExecutor(max_workers=min(4, len(companies))) as executor:
        futures = {executor.submit(run_for_company, c): c for c in companies}
        for future in as_completed(futures):
            company, intel, local = future.result()
            output[company] = intel
            usages.extend(local)
            if status_cb:
                status_cb(company)
    return output, usages

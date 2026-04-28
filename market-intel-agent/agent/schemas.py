"""Pydantic schemas for planning, intel, memo synthesis, and critique."""

from typing import Literal

from pydantic import BaseModel


class ScopeQuestion(BaseModel):
    question: str
    rationale: str


class ResearchPlan(BaseModel):
    sector: str
    companies: list[str]
    inferred_companies: list[str]
    key_questions: list[ScopeQuestion]
    geographic_scope: str
    lookback_days: int
    decision_context: Literal[
        "evaluating_partnership",
        "investment_thesis",
        "career_move",
        "competitive_strategy",
        "general_market_awareness",
        "other",
    ]
    decision_context_detail: str
    confidence: Literal["high", "medium", "low"]
    clarification_needed: list[str]
    plan_summary: str


class Positioning(BaseModel):
    company_name: str
    one_line_positioning: str
    target_customer: str
    key_differentiators: list[str]
    pricing_model: str
    supported_models: list[str]
    source_url: str


class NewsItem(BaseModel):
    title: str
    url: str
    published_date: str
    summary: str
    sentiment: Literal["positive", "neutral", "negative"]
    signal_type: Literal[
        "product_launch",
        "funding",
        "partnership",
        "leadership",
        "controversy",
        "performance_claim",
        "other",
    ]


class NewsList(BaseModel):
    items: list[NewsItem]


class InvestorSignal(BaseModel):
    signal: str
    url: str
    confidence: Literal["confirmed", "rumored", "speculative"]


class InvestorSignalList(BaseModel):
    items: list[InvestorSignal]


class CompanyIntel(BaseModel):
    positioning: Positioning
    recent_news: list[NewsItem]
    investor_signals: list[InvestorSignal]
    overall_momentum: Literal["accelerating", "steady", "declining", "unclear"]
    momentum_rationale: str


class ExecMemo(BaseModel):
    sector: str
    date: str
    bottom_line: str
    key_movements: list[str]
    competitive_dynamics: str
    investor_sentiment_read: str
    risks_watch_items: list[str]
    recommended_actions: list[str]
    sources: list[dict]
    full_markdown: str


class CritiqueFinding(BaseModel):
    dimension: Literal[
        "source_quality",
        "recency",
        "logical_leap",
        "selection_bias",
        "hallucination",
        "strategic_blind_spot",
        "actionability",
        "brief_alignment",
    ]
    severity: Literal["low", "medium", "high"]
    location: str
    finding: str
    evidence: str
    recommended_fix: str


class CritiqueReport(BaseModel):
    overall_verdict: Literal["ship", "revise", "reject"]
    confidence_score: int
    confidence_rationale: str
    findings: list[CritiqueFinding]
    strongest_aspects: list[str]
    top_3_risks_to_recipient: list[str]
    open_questions_for_followup: list[str]

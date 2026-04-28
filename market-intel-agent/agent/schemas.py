"""Pydantic schemas for planning, intel, memo synthesis, and critique."""

from typing import Literal

from pydantic import BaseModel, Field


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


class SocialSentiment(BaseModel):
    channel: str
    audience: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    summary: str
    evidence_url: str
    confidence: Literal["high", "medium", "low"]


class SocialSentimentList(BaseModel):
    items: list[SocialSentiment]


class CustomerReviewSignal(BaseModel):
    source: str
    segment: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    themes: list[str]
    summary: str
    evidence_url: str
    confidence: Literal["high", "medium", "low"]


class CustomerReviewSignalList(BaseModel):
    items: list[CustomerReviewSignal]


class CompanyIntel(BaseModel):
    positioning: Positioning
    recent_news: list[NewsItem]
    investor_signals: list[InvestorSignal]
    social_sentiment: list[SocialSentiment] = Field(default_factory=list)
    customer_review_signals: list[CustomerReviewSignal] = Field(default_factory=list)
    overall_momentum: Literal["accelerating", "steady", "declining", "unclear"]
    momentum_rationale: str


class Source(BaseModel):
    title: str
    url: str
    publisher: str = ""
    note: str = ""


class ExecMemo(BaseModel):
    sector: str
    date: str
    bottom_line: str
    key_movements: list[str]
    competitive_dynamics: str
    investor_sentiment_read: str
    social_customer_sentiment_read: str = ""
    risks_watch_items: list[str]
    recommended_actions: list[str]
    sources: list[Source]
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

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
    company_name: str = ""
    one_line_positioning: str = ""
    target_customer: str = ""
    key_differentiators: list[str] = Field(default_factory=list)
    pricing_model: str = ""
    supported_models: list[str] = Field(default_factory=list)
    source_url: str = ""


class NewsItem(BaseModel):
    title: str = ""
    url: str = ""
    published_date: str = ""
    summary: str = ""
    sentiment: str = "neutral"
    signal_type: str = "other"


class NewsList(BaseModel):
    items: list[NewsItem]


class InvestorSignal(BaseModel):
    signal: str = ""
    url: str = ""
    confidence: str = "speculative"


class InvestorSignalList(BaseModel):
    items: list[InvestorSignal]


class SocialSentiment(BaseModel):
    channel: str = ""
    audience: str = ""
    sentiment: str = "mixed"
    summary: str = ""
    evidence_url: str = ""
    confidence: str = "medium"


class SocialSentimentList(BaseModel):
    items: list[SocialSentiment]


class CustomerReviewSignal(BaseModel):
    source: str = ""
    segment: str = ""
    sentiment: str = "mixed"
    themes: list[str] = Field(default_factory=list)
    summary: str = ""
    evidence_url: str = ""
    confidence: str = "medium"


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
    sector: str = ""
    date: str = ""
    bottom_line: str = ""
    key_movements: list[str] = Field(default_factory=list)
    competitive_dynamics: str = ""
    investor_sentiment_read: str = ""
    social_customer_sentiment_read: str = ""
    risks_watch_items: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    full_markdown: str = ""


class CritiqueFinding(BaseModel):
    dimension: str = "brief_alignment"
    severity: str = "medium"
    location: str = ""
    finding: str = ""
    evidence: str = ""
    recommended_fix: str = ""


class CritiqueReport(BaseModel):
    overall_verdict: str = "revise"
    confidence_score: int = 70
    confidence_rationale: str = ""
    findings: list[CritiqueFinding] = Field(default_factory=list)
    strongest_aspects: list[str] = Field(default_factory=list)
    top_3_risks_to_recipient: list[str] = Field(default_factory=list)
    open_questions_for_followup: list[str] = Field(default_factory=list)

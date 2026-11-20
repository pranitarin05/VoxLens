"""Pydantic v2 request/response models (PRD section 9)."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field

ModelName = Literal["tfidf", "distilbert", "bert"]
Sentiment = Literal["negative", "neutral", "positive"]
Granularity = Literal["day", "week", "month"]
Text = Annotated[str, Field(min_length=1, max_length=5000)]


class AspectMentionOut(BaseModel):
    """One aspect mention with sentence-level sentiment and surface span."""

    aspect: str
    sentiment: Sentiment
    span: str


class PredictRequest(BaseModel):
    """One review; empty or >5000 characters is rejected with 422."""

    text: Text
    model: ModelName | None = Field(default=None, description="Defaults to the production model.")


class PredictResponse(BaseModel):
    """Prediction result shared by single and batch endpoints."""

    label: Sentiment
    probabilities: dict[str, float]
    aspects: list[AspectMentionOut]
    model_name: str
    model_version: str
    latency_ms: float = Field(description="Server-side scoring time in milliseconds.")


class BatchPredictRequest(BaseModel):
    """Up to 64 reviews; each obeys the 1-5000 character rule."""

    texts: Annotated[list[Text], Field(min_length=1, max_length=64)]
    model: ModelName | None = None


class HealthResponse(BaseModel):
    """Liveness plus the loaded production model and database reachability."""

    status: Literal["ok", "degraded"]
    model_name: str
    model_version: str
    database: bool


class ModelInfo(BaseModel):
    """A model available for serving."""

    name: str
    version: str


class ModelsResponse(BaseModel):
    """All loaded models; `production` is the default for /v1/predict."""

    production: str
    models: list[ModelInfo]


class TrendPoint(BaseModel):
    """Counts for one (time bucket, label) pair."""

    bucket: str
    label: Sentiment
    count: int


class TrendResponse(BaseModel):
    """Sentiment over time; `points` is long-format for easy charting."""

    granularity: Granularity
    points: list[TrendPoint]


class AspectCount(BaseModel):
    """Aspect frequency, optionally filtered by sentiment/category."""

    aspect: str
    count: int
    share: float


class TopAspectsResponse(BaseModel):
    """Most frequent aspects for the given filters."""

    aspects: list[AspectCount]


class AskFilters(BaseModel):
    """Optional structured filters for retrieval and whitelisted SQL."""

    category: Annotated[str | None, Field(max_length=64)] = None
    product_id: Annotated[str | None, Field(max_length=64)] = None
    sentiment: Sentiment | None = None
    start: date | None = None
    end: date | None = None


class AskRequest(BaseModel):
    """Natural-language question with optional filters (PRD section 9)."""

    question: Annotated[str, Field(min_length=1, max_length=500)]
    filters: AskFilters | None = None


class SourceOut(BaseModel):
    """A review used as evidence: id, snippet and cosine similarity."""

    review_id: str
    snippet: str
    score: float
    label: Sentiment | None = None
    category: str | None = None


class AskResponse(BaseModel):
    """Grounded answer with evidence, abstention flag and route."""

    answer: str
    sources: list[SourceOut]
    abstained: bool
    route: Literal["rag", "sql"]
    latency_ms: float
    cost_usd: float = 0.0
    citations: list[int] = Field(default_factory=list)
    parse_failed: bool = False

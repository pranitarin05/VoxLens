"""FR-9: FastAPI app - models loaded once at startup, request ids, JSON logs."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.responses import JSONResponse, Response

from voc.api.deps import (
    ModelRegistry,
    Settings,
    build_aspect_extractor,
    build_rag_service,
    db_connection,
    log_requests,
    ping_database,
)
from voc.api.schemas import (
    AskRequest,
    AskResponse,
    AspectCount,
    BatchPredictRequest,
    Granularity,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    Sentiment,
    SourceOut,
    TopAspectsResponse,
    TrendPoint,
    TrendResponse,
)
from voc.db.queries import fetch_top_aspects, fetch_trend
from voc.models.llm_zero_shot import LLMError, SpendCapExceeded

LOGGER = logging.getLogger("voc.api")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including request-scoped extras."""

    extras = ("request_id", "method", "path", "status", "latency_ms", "event")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in self.extras:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Send voc.api logs to stdout as JSON (idempotent across imports)."""
    logger = logging.getLogger("voc.api")
    if any(getattr(handler, "_voc_json", False) for handler in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._voc_json = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def create_app(
    *,
    settings: Settings | None = None,
    registry_factory: Callable[[Settings], ModelRegistry] | None = None,
    aspect_factory: Callable[[ModelRegistry, Settings], Any] | None = None,
    rag_factory: Callable[[Settings], Any] | None = None,
    db_ping: Callable[[], bool] | None = None,
    request_sink: Callable[[list[tuple[str, str, float, str, int]]], bool] | None = None,
) -> FastAPI:
    """Build the API; factories are injectable so tests avoid torch/spaCy."""
    settings = settings or Settings.from_env()
    registry_factory = registry_factory or ModelRegistry.load
    aspect_factory = aspect_factory or build_aspect_extractor
    rag_factory = rag_factory or build_rag_service
    db_ping = db_ping or ping_database
    request_sink = request_sink or log_requests
    configure_logging()

    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.registry = registry_factory(settings)
        app.state.aspects = aspect_factory(app.state.registry, settings)
        app.state.rag = rag_factory(settings)
        app.state.db_ping = db_ping
        app.state.request_sink = request_sink
        LOGGER.info(
            json.dumps(
                {
                    "event": "startup",
                    "production": app.state.registry.production,
                    "models": [m["name"] for m in app.state.registry.info()],
                    "aspects": app.state.aspects is not None,
                    "rag": app.state.rag is not None,
                }
            )
        )
        yield

    app = FastAPI(title="VoxLens API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "unhandled error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "internal server error", "request_id": request_id},
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 3),
            },
        )
        return response

    def _predict(
        state: Any, texts: list[str], model_name: str | None
    ) -> tuple[list[PredictResponse], str, float]:
        registry: ModelRegistry = state.registry
        name = model_name or registry.production
        try:
            loaded = registry.get(name)
        except KeyError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"model {name!r} is not loaded; see /v1/models",
            ) from exc
        started = time.perf_counter()
        predictions = loaded.predict(texts)
        aspects = state.aspects.extract(texts) if state.aspects is not None else [[] for _ in texts]
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        items = [
            PredictResponse(
                label=label,
                probabilities=probabilities,
                aspects=aspects[index],
                model_name=loaded.name,
                model_version=loaded.version,
                latency_ms=latency_ms,
            )
            for index, (label, probabilities) in enumerate(predictions)
        ]
        return items, loaded.name, latency_ms

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness plus the loaded production model and database state."""
        registry: ModelRegistry = request.app.state.registry
        production = registry.get(registry.production)
        database = request.app.state.db_ping()
        return HealthResponse(
            status="ok" if database else "degraded",
            model_name=production.name,
            model_version=production.version,
            database=database,
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    def models(request: Request) -> ModelsResponse:
        """List loaded models and the production default."""
        registry: ModelRegistry = request.app.state.registry
        return ModelsResponse(
            production=registry.production,
            models=[ModelInfo(**item) for item in registry.info()],
        )

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        """Score one review and record the call in api_requests (best effort)."""
        items, name, latency_ms = _predict(request.app.state, [payload.text], payload.model)
        request.app.state.request_sink(
            [("/v1/predict", name, latency_ms, items[0].label, len(payload.text))]
        )
        return items[0]

    @app.post("/v1/predict/batch", response_model=list[PredictResponse])
    def predict_batch(payload: BatchPredictRequest, request: Request) -> list[PredictResponse]:
        """Score up to 64 reviews with one model (latency is the whole batch)."""
        items, name, latency_ms = _predict(request.app.state, payload.texts, payload.model)
        request.app.state.request_sink(
            [
                ("/v1/predict/batch", name, latency_ms, item.label, len(text))
                for item, text in zip(items, payload.texts, strict=False)
            ]
        )
        return items

    @app.get("/v1/stats/sentiment-trend", response_model=TrendResponse)
    def sentiment_trend(
        category: Annotated[str | None, Query(max_length=64)] = None,
        product_id: Annotated[str | None, Query(max_length=64)] = None,
        start: Annotated[date | None, Query(alias="from")] = None,
        end: Annotated[date | None, Query(alias="to")] = None,
        granularity: Annotated[Granularity, Query()] = "month",
    ) -> TrendResponse:
        """Ground-truth sentiment counts bucketed by day, week or month."""
        try:
            with db_connection() as conn:
                points = fetch_trend(
                    conn,
                    category=category,
                    product_id=product_id,
                    start=start,
                    end=end,
                    granularity=granularity,
                )
        except Exception as exc:  # noqa: BLE001 - surface a 503, not a stack trace
            LOGGER.exception("sentiment-trend query failed")
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return TrendResponse(
            granularity=granularity, points=[TrendPoint(**point) for point in points]
        )

    @app.get("/v1/stats/top-aspects", response_model=TopAspectsResponse)
    def top_aspects(
        sentiment: Annotated[Sentiment | None, Query()] = None,
        category: Annotated[str | None, Query(max_length=64)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> TopAspectsResponse:
        """Most frequent aspects, optionally filtered by sentiment or category."""
        try:
            with db_connection() as conn:
                rows = fetch_top_aspects(conn, sentiment=sentiment, category=category, limit=limit)
        except Exception as exc:  # noqa: BLE001 - surface a 503, not a stack trace
            LOGGER.exception("top-aspects query failed")
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return TopAspectsResponse(aspects=[AspectCount(**row) for row in rows])

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        """Grounded RAG/SQL answer with sources and explicit abstention."""
        service = request.app.state.rag
        if service is None:
            raise HTTPException(
                status_code=501,
                detail="RAG is disabled (API_RAG=0)",
                headers={"X-Request-ID": request.state.request_id},
            )
        filters = payload.filters.model_dump(exclude_none=True) if payload.filters else {}
        try:
            result = service.answer(payload.question, filters)
        except SpendCapExceeded as exc:
            raise HTTPException(status_code=503, detail="LLM spend cap reached") from exc
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - DB/provider outages become 503
            LOGGER.exception("ask failed")
            raise HTTPException(status_code=503, detail="answer service unavailable") from exc
        return AskResponse(
            answer=result.answer,
            sources=[SourceOut(**source) for source in result.sources],
            abstained=result.abstained,
            route=result.route,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            citations=result.citations,
            parse_failed=result.parse_failed,
        )

    return app


app = create_app()

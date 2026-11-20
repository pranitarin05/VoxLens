"""Phase 8: API contract, validation, request logging and stats SQL templates."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from voc.api.deps import Settings
from voc.api.main import create_app
from voc.db.load import MIGRATIONS_DIR, apply_migrations, database_url
from voc.db.queries import fetch_top_aspects, fetch_trend, top_aspects_query, trend_query

PROBS = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}


class FakeModel:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version

    def predict(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        return [("positive", dict(PROBS)) for _ in texts]


class FakeRegistry:
    production = "distilbert"

    def __init__(self) -> None:
        self._models = {
            "tfidf": FakeModel("tfidf", "tfidf-logreg-v1"),
            "distilbert": FakeModel("distilbert", "distilbert-60k-v1"),
            "bert": FakeModel("bert", "bert-30k-v1"),
        }

    def get(self, name: str) -> FakeModel:
        return self._models[name]

    def info(self) -> list[dict[str, str]]:
        return [{"name": name, "version": model.version} for name, model in self._models.items()]


class FakeAspects:
    def extract(self, texts: list[str]) -> list[list[dict[str, str]]]:
        return [
            [{"aspect": "price", "sentiment": "negative", "span": "too expensive"}] for _ in texts
        ]


def make_client(*, database: bool = True) -> tuple[TestClient, list[tuple]]:
    recorded: list[tuple] = []

    def sink(records: list[tuple]) -> bool:
        recorded.extend(records)
        return True

    app = create_app(
        settings=Settings(),
        registry_factory=lambda _settings: FakeRegistry(),
        aspect_factory=lambda _registry, _settings: FakeAspects(),
        db_ping=lambda: database,
        request_sink=sink,
    )
    return TestClient(app), recorded


def test_health_reports_model_and_database() -> None:
    client, _ = make_client(database=True)
    with client:
        body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model_name": "distilbert",
        "model_version": "distilbert-60k-v1",
        "database": True,
    }


def test_health_degraded_when_database_down() -> None:
    client, _ = make_client(database=False)
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] is False


def test_models_lists_loaded_models_and_production() -> None:
    client, _ = make_client()
    with client:
        body = client.get("/v1/models").json()
    assert body["production"] == "distilbert"
    assert [item["name"] for item in body["models"]] == ["tfidf", "distilbert", "bert"]
    assert body["models"][1]["version"] == "distilbert-60k-v1"


def test_predict_returns_label_probabilities_aspects_and_request_id() -> None:
    client, recorded = make_client()
    with client:
        response = client.post("/v1/predict", json={"text": "The battery is awful."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["probabilities"] == PROBS
    assert body["aspects"] == [
        {"aspect": "price", "sentiment": "negative", "span": "too expensive"}
    ]
    assert body["model_name"] == "distilbert"
    assert body["model_version"] == "distilbert-60k-v1"
    assert body["latency_ms"] >= 0
    assert response.headers["X-Request-ID"]
    assert recorded == [
        ("/v1/predict", "distilbert", body["latency_ms"], "positive", len("The battery is awful."))
    ]


def test_predict_accepts_explicit_model() -> None:
    client, recorded = make_client()
    with client:
        body = client.post("/v1/predict", json={"text": "ok", "model": "tfidf"}).json()
    assert body["model_name"] == "tfidf"
    assert recorded[0][1] == "tfidf"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "x" * 5001},
        {"text": "ok", "model": "gpt-oss"},
    ],
)
def test_predict_rejects_invalid_payload(payload: dict) -> None:
    client, _ = make_client()
    with client:
        assert client.post("/v1/predict", json=payload).status_code == 422


def test_batch_returns_one_object_per_text_and_records_each() -> None:
    client, recorded = make_client()
    with client:
        response = client.post("/v1/predict/batch", json={"texts": ["a", "b", "c"]})
    assert response.status_code == 200
    assert len(response.json()) == 3
    assert len(recorded) == 3
    assert {row[0] for row in recorded} == {"/v1/predict/batch"}


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": ["ok"] * 65},
        {"texts": [""]},
        {"texts": ["x" * 5001]},
    ],
)
def test_batch_rejects_invalid_payload(payload: dict) -> None:
    client, _ = make_client()
    with client:
        assert client.post("/v1/predict/batch", json=payload).status_code == 422


@pytest.mark.parametrize(
    "url",
    [
        "/v1/stats/sentiment-trend?granularity=year",
        "/v1/stats/sentiment-trend?from=not-a-date",
        "/v1/stats/top-aspects?limit=0",
        "/v1/stats/top-aspects?limit=101",
        "/v1/stats/top-aspects?sentiment=meh",
    ],
)
def test_stats_reject_invalid_query(url: str) -> None:
    client, _ = make_client()
    with client:
        assert client.get(url).status_code == 422


def test_ask_is_reserved_for_phase_9() -> None:
    client, _ = make_client()
    with client:
        response = client.post("/v1/ask", json={"question": "How many negative reviews?"})
    assert response.status_code == 501
    assert "Phase 9" in response.json()["detail"]


def test_trend_query_whitelists_granularity_and_parameterises_filters() -> None:
    sql, params = trend_query(
        category="Amazon_Fashion'; DROP TABLE reviews; --",
        product_id="P1",
        start="2024-01-01",
        end="2024-02-01",
        granularity="week",
    )
    assert "date_trunc(%s" in sql
    assert "DROP TABLE" not in sql
    assert params[0] == "week"
    assert params[1] == "Amazon_Fashion'; DROP TABLE reviews; --"
    assert params[2:] == ["P1", "2024-01-01", "2024-02-01"]
    with pytest.raises(KeyError):
        trend_query(granularity="year")


def test_top_aspects_query_parameterises_filters_and_limit() -> None:
    sql, params = top_aspects_query(sentiment="negative", category="All_Beauty", limit=7)
    assert "a.sentiment = %s" in sql
    assert "p.category = %s" in sql
    assert params == ["negative", "All_Beauty", 7]


@pytest.mark.integration
def test_stats_endpoints_against_postgres() -> None:
    import psycopg

    try:
        conn = psycopg.connect(database_url(), connect_timeout=2)
    except Exception as exc:  # noqa: BLE001 - integration test needs the stack
        pytest.skip(f"postgres not reachable: {exc}")
    apply_migrations(MIGRATIONS_DIR, conn=conn)
    category = "Test_Phase8"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO products (product_id, category, title) VALUES (%s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                ("phase8-product", category, "Phase 8"),
            )
            cur.execute(
                "INSERT INTO reviews (review_id, product_id, rating, label, text, clean_text, "
                "reviewed_at, split) VALUES (%s, %s, 2, 'negative', 'bad', 'bad', "
                "'2024-01-15T00:00:00Z', 'test') ON CONFLICT DO NOTHING",
                ("phase8-review", "phase8-product"),
            )
            cur.execute(
                "INSERT INTO aspects (review_id, aspect, sentiment, span_text) "
                "VALUES (%s, 'price', 'negative', 'too expensive')",
                ("phase8-review",),
            )
            points = fetch_trend(conn, category=category, granularity="month")
            rows = fetch_top_aspects(conn, category=category, limit=5)
    finally:
        conn.rollback()
        conn.close()
    assert points == [{"bucket": "2024-01-01", "label": "negative", "count": 1}]
    assert rows[0]["aspect"] == "price"
    assert rows[0]["count"] == 1
    assert rows[0]["share"] == pytest.approx(1.0)

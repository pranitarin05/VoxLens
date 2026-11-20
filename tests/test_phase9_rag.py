"""Phase 9: stratification, filters, router templates, answer parsing, /v1/ask."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from voc.api.deps import Settings
from voc.api.main import create_app
from voc.rag.answer import (
    AnswerResult,
    RAGParseError,
    build_snippets,
    parse_rag_answer,
)
from voc.rag.embed import allocate_quotas
from voc.rag.retrieve import RetrievedReview, filter_clauses
from voc.rag.router import route, run_analysis

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
        self._models = {"distilbert": FakeModel("distilbert", "distilbert-60k-v1")}

    def get(self, name: str) -> FakeModel:
        return self._models[name]

    def info(self) -> list[dict[str, str]]:
        return [{"name": "distilbert", "version": "distilbert-60k-v1"}]


class FakeAspects:
    def extract(self, texts: list[str]) -> list[list[dict[str, str]]]:
        return [[] for _ in texts]


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.sql = ""

    def execute(self, sql: str, params: list | None = None) -> None:
        self.sql = sql
        self.conn.executed.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self.conn.rows_for(self.sql)

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class FakeConn:
    """Returns canned rows based on the SQL shape, capturing parameters."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, list | None]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def rows_for(self, sql: str) -> list[tuple]:
        if "GROUP BY r.label ORDER BY r.label" in sql:
            return [("negative", 2), ("positive", 3)]
        if "GROUP BY bucket" in sql:
            return [("2024-01", "negative", 1), ("2024-01", "positive", 2)]
        if "GROUP BY a.aspect" in sql:
            return [("battery", 4)]
        if "PARTITION BY" in sql:
            return [("All_Beauty", "negative", 1, 0.25), ("All_Beauty", "positive", 3, 0.75)]
        if "ORDER BY count(*) DESC, r.product_id" in sql:
            return [("P1", "Thing", 3)]
        if "DISTINCT category" in sql:
            return [("All_Beauty",), ("Amazon_Fashion",)]
        if "DISTINCT aspect" in sql:
            return [("battery",), ("shipping",)]
        return []


def test_allocate_quotas_sums_and_caps() -> None:
    counts = {("All_Beauty", "positive"): 100, ("Amazon_Fashion", "negative"): 50}
    quotas = allocate_quotas(counts, 30)
    assert sum(quotas.values()) == 30
    assert all(quotas[key] <= counts[key] for key in counts)
    assert allocate_quotas(counts, 1000) == counts


def test_filter_clauses_parameterise_every_value() -> None:
    clauses, params = filter_clauses(
        category="All_Beauty'; DROP TABLE reviews; --",
        product_id="P1",
        sentiment="negative",
        start="2024-01-01",
        end="2024-02-01",
    )
    assert len(clauses) == 5
    assert all("%s" in clause for clause in clauses)
    assert params[0] == "All_Beauty'; DROP TABLE reviews; --"
    assert "DROP" not in " ".join(clauses)


def test_route_separates_analytical_from_qualitative() -> None:
    assert route("How many negative reviews are there?") == "sql"
    assert route("What do customers complain about?") == "rag"


def test_run_analysis_counts_with_filters() -> None:
    conn = FakeConn()
    analysis = run_analysis(
        conn,
        "How many negative reviews?",
        ["All_Beauty"],
        ["battery"],
        {"category": "All_Beauty", "sentiment": "negative"},
    )
    assert analysis is not None
    assert analysis.intent == "counts"
    assert analysis.facts["total"] == 5
    sql, params = conn.executed[-1]
    assert "All_Beauty" not in sql
    assert params == ["negative", "All_Beauty"]


def test_run_analysis_trend_compare_and_products() -> None:
    conn = FakeConn()
    trend = run_analysis(conn, "Show the sentiment trend by month", [], [])
    compare = run_analysis(conn, "Compare categories", [], [])
    products = run_analysis(conn, "Which product has the most negative reviews?", [], [])
    assert trend is not None and trend.intent == "trend"
    assert compare is not None and compare.intent == "compare"
    assert products is not None and products.intent == "top_products"


def test_run_analysis_returns_none_for_qualitative_question() -> None:
    conn = FakeConn()
    assert run_analysis(conn, "Why did people enjoy it?", [], ["battery"]) is None


def test_snippets_are_numbered_and_truncated() -> None:
    doc = RetrievedReview(
        review_id="r1",
        product_id="p1",
        category="All_Beauty",
        label="positive",
        text="x" * 500,
        reviewed_at=None,
        score=0.9,
    )
    snippets = build_snippets([doc], max_chars=100)
    assert snippets.startswith("[1] review_id=r1")
    assert len(snippets) < 400


def test_parse_rag_answer_accepts_fenced_json() -> None:
    raw = '```json\n{"answer": "Battery is bad.", "citations": [2, 1], "abstained": false}\n```'
    parsed = parse_rag_answer(raw, n_snippets=3)
    assert parsed.answer == "Battery is bad."
    assert parsed.citations == [1, 2]
    assert parsed.abstained is False


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "No citations.", "citations": [], "abstained": False},
        {"answer": "Out of range.", "citations": [4], "abstained": False},
        {"answer": "", "citations": [1], "abstained": False},
    ],
)
def test_parse_rag_answer_rejects_invalid_contract(payload: dict) -> None:
    with pytest.raises(RAGParseError):
        parse_rag_answer(json.dumps(payload), n_snippets=3)


def test_parse_rag_answer_allows_abstention_without_citations() -> None:
    parsed = parse_rag_answer(
        '{"answer": "Not enough evidence.", "citations": [], "abstained": true}', n_snippets=3
    )
    assert parsed.abstained is True
    assert parsed.citations == []


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def answer(self, question: str, filters: dict) -> AnswerResult:
        self.calls.append((question, filters))
        return AnswerResult(
            answer="Battery is the main complaint.",
            sources=[
                {
                    "review_id": "r1",
                    "snippet": "battery died",
                    "score": 0.9,
                    "label": "negative",
                    "category": "All_Beauty",
                }
            ],
            abstained=False,
            route="rag",
            latency_ms=1.5,
            cost_usd=0.0001,
            cached=False,
            citations=[1],
        )


def make_client(rag: FakeRag | None) -> TestClient:
    app = create_app(
        settings=Settings(),
        registry_factory=lambda _settings: FakeRegistry(),
        aspect_factory=lambda _registry, _settings: FakeAspects(),
        rag_factory=lambda _settings: rag,
        db_ping=lambda: True,
        request_sink=lambda records: True,
    )
    return TestClient(app)


def test_ask_returns_contract_and_forwards_filters() -> None:
    rag = FakeRag()
    with make_client(rag) as client:
        response = client.post(
            "/v1/ask",
            json={
                "question": "What do people dislike?",
                "filters": {"category": "All_Beauty", "sentiment": "negative"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "rag"
    assert body["abstained"] is False
    assert body["sources"][0]["review_id"] == "r1"
    assert body["citations"] == [1]
    assert response.headers["X-Request-ID"]
    assert rag.calls[0][1] == {"category": "All_Beauty", "sentiment": "negative"}


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 501},
        {"question": "ok", "filters": {"sentiment": "meh"}},
        {"question": "ok", "filters": {"start": "not-a-date"}},
    ],
)
def test_ask_validates_payload(payload: dict) -> None:
    with make_client(FakeRag()) as client:
        assert client.post("/v1/ask", json=payload).status_code == 422


def test_ask_still_501_when_rag_disabled() -> None:
    with make_client(None) as client:
        response = client.post("/v1/ask", json={"question": "How many reviews?"})
    assert response.status_code == 501


@pytest.mark.integration
def test_retrieval_and_hnsw_on_real_embeddings() -> None:
    pytest.importorskip("sentence_transformers")
    import psycopg

    from voc.db.load import database_url
    from voc.rag.embed import load_embedder
    from voc.rag.retrieve import retrieve

    try:
        conn = psycopg.connect(database_url(), connect_timeout=2)
    except Exception as exc:  # noqa: BLE001 - integration test needs the stack
        pytest.skip(f"postgres not reachable: {exc}")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM review_embeddings")
            if cur.fetchone()[0] == 0:
                pytest.skip("review_embeddings empty; run make rag-embed")
            cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='review_embeddings'")
            assert "idx_review_embeddings_hnsw" in {row[0] for row in cur.fetchall()}
            cur.execute(
                "SELECT r.text FROM reviews r JOIN review_embeddings e USING (review_id) LIMIT 1"
            )
            text = cur.fetchone()[0]
        model = load_embedder("cpu")
        vector = model.encode([text], normalize_embeddings=True)[0]
        docs = retrieve(conn, vector, keep=3)
        assert len(docs) == 3
        assert all(-1.0 <= doc.score <= 1.0 for doc in docs)
    finally:
        conn.close()


def test_compare_facts_matches_only_expected_subset() -> None:
    from voc.rag.evaluate import compare_facts

    counts = {"counts": {"negative": 2}, "total": 2}
    assert compare_facts("counts", counts, {**counts, "sentiment": "negative"})
    assert not compare_facts("counts", counts, {"counts": {"negative": 3}, "total": 3})
    aspects = {"aspects": [{"aspect": "battery", "count": 4}]}
    assert compare_facts("aspects", aspects, aspects)
    assert not compare_facts("aspects", aspects, {"aspects": []})
    products = {"products": [{"product_id": "P1", "title": "T", "negative_count": 3}]}
    assert compare_facts("top_products", products, products)
    trend = {"points": 1, "first_month": "2024-01", "last_month": "2024-01", "latest": []}
    assert compare_facts("trend", trend, trend)
    rows = {"rows": [{"category": "C", "label": "positive", "count": 3, "share": 0.75}]}
    assert compare_facts("compare", rows, rows)
    other = {"rows": [{"category": "C", "label": "positive", "count": 3, "share": 0.5}]}
    assert not compare_facts("compare", rows, other)


def test_hit_at_8_and_percentile() -> None:
    from voc.rag.evaluate import hit_at_k, percentile

    doc = RetrievedReview(
        review_id="r1",
        product_id="p1",
        category="All_Beauty",
        label="negative",
        text="The battery drains.",
        reviewed_at=None,
        score=0.9,
    )
    assert hit_at_k(["battery", "screen"], [doc])
    assert not hit_at_k(["screen"], [doc])
    assert percentile([], 50) == 0.0
    assert percentile([10.0, 20.0], 50) == 15.0

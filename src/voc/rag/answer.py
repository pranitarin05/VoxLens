"""PRD 10: grounded RAG answering and LLM phrasing of SQL results.

The LLM sees numbered snippets, must cite them and must abstain when the
evidence is insufficient. Analytical answers are phrased from whitelisted SQL
facts; the prompt forbids introducing numbers that are not in those facts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from voc.models.llm_zero_shot import ChatResult, LLMClient
from voc.rag.retrieve import RetrievedReview
from voc.rag.router import Analysis

RAG_SYSTEM = (
    "You answer questions about Amazon customer reviews using ONLY the numbered "
    "snippets provided. Cite snippets by their bracketed number. If the snippets "
    "do not contain enough evidence, abstain. Never use outside knowledge. "
    'Reply with exactly one JSON object: {"answer": string, "citations": [int], '
    '"abstained": boolean}. If abstaining, use an empty citations list and say '
    "that the available reviews do not answer the question."
)

ANALYST_SYSTEM = (
    "You phrase the result of a database query for a business user. Use only the "
    "numbers in the supplied facts; never add, estimate or recalculate a number. "
    "Be concise (1-3 sentences) and mention the filters that were applied."
)


class RAGParseError(RuntimeError):
    """Raised when the model output is not the required JSON object."""


@dataclass(frozen=True)
class RagAnswer:
    """Parsed grounded answer."""

    answer: str
    citations: list[int]
    abstained: bool
    parse_failed: bool = False


@dataclass(frozen=True)
class AnswerResult:
    """API/eval result for one question."""

    answer: str
    sources: list[dict[str, object]]
    abstained: bool
    route: str
    latency_ms: float
    cost_usd: float
    cached: bool
    citations: list[int] = field(default_factory=list)
    parse_failed: bool = False


def build_snippets(docs: list[RetrievedReview], max_chars: int = 400) -> str:
    """Numbered snippet block passed to the LLM."""
    lines = []
    for index, doc in enumerate(docs, start=1):
        reviewed = doc.reviewed_at.isoformat() if hasattr(doc.reviewed_at, "isoformat") else "?"
        text = doc.text.replace("\n", " ")[:max_chars]
        lines.append(
            f"[{index}] review_id={doc.review_id} category={doc.category} "
            f"label={doc.label} reviewed_at={reviewed}\n{text}"
        )
    return "\n\n".join(lines)


def build_rag_prompt(question: str, docs: list[RetrievedReview]) -> str:
    """Deterministic RAG user prompt."""
    return (
        f"Question: {question}\n\n"
        f"Snippets (use only these):\n{build_snippets(docs)}\n\n"
        "Return the JSON object now."
    )


def parse_json_object(raw: str) -> dict[str, object]:
    """Extract the first JSON object from a model reply (tolerates fences)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RAGParseError(f"no JSON object in reply: {raw[:200]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RAGParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RAGParseError("JSON payload is not an object")
    return payload


def parse_rag_answer(raw: str, n_snippets: int) -> RagAnswer:
    """Validate the answer/citations/abstained contract."""
    payload = parse_json_object(raw)
    answer = str(payload.get("answer", "")).strip()
    abstained = bool(payload.get("abstained", False))
    citations_raw = payload.get("citations", [])
    if not isinstance(citations_raw, list):
        raise RAGParseError("citations must be a list")
    citations = sorted({int(value) for value in citations_raw})
    if any(index < 1 or index > n_snippets for index in citations):
        raise RAGParseError(f"citation out of range 1..{n_snippets}: {citations}")
    if not answer:
        raise RAGParseError("empty answer")
    if not abstained and not citations:
        raise RAGParseError("non-abstained answer has no citations")
    return RagAnswer(answer=answer, citations=citations, abstained=abstained)


def sources_from_docs(docs: list[RetrievedReview], max_chars: int = 240) -> list[dict]:
    """API sources: review id, snippet, score (PRD section 9)."""
    return [
        {
            "review_id": doc.review_id,
            "snippet": doc.text.replace("\n", " ")[:max_chars],
            "score": round(doc.score, 6),
            "label": doc.label,
            "category": doc.category,
        }
        for doc in docs
    ]


def answer_from_docs(client: LLMClient, question: str, docs: list[RetrievedReview]) -> AnswerResult:
    """Ask the LLM to answer from snippets; abstain on empty evidence."""
    started = time.perf_counter()
    if not docs:
        return AnswerResult(
            answer="The available reviews do not answer this question.",
            sources=[],
            abstained=True,
            route="rag",
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            cost_usd=0.0,
            cached=False,
        )
    try:
        result: ChatResult = client.complete(
            build_rag_prompt(question, docs),
            system_prompt=RAG_SYSTEM,
            max_completion_tokens=512,
            cache_tag="rag-answer-v1",
        )
        parsed = parse_rag_answer(result.content, len(docs))
        answer = parsed.answer
        abstained = parsed.abstained
        citations = parsed.citations
        parse_failed = False
    except RAGParseError:
        result = None
        answer = "The available reviews did not produce a verifiable answer."
        abstained = True
        citations = []
        parse_failed = True
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return AnswerResult(
        answer=answer,
        sources=sources_from_docs(docs),
        abstained=abstained,
        route="rag",
        latency_ms=latency_ms,
        cost_usd=result.cost_usd if result else 0.0,
        cached=result.cached if result else False,
        citations=citations,
        parse_failed=parse_failed,
    )


def phrase_analysis(client: LLMClient, question: str, analysis: Analysis) -> AnswerResult:
    """LLM phrases whitelisted SQL facts; falls back to the deterministic summary."""
    started = time.perf_counter()
    prompt = (
        f"Question: {question}\n"
        f"SQL template: {analysis.sql_name}\n"
        f"Filters: {json.dumps(analysis.params, default=str)}\n"
        f"Facts: {json.dumps(analysis.facts, default=str)}\n"
        f"Deterministic summary: {analysis.summary}\n"
        "Write the answer now."
    )
    try:
        result = client.complete(
            prompt,
            system_prompt=ANALYST_SYSTEM,
            max_completion_tokens=256,
            cache_tag="rag-sql-phrase-v1",
        )
        answer = result.content.strip() or analysis.summary
        cost = result.cost_usd
        cached = result.cached
    except Exception:  # noqa: BLE001 - phrasing must not break a correct query
        result = None
        answer = analysis.summary
        cost = 0.0
        cached = False
    return AnswerResult(
        answer=answer,
        sources=[],
        abstained=False,
        route="sql",
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        cost_usd=cost,
        cached=cached,
    )

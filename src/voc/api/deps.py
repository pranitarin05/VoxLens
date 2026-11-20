"""Shared dependencies: DB session, model holder, settings."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from voc.db.load import database_url
from voc.db.queries import insert_api_requests

LOGGER = logging.getLogger("voc.api")

ScoreFn = Callable[[Sequence[str]], list[dict[str, float]]]


@dataclass(frozen=True)
class Settings:
    """Runtime settings; all overridable through environment variables."""

    production_model: str = "distilbert"
    model_dir: Path = Path("models")
    taxonomy_path: Path = Path("configs/aspects.yaml")
    spacy_model: str = "en_core_web_sm"
    max_length: int = 128
    torch_threads: int | None = None
    aspects_enabled: bool = True
    rag_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    rag_keep: int = 8

    @classmethod
    def from_env(cls) -> Settings:
        """Read API_* variables with production defaults."""
        threads = os.environ.get("API_TORCH_THREADS")
        return cls(
            production_model=os.environ.get("API_PRODUCTION_MODEL", "distilbert"),
            model_dir=Path(os.environ.get("API_MODEL_DIR", "models")),
            taxonomy_path=Path(os.environ.get("API_TAXONOMY", "configs/aspects.yaml")),
            spacy_model=os.environ.get("API_SPACY_MODEL", "en_core_web_sm"),
            max_length=int(os.environ.get("API_MAX_LENGTH", "128")),
            torch_threads=int(threads) if threads else None,
            aspects_enabled=os.environ.get("API_ASPECTS", "1") != "0",
            rag_enabled=os.environ.get("API_RAG", "1") != "0",
            embedding_model=os.environ.get(
                "API_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_device=os.environ.get("API_EMBED_DEVICE", "cpu"),
            rag_keep=int(os.environ.get("API_RAG_KEEP", "8")),
        )


@dataclass
class LoadedModel:
    """A model plus the lock that serialises calls into its tokenizer/model."""

    name: str
    version: str
    score_fn: ScoreFn
    lock: threading.Lock

    def probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return one {label: probability} dict per text."""
        with self.lock:
            return self.score_fn(list(texts))

    def predict(self, texts: Sequence[str]) -> list[tuple[str, dict[str, float]]]:
        """Return (argmax label, probabilities) per text."""
        return [(max(row, key=lambda label: row[label]), row) for row in self.probabilities(texts)]


MODEL_SPECS: dict[str, tuple[str, str]] = {
    "tfidf": ("tfidf.joblib", "tfidf-logreg-v1"),
    "distilbert": ("distilbert-60k", "distilbert-60k-v1"),
    "bert": ("bert-30k", "bert-30k-v1"),
}


def _load_tfidf(path: Path, name: str, version: str) -> LoadedModel:
    from voc.models.tfidf import load_model

    pipeline = load_model(path)
    classes = [str(label) for label in pipeline.classes_]

    def score(texts: Sequence[str]) -> list[dict[str, float]]:
        proba = pipeline.predict_proba(list(texts))
        return [{label: float(p) for label, p in zip(classes, row, strict=False)} for row in proba]

    return LoadedModel(name=name, version=version, score_fn=score, lock=threading.Lock())


def _load_transformer(path: Path, name: str, version: str, settings: Settings) -> LoadedModel:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()
    labels = [str(model.config.id2label[i]) for i in range(len(model.config.id2label))]

    def score(texts: Sequence[str]) -> list[dict[str, float]]:
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=settings.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).tolist()
        return [{label: float(p) for label, p in zip(labels, row, strict=False)} for row in probs]

    return LoadedModel(name=name, version=version, score_fn=score, lock=threading.Lock())


class ModelRegistry:
    """All models loaded once at startup; never reloaded per request."""

    def __init__(self, models: dict[str, LoadedModel], production: str) -> None:
        self._models = models
        self.production = production

    @classmethod
    def load(cls, settings: Settings) -> ModelRegistry:
        """Load every model present on disk; fail if production is missing."""
        models: dict[str, LoadedModel] = {}
        for name, (relative, version) in MODEL_SPECS.items():
            path = settings.model_dir / relative
            if not path.exists():
                LOGGER.warning("model %s not found at %s; skipping", name, path)
                continue
            if name == "tfidf":
                models[name] = _load_tfidf(path, name, version)
            else:
                models[name] = _load_transformer(path, name, version, settings)
            LOGGER.info("loaded model %s (%s)", name, version)
        registry = cls(models, settings.production_model)
        registry.get(settings.production_model)  # raises KeyError if absent
        return registry

    def __contains__(self, name: object) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)

    def get(self, name: str) -> LoadedModel:
        """Return a loaded model or raise KeyError."""
        return self._models[name]

    def info(self) -> list[dict[str, str]]:
        """List loaded models in a stable order."""
        return [
            {"name": name, "version": self._models[name].version}
            for name in MODEL_SPECS
            if name in self._models
        ]


class AspectExtractor:
    """spaCy mention detection with DistilBERT sentence sentiment."""

    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        import spacy

        from voc.nlp.aspects import build_matcher, load_taxonomy

        self.registry = registry
        self.nlp = spacy.load(settings.spacy_model)
        self.matcher, self.phrase_to_aspect, self.lemma_to_aspect = build_matcher(
            self.nlp, load_taxonomy(settings.taxonomy_path)
        )
        self._lock = threading.Lock()

    def extract(self, texts: Sequence[str]) -> list[list[dict[str, str]]]:
        """Return aspect/sentiment/span dicts for each text."""
        from voc.nlp.aspects import extract_mentions

        if "distilbert" not in self.registry:
            return [[] for _ in texts]
        with self._lock:
            docs = list(self.nlp.pipe(list(texts), batch_size=16))

        mentions_by_text = [
            extract_mentions(doc, self.matcher, self.phrase_to_aspect, self.lemma_to_aspect)
            for doc in docs
        ]
        sentences: list[str] = []
        sentence_index: dict[str, int] = {}
        for mentions in mentions_by_text:
            for mention in mentions:
                sentence = mention.sentence or mention.span_text
                if sentence not in sentence_index:
                    sentence_index[sentence] = len(sentences)
                    sentences.append(sentence)

        sentiments: list[str] = []
        if sentences:
            scored = self.registry.get("distilbert").predict(sentences)
            sentiments = [label for label, _ in scored]

        results: list[list[dict[str, str]]] = []
        for mentions in mentions_by_text:
            results.append(
                [
                    {
                        "aspect": mention.aspect,
                        "sentiment": sentiments[
                            sentence_index[mention.sentence or mention.span_text]
                        ],
                        "span": mention.span_text,
                    }
                    for mention in mentions
                ]
            )
        return results


def build_aspect_extractor(registry: ModelRegistry, settings: Settings) -> AspectExtractor | None:
    """Build the aspect service unless explicitly disabled."""
    if not settings.aspects_enabled:
        return None
    return AspectExtractor(registry, settings)


class RagService:
    """Embeddings + retrieval + whitelisted SQL + Groq phrasing, loaded once."""

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        from voc.models.llm_zero_shot import LLMClient

        self.settings = settings
        self.embedder = SentenceTransformer(
            settings.embedding_model, device=settings.embedding_device
        )
        self.client = LLMClient()
        self._embed_lock = threading.Lock()

    def answer(self, question: str, filters: dict[str, object] | None = None):
        """Route one question; returns voc.rag.answer.AnswerResult."""
        from voc.rag.answer import answer_from_docs, phrase_analysis
        from voc.rag.retrieve import retrieve
        from voc.rag.router import list_aspects, list_categories, route, run_analysis

        filters = filters or {}
        retrieval_keys = ("category", "product_id", "sentiment", "start", "end")
        kwargs = {key: value for key, value in filters.items() if key in retrieval_keys}
        sql_blocked = any(key in filters for key in ("product_id", "start", "end"))
        with db_connection() as conn:
            if route(question) == "sql" and not sql_blocked:
                analysis = run_analysis(
                    conn,
                    question,
                    list_categories(conn),
                    list_aspects(conn),
                    filters,
                )
                if analysis is not None:
                    return phrase_analysis(self.client, question, analysis)
            with self._embed_lock:
                vector = self.embedder.encode(
                    [question], normalize_embeddings=True, convert_to_numpy=True
                )[0]
            docs = retrieve(conn, vector, keep=self.settings.rag_keep, **kwargs)
        return answer_from_docs(self.client, question, docs)


def build_rag_service(settings: Settings) -> RagService | None:
    """Build the RAG service unless explicitly disabled."""
    if not settings.rag_enabled:
        return None
    return RagService(settings)


def db_connection(timeout: int = 3) -> psycopg.Connection:
    """Open a short-lived psycopg connection using DATABASE_URL."""
    return psycopg.connect(database_url(), connect_timeout=timeout)


def ping_database() -> bool:
    """Return True when a trivial query succeeds."""
    try:
        with db_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - health must never raise
        LOGGER.exception("database ping failed")
        return False


def log_requests(records: Sequence[tuple[str, str, float, str, int]]) -> bool:
    """Best-effort audit insert; returns False when the database is unavailable."""
    if not records:
        return True
    try:
        with db_connection() as conn:
            insert_api_requests(conn, records)
        return True
    except Exception:  # noqa: BLE001 - prediction must not fail because of audit
        LOGGER.exception("failed to record api_requests")
        return False

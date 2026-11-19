"""PRD 8 / FR-8: aspect extraction and sentence-level sentiment.

Pipeline per review: spaCy (en_core_web_sm) sentence splitting + noun chunks,
PhraseMatcher over the taxonomy in configs/aspects.yaml, then the production
DistilBERT classifier applied to the sentence containing each aspect span.
Results are written to parquet and loaded into the ``aspects`` table with an
idempotent unique index on (review_id, aspect, sentence_index, char_start,
char_end).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import psycopg
import typer
import yaml

from voc.db.load import apply_migrations, database_url

DEFAULT_TAXONOMY = Path("configs/aspects.yaml")
DEFAULT_REVIEWS = Path("data/processed/reviews.parquet")
DEFAULT_MODEL_DIR = Path("models/distilbert-60k")
DEFAULT_ASPECTS_PARQUET = Path("data/processed/aspects.parquet")
DEFAULT_GOLD = Path("eval/aspect_gold.jsonl")
DEFAULT_REPORT = Path("results/aspects.md")

app = typer.Typer(add_completion=False, help="PRD 8: aspect extraction.")


@dataclass(frozen=True)
class AspectMention:
    aspect: str
    span_text: str
    char_start: int
    char_end: int
    sentence: str
    sentence_index: int


def load_taxonomy(path: Path = DEFAULT_TAXONOMY) -> dict[str, list[str]]:
    """Return {aspect_name: [terms]} from YAML."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    taxonomy: dict[str, list[str]] = {}
    for item in data.get("aspects", []):
        terms = [str(term).lower() for term in item.get("terms", [])]
        if terms:
            taxonomy[str(item["name"])] = terms
    if not taxonomy:
        raise ValueError(f"no aspects in {path}")
    return taxonomy


def build_matcher(nlp: Any, taxonomy: dict[str, list[str]]):
    """Build a PhraseMatcher plus phrase/lemma lookup maps."""
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    phrase_to_aspect: dict[str, str] = {}
    lemma_to_aspect: dict[str, str] = {}
    for aspect, terms in taxonomy.items():
        patterns = []
        for term in terms:
            key = term.lower()
            phrase_to_aspect[key] = aspect
            patterns.append(nlp.make_doc(key))
            doc = nlp.make_doc(key)
            if len(doc) == 1:
                lemma_to_aspect.setdefault(doc[0].lemma_.lower(), aspect)
                lemma_to_aspect.setdefault(key, aspect)
        matcher.add(aspect, patterns)
    return matcher, phrase_to_aspect, lemma_to_aspect


def _sentence_lookup(doc: Any):
    spans = [
        (sent.start_char, sent.end_char, sent.text.strip(), i) for i, sent in enumerate(doc.sents)
    ]

    def find(char_pos: int) -> tuple[str, int]:
        for start, end, text, index in spans:
            if start <= char_pos < end:
                return text, index
        return "", -1

    return find


def extract_mentions(
    doc: Any,
    matcher: Any,
    phrase_to_aspect: dict[str, str],
    lemma_to_aspect: dict[str, str],
) -> list[AspectMention]:
    """Return one mention per (aspect, sentence), keeping the longest span."""
    find_sentence = _sentence_lookup(doc)
    found: dict[tuple[str, int], AspectMention] = {}

    def add(aspect: str, start: int, end: int, span_text: str) -> None:
        sentence, sentence_index = find_sentence(start)
        key = (aspect, sentence_index)
        current = found.get(key)
        if current is None or (end - start) > (current.char_end - current.char_start):
            found[key] = AspectMention(
                aspect=aspect,
                span_text=span_text,
                char_start=start,
                char_end=end,
                sentence=sentence,
                sentence_index=sentence_index,
            )

    for match_id, start, end in matcher(doc):
        aspect = str(doc.vocab.strings[match_id])
        span = doc[start:end]
        add(aspect, span.start_char, span.end_char, span.text)

    for chunk in doc.noun_chunks:
        chunk_key = chunk.text.lower()
        if chunk_key in phrase_to_aspect:
            add(phrase_to_aspect[chunk_key], chunk.start_char, chunk.end_char, chunk.text)
            continue
        for token in chunk:
            aspect = lemma_to_aspect.get(token.lemma_.lower())
            if aspect:
                add(aspect, chunk.start_char, chunk.end_char, chunk.text)
                break

    for token in doc:
        aspect = lemma_to_aspect.get(token.lemma_.lower())
        if aspect:
            add(aspect, token.idx, token.idx + len(token.text), token.text)

    return sorted(found.values(), key=lambda m: (m.sentence_index, m.char_start))


class SentimentScorer:
    """Batch classifier wrapper around the local fine-tuned DistilBERT."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_dir.exists():
            raise FileNotFoundError(
                f"{model_dir} not found — run Phase 5 training or pass --model-dir"
            )
        self.torch = torch
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.version = model_dir.name

    def predict(self, texts: Sequence[str], batch_size: int = 64) -> list[str]:
        labels: list[str] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits
                labels.extend(
                    self.id2label[int(index)] for index in self.torch.argmax(logits, dim=-1).cpu()
                )
        return labels


def extract_frame(
    frame: pd.DataFrame,
    nlp: Any,
    matcher: Any,
    phrase_to_aspect: dict[str, str],
    lemma_to_aspect: dict[str, str],
    predict_fn: Callable[[list[str]], Sequence[str]],
    *,
    model_version: str = "unknown",
    limit: int | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Extract aspects for ``frame`` and label their sentences in one batch."""
    subset = frame.head(limit) if limit is not None else frame
    docs = nlp.pipe(subset["text"].astype(str).tolist(), batch_size=batch_size)
    per_review: list[tuple[str, list[AspectMention]]] = []
    sentences: dict[str, None] = {}
    for review_id, doc in zip(subset["review_id"].tolist(), docs, strict=False):
        mentions = extract_mentions(doc, matcher, phrase_to_aspect, lemma_to_aspect)
        per_review.append((review_id, mentions))
        for mention in mentions:
            sentences.setdefault(mention.sentence, None)

    sentence_list = list(sentences)
    labels = list(predict_fn(sentence_list)) if sentence_list else []
    label_map = dict(zip(sentence_list, labels, strict=False))

    rows = []
    for review_id, mentions in per_review:
        for mention in mentions:
            rows.append(
                {
                    "review_id": review_id,
                    "aspect": mention.aspect,
                    "sentiment": label_map.get(mention.sentence, "unknown"),
                    "span_text": mention.span_text,
                    "sentence_index": mention.sentence_index,
                    "char_start": mention.char_start,
                    "char_end": mention.char_end,
                    "sentence": mention.sentence,
                    "model_version": model_version,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "review_id",
            "aspect",
            "sentiment",
            "span_text",
            "sentence_index",
            "char_start",
            "char_end",
            "sentence",
            "model_version",
        ],
    )


def load_nlp():
    import en_core_web_sm

    return en_core_web_sm.load()


def _pipeline(taxonomy_path: Path, model_dir: Path, device: str):
    nlp = load_nlp()
    taxonomy = load_taxonomy(taxonomy_path)
    matcher, phrase_to_aspect, lemma_to_aspect = build_matcher(nlp, taxonomy)
    scorer = SentimentScorer(model_dir, device=device)
    return nlp, matcher, phrase_to_aspect, lemma_to_aspect, scorer


@app.command()
def extract(
    reviews: Annotated[Path, typer.Option()] = DEFAULT_REVIEWS,
    out: Annotated[Path, typer.Option()] = DEFAULT_ASPECTS_PARQUET,
    taxonomy: Annotated[Path, typer.Option()] = DEFAULT_TAXONOMY,
    model_dir: Annotated[Path, typer.Option()] = DEFAULT_MODEL_DIR,
    device: Annotated[str, typer.Option()] = "cpu",
    limit: Annotated[int, typer.Option()] = 50000,
    batch_size: Annotated[int, typer.Option()] = 32,
) -> None:
    """Extract aspects + sentence sentiment for the first ``limit`` reviews."""
    frame = pd.read_parquet(reviews, columns=["review_id", "text"]).head(limit)
    nlp, matcher, phrase_to_aspect, lemma_to_aspect, scorer = _pipeline(taxonomy, model_dir, device)
    parts: list[pd.DataFrame] = []
    chunk = 1000
    for start in range(0, len(frame), chunk):
        part = frame.iloc[start : start + chunk]
        parts.append(
            extract_frame(
                part,
                nlp,
                matcher,
                phrase_to_aspect,
                lemma_to_aspect,
                scorer.predict,
                model_version=scorer.version,
                batch_size=batch_size,
            )
        )
        typer.echo(f"processed {min(start + chunk, len(frame))}/{len(frame)} reviews")
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    typer.echo(
        f"reviews={len(frame)} aspects={len(result)} "
        f"unique_reviews={result['review_id'].nunique() if len(result) else 0} -> {out}"
    )


@app.command("load")
def load_db(path: Annotated[Path, typer.Option()] = DEFAULT_ASPECTS_PARQUET) -> None:
    """COPY aspects parquet into Postgres (idempotent on the span key)."""
    if not path.exists():
        raise FileNotFoundError(path)
    applied = apply_migrations()
    if applied:
        typer.echo(f"applied migrations: {applied}")
    frame = pd.read_parquet(path)
    columns = (
        "review_id",
        "aspect",
        "sentiment",
        "span_text",
        "sentence_index",
        "char_start",
        "char_end",
        "sentence",
        "model_version",
    )
    frame = frame[list(columns)]
    with psycopg.connect(database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE tmp_aspects (LIKE aspects INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        with cur.copy(f"COPY tmp_aspects ({', '.join(columns)}) FROM STDIN") as copy:
            for row in frame.itertuples(index=False, name=None):
                copy.write_row(row)
        cur.execute(
            f"""
            INSERT INTO aspects ({", ".join(columns)})
            SELECT {", ".join(columns)} FROM tmp_aspects
            ON CONFLICT (review_id, aspect, sentence_index, char_start, char_end)
            DO NOTHING
            """
        )
        inserted = cur.rowcount
        cur.execute("SELECT count(*) FROM aspects")
        total = cur.fetchone()[0]
    typer.echo(f"inserted={inserted} total_aspects={total}")


def _sentence_candidates(mentions: pd.DataFrame, max_len: int = 220) -> list[dict[str, Any]]:
    """Unique single-aspect sentences, diversified by aspect."""
    candidates: dict[str, dict[str, Any]] = {}
    for row in mentions.itertuples(index=False):
        sentence = str(row.sentence).strip()
        if not (20 <= len(sentence) <= max_len):
            continue
        if sentence in candidates and candidates[sentence]["aspect"] != row.aspect:
            continue
        candidates.setdefault(
            sentence,
            {"text": sentence, "aspect": row.aspect, "sentiment": row.sentiment},
        )
    by_aspect: dict[str, list[dict[str, Any]]] = {}
    for item in candidates.values():
        by_aspect.setdefault(item["aspect"], []).append(item)
    ordered: list[dict[str, Any]] = []
    while any(by_aspect.values()):
        for aspect in list(by_aspect):
            if by_aspect[aspect]:
                ordered.append(by_aspect[aspect].pop(0))
    return ordered


@app.command("draft-gold")
def draft_gold(
    out: Annotated[Path, typer.Option()] = DEFAULT_GOLD,
    n: Annotated[int, typer.Option()] = 100,
    limit: Annotated[int, typer.Option()] = 5000,
    seed: Annotated[int, typer.Option()] = 42,
    mentions_path: Annotated[
        Path | None, typer.Option(help="reuse an existing extracted parquet")
    ] = None,
) -> None:
    """Draft a gold file from real sentences (DRAFT: owner must review labels)."""
    if mentions_path is not None and mentions_path.exists():
        mentions = pd.read_parquet(mentions_path)
    else:
        nlp, matcher, phrase_to_aspect, lemma_to_aspect, scorer = _pipeline(
            DEFAULT_TAXONOMY, DEFAULT_MODEL_DIR, "cpu"
        )
        frame = pd.read_parquet(DEFAULT_REVIEWS, columns=["review_id", "text"])
        frame = frame.sample(n=min(limit, len(frame)), random_state=seed)
        mentions = extract_frame(
            frame,
            nlp,
            matcher,
            phrase_to_aspect,
            lemma_to_aspect,
            scorer.predict,
            model_version=scorer.version,
        )
    items = _sentence_candidates(mentions)[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            record = {
                "text": item["text"],
                "aspects": [{"aspect": item["aspect"], "sentiment": item["sentiment"]}],
                "review_status": "draft-pending-owner-review",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    typer.echo(f"wrote {len(items)} draft sentences -> {out}")


def evaluate_gold(
    gold: Sequence[dict[str, Any]],
    predictions: Sequence[set[str]],
    sentiment_lookup: Callable[[int, str], str | None],
) -> dict[str, Any]:
    """Micro aspect P/R/F1 and sentiment accuracy on matched aspects."""
    tp = fp = fn = 0
    sentiment_correct = sentiment_total = 0
    per_aspect: dict[str, dict[str, int]] = {}
    for index, (record, predicted) in enumerate(zip(gold, predictions, strict=False)):
        gold_labels = {item["aspect"]: item["sentiment"] for item in record["aspects"]}
        gold_set = set(gold_labels)
        tp += len(gold_set & predicted)
        fp += len(predicted - gold_set)
        fn += len(gold_set - predicted)
        for aspect in gold_set & predicted:
            sentiment_total += 1
            stats = per_aspect.setdefault(aspect, {"correct": 0, "total": 0})
            stats["total"] += 1
            if sentiment_lookup(index, aspect) == gold_labels[aspect]:
                sentiment_correct += 1
                stats["correct"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sentiment_accuracy": (sentiment_correct / sentiment_total if sentiment_total else 0.0),
        "sentiment_matched": sentiment_total,
        "per_aspect": per_aspect,
    }


@app.command()
def evaluate(
    gold: Annotated[Path, typer.Option()] = DEFAULT_GOLD,
    out: Annotated[Path, typer.Option()] = DEFAULT_REPORT,
) -> None:
    """Score aspect detection + sentiment against the gold JSONL."""
    records = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line]
    statuses = {record.get("review_status", "approved") for record in records}
    nlp, matcher, phrase_to_aspect, lemma_to_aspect, scorer = _pipeline(
        DEFAULT_TAXONOMY, DEFAULT_MODEL_DIR, "cpu"
    )
    frame = pd.DataFrame(
        {"review_id": [str(i) for i in range(len(records))], "text": [r["text"] for r in records]}
    )
    predicted_rows = extract_frame(
        frame,
        nlp,
        matcher,
        phrase_to_aspect,
        lemma_to_aspect,
        scorer.predict,
        model_version=scorer.version,
    )
    predictions = [
        set(predicted_rows.loc[predicted_rows["review_id"] == str(i), "aspect"])
        for i in range(len(records))
    ]
    sentiments = {
        (row.review_id, row.aspect): row.sentiment for row in predicted_rows.itertuples(index=False)
    }
    scores = evaluate_gold(
        records,
        predictions,
        lambda index, aspect: sentiments.get((str(index), aspect)),
    )

    lines = [
        "# Aspect extraction (measured)",
        "",
        f"Gold file: `{gold}` ({len(records)} sentences).",
        "Detection = micro P/R/F1 over aspect mentions; sentiment accuracy on matched aspects.",
        "",
    ]
    if statuses != {"approved"}:
        lines += [
            "> **DRAFT GOLD SET**: labels were drafted by the agent and are pending",
            "> owner review. Treat these numbers as pipeline validation, not final.",
            "",
        ]
    lines += [
        f"- Aspect precision: {scores['precision']:.4f}",
        f"- Aspect recall: {scores['recall']:.4f}",
        f"- Aspect F1: {scores['f1']:.4f}",
        f"- Sentiment accuracy: {scores['sentiment_accuracy']:.4f} "
        f"({scores['sentiment_matched']} matched)",
        "",
        "| aspect | sentiment correct | sentiment total |",
        "| --- | --- | --- |",
    ]
    for aspect, stats in sorted(scores["per_aspect"].items()):
        lines.append(f"| {aspect} | {stats['correct']} | {stats['total']} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(
        f"P={scores['precision']:.4f} R={scores['recall']:.4f} F1={scores['f1']:.4f} "
        f"sentiment_acc={scores['sentiment_accuracy']:.4f} -> {out}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""Phase 7: taxonomy matching, mention dedupe and gold-set scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from voc.nlp.aspects import (
    _sentence_candidates,
    build_matcher,
    evaluate_gold,
    extract_frame,
    extract_mentions,
    load_taxonomy,
)

pytest.importorskip("en_core_web_sm")


@pytest.fixture(scope="module")
def pipeline():
    import en_core_web_sm

    nlp = en_core_web_sm.load()
    taxonomy = load_taxonomy()
    matcher, phrase_to_aspect, lemma_to_aspect = build_matcher(nlp, taxonomy)
    return nlp, matcher, phrase_to_aspect, lemma_to_aspect


def test_taxonomy_has_expected_aspects() -> None:
    taxonomy = load_taxonomy()
    assert "battery" in taxonomy
    assert "screen" in taxonomy
    assert len(taxonomy) >= 15


def test_extract_mentions_splits_sentences_and_dedupes(pipeline) -> None:
    nlp, matcher, phrase_to_aspect, lemma_to_aspect = pipeline
    doc = nlp("The battery life is great. The screen is awful.")
    mentions = extract_mentions(doc, matcher, phrase_to_aspect, lemma_to_aspect)
    by_aspect = {m.aspect: m for m in mentions}
    assert set(by_aspect) == {"battery", "screen"}
    assert by_aspect["battery"].sentence_index == 0
    assert by_aspect["screen"].sentence_index == 1
    assert "battery" in by_aspect["battery"].span_text.lower()
    assert "screen" in by_aspect["screen"].span_text.lower()


def test_extract_mentions_uses_lemmas(pipeline) -> None:
    nlp, matcher, phrase_to_aspect, lemma_to_aspect = pipeline
    doc = nlp("The batteries died after a week.")
    aspects = {m.aspect for m in extract_mentions(doc, matcher, phrase_to_aspect, lemma_to_aspect)}
    assert "battery" in aspects


def test_extract_frame_assigns_sentence_sentiment(pipeline) -> None:
    nlp, matcher, phrase_to_aspect, lemma_to_aspect = pipeline
    frame = pd.DataFrame(
        {"review_id": ["r1", "r2"], "text": ["Great battery life.", "The screen is awful."]}
    )
    result = extract_frame(
        frame,
        nlp,
        matcher,
        phrase_to_aspect,
        lemma_to_aspect,
        lambda sentences: ["positive" for _ in sentences],
        model_version="fake",
    )
    assert len(result) == 2
    assert set(result["review_id"]) == {"r1", "r2"}
    assert set(result["sentiment"]) == {"positive"}


def test_evaluate_gold_scores_detection_and_sentiment() -> None:
    gold = [
        {"text": "a", "aspects": [{"aspect": "battery", "sentiment": "positive"}]},
        {"text": "b", "aspects": [{"aspect": "screen", "sentiment": "negative"}]},
    ]
    predictions = [{"battery", "price"}, {"screen"}]
    sentiments = {(0, "battery"): "positive", (1, "screen"): "positive"}
    scores = evaluate_gold(gold, predictions, lambda i, a: sentiments.get((i, a)))
    assert scores["tp"] == 2
    assert scores["fp"] == 1
    assert scores["fn"] == 0
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["sentiment_accuracy"] == pytest.approx(0.5)


def test_sentence_candidates_diversifies_by_aspect() -> None:
    frame = pd.DataFrame(
        {
            "sentence": [
                "The battery life is fine overall.",
                "The screen quality is fine overall.",
                "The battery life is good overall.",
            ],
            "aspect": ["battery", "screen", "battery"],
            "sentiment": ["positive", "positive", "positive"],
        }
    )
    candidates = _sentence_candidates(frame)
    aspects = [item["aspect"] for item in candidates]
    assert aspects[0] != aspects[1]

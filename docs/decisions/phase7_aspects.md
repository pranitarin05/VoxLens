# Phase 7: aspect extraction

- Taxonomy: 16 aspects in `configs/aspects.yaml`; PhraseMatcher + noun chunks + lemma mapping.
- Sentiment: production DistilBERT applied to the sentence containing the aspect.
- spaCy model: `en_core_web_sm==3.8.0` installed from the official wheel (spaCy download timed out at 15 s).
- Extraction run: 5,000 reviews -> 7,111 aspect rows (3,125 reviews) in ~7 min on CPU.
- DB: migration `0002_aspects_span.sql`, unique span key; load 7,111; re-run inserted 0.
- Gold: `eval/aspect_gold.jsonl` has 100 sentences marked `draft-pending-owner-review`.
  Draft scores (P 0.8929 / R 1.0 / F1 0.9434, sentiment 1.0) are **circular** and must not be
  treated as final until the owner corrects the labels.

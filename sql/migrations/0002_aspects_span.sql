-- Phase 7: span-level aspect rows, idempotent on (review_id, aspect, sentence, span).
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS sentence_index INT;
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS char_start INT;
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS char_end INT;
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS sentence TEXT;
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();

CREATE UNIQUE INDEX IF NOT EXISTS uq_aspects_span
  ON aspects (review_id, aspect, sentence_index, char_start, char_end);
CREATE INDEX IF NOT EXISTS idx_aspects_review ON aspects (review_id);

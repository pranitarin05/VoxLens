-- Business question: What does the train/val/test label balance look like for ML readiness?
SELECT
  split,
  label,
  COUNT(*) AS reviews,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 2) AS pct_in_split
FROM reviews
GROUP BY split, label
ORDER BY split, reviews DESC;

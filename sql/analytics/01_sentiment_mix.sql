-- Business question: What is the overall sentiment mix across all reviews?
-- (PRD FR-4: every analytics file starts with the business question it answers.)
SELECT
  label,
  COUNT(*) AS reviews,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM reviews
GROUP BY label
ORDER BY reviews DESC;

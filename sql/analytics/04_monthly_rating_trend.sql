-- Business question: How has average rating trended month over month?
SELECT
  date_trunc('month', reviewed_at) AS month,
  COUNT(*) AS reviews,
  ROUND(AVG(rating)::numeric, 3) AS avg_rating,
  ROUND(100.0 * COUNT(*) FILTER (WHERE label = 'negative') / COUNT(*), 2) AS negative_pct
FROM reviews
WHERE reviewed_at IS NOT NULL
GROUP BY 1
ORDER BY 1;

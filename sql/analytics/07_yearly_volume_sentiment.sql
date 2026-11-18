-- Business question: How does review volume and sentiment vary by year?
SELECT
  EXTRACT(YEAR FROM reviewed_at)::int AS year,
  COUNT(*) AS reviews,
  ROUND(AVG(rating)::numeric, 3) AS avg_rating,
  ROUND(100.0 * COUNT(*) FILTER (WHERE label = 'positive') / COUNT(*), 2) AS positive_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE label = 'neutral') / COUNT(*), 2) AS neutral_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE label = 'negative') / COUNT(*), 2) AS negative_pct
FROM reviews
WHERE reviewed_at IS NOT NULL
GROUP BY 1
ORDER BY 1;

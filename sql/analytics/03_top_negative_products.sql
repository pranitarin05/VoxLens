-- Business question: Which products generate the most negative reviews?
SELECT
  r.product_id,
  p.category,
  COUNT(*) AS negative_reviews,
  ROUND(100.0 * COUNT(*) / totals.total_reviews, 2) AS negative_rate_pct
FROM reviews r
JOIN products p ON p.product_id = r.product_id
JOIN (
  SELECT product_id, COUNT(*) AS total_reviews
  FROM reviews
  GROUP BY product_id
) totals ON totals.product_id = r.product_id
WHERE r.label = 'negative'
GROUP BY r.product_id, p.category, totals.total_reviews
HAVING COUNT(*) >= 20
ORDER BY negative_reviews DESC
LIMIT 25;

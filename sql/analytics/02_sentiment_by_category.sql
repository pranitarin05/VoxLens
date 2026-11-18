-- Business question: How does sentiment mix differ by product category?
SELECT
  p.category,
  r.label,
  COUNT(*) AS reviews,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY p.category), 2) AS pct_in_category
FROM reviews r
JOIN products p ON p.product_id = r.product_id
GROUP BY p.category, r.label
ORDER BY p.category, reviews DESC;

-- Business question: Which review titles appear most often among negative reviews?
-- (Proxy for recurring complaint themes before Phase 7 aspect extraction.)
SELECT
  lower(trim(r.title)) AS title_norm,
  p.category,
  COUNT(*) AS negative_count
FROM reviews r
JOIN products p ON p.product_id = r.product_id
WHERE r.label = 'negative' AND r.title IS NOT NULL AND length(trim(r.title)) >= 8
GROUP BY 1, 2
HAVING COUNT(*) >= 15
ORDER BY negative_count DESC
LIMIT 30;

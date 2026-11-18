-- Business question: How do average ratings and rating bands compare across categories?
SELECT
  p.category,
  COUNT(*) AS reviews,
  COUNT(*) FILTER (WHERE r.rating >= 4) AS high_ratings,
  COUNT(*) FILTER (WHERE r.rating = 3) AS mid_ratings,
  COUNT(*) FILTER (WHERE r.rating <= 2) AS low_ratings,
  ROUND(AVG(r.rating)::numeric, 3) AS avg_rating
FROM reviews r
JOIN products p ON p.product_id = r.product_id
GROUP BY p.category
ORDER BY avg_rating DESC;

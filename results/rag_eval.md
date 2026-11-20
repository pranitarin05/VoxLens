# RAG evaluation (Phase 9)

Generated: 2026-11-20T06:00:01Z
Gold set: `eval/rag_gold.jsonl` (60 questions)
Wall time: 488.5s | LLM live calls: 87 | cache hits: 2 | Spend this run: $0.0141
Spend ledger total: $0.0581 of $5.00 cap

> Status: DRAFT. The 30 qualitative reference key points and the faithfulness labels were produced by the agent and still require owner review, as does the PRD's manual spot check of 20 answers.

## Definitions
- **Numeric accuracy**: router facts equal independently computed SQL gold facts (exact counts/shares, tolerance 1e-9 for shares).
- **Hit@8**: at least one gold key point string appears in one of the 8 retrieved review snippets.
- **Faithfulness**: an LLM judge finds every claim supported by the cited snippets; judge parse failures are excluded from the denominator (`judged`).
- **Abstention**: the unanswerable question produced `abstained=true`.

## Metrics

| Set | n | Metric | Value |
| --- | --- | --- | --- |
| analytical | 20 | Numeric accuracy | 100.0% |
| analytical | 20 | SQL route rate | 100.0% |
| analytical | 20 | Latency p50 / p95 (ms) | 3065 / 15952 |
| analytical | 20 | Cost (USD) | $0.0019 |
| qualitative | 30 | Hit@8 | 96.7% |
| qualitative | 30 | Answer rate | 96.7% |
| qualitative | 30 | Faithfulness | 93.1% (29 judged) |
| qualitative | 30 | Latency p50 / p95 (ms) | 3428 / 14968 |
| qualitative | 30 | Cost (USD) | $0.0054 |
| unanswerable | 10 | Abstention accuracy | 80.0% |
| unanswerable | 10 | Latency p50 / p95 (ms) | 1267 / 5421 |
| unanswerable | 10 | Cost (USD) | $0.0018 |

## Analytical details

| id | correct | route | question |
| --- | --- | --- | --- |
| ana-01 | yes | sql | How many negative reviews are there in All_Beauty? |
| ana-02 | yes | sql | How many positive reviews are there in Amazon_Fashion? |
| ana-03 | yes | sql | How many neutral reviews are there in All_Beauty? |
| ana-04 | yes | sql | How many reviews are there in Amazon_Fashion? |
| ana-05 | yes | sql | How many negative reviews were there in 2022? |
| ana-06 | yes | sql | How many positive reviews were there in 2020? |
| ana-07 | yes | sql | How many negative reviews were there in All_Beauty in 2021? |
| ana-08 | yes | sql | Show the monthly sentiment trend in All_Beauty. |
| ana-09 | yes | sql | Show the monthly sentiment trend in Amazon_Fashion. |
| ana-10 | yes | sql | Show the sentiment trend over time in 2022. |
| ana-11 | yes | sql | Compare sentiment across categories. |
| ana-12 | yes | sql | Compare negative sentiment across categories. |
| ana-13 | yes | sql | Compare sentiment across categories in 2021. |
| ana-14 | yes | sql | What are the top aspects in negative reviews? |
| ana-15 | yes | sql | What are the top aspects in positive reviews? |
| ana-16 | yes | sql | What are the top aspects for All_Beauty? |
| ana-17 | yes | sql | What are the top aspects in negative reviews for All_Beauty? |
| ana-18 | yes | sql | Which product has the most negative reviews? |
| ana-19 | yes | sql | Which product has the most negative reviews in All_Beauty? |
| ana-20 | yes | sql | Which product has the most negative reviews in Amazon_Fashion? |

## Qualitative details

| id | hit@8 | abstained | faithful | latency ms | question |
| --- | --- | --- | --- | --- | --- |
| qual-01 | yes | False | yes | 1286 | What do customers say about battery life? |
| qual-02 | yes | False | yes | 15744 | How do customers describe the screen? |
| qual-03 | yes | True | **no** | 3780 | What do reviewers say about the camera? |
| qual-04 | yes | False | yes | 1211 | How do customers describe the shipping experience? |
| qual-05 | yes | False | yes | 3375 | Do customers think the product is good value for the price? |
| qual-06 | yes | False | yes | 3577 | What quality issues do reviewers mention? |
| qual-07 | yes | False | yes | 8893 | What do customers say about how the product fits? |
| qual-08 | yes | False | **no** | 7056 | Is the product comfortable to wear? |
| qual-09 | yes | False | yes | 8861 | How long do customers say the product lasts? |
| qual-10 | yes | False | yes | 1253 | What are customers saying about customer service? |
| qual-11 | yes | False | yes | 4124 | How do customers describe the packaging? |
| qual-12 | yes | False | yes | 13576 | What do reviewers say about sound quality? |
| qual-13 | no | False | yes | 1262 | How does the product perform in daily use? |
| qual-14 | yes | False | yes | 3244 | Is the product easy to use? |
| qual-15 | yes | False | yes | 13830 | What do customers think about the design? |
| qual-16 | yes | False | yes | 14019 | What do customers say about the companion app? |
| qual-17 | yes | False | yes | 27451 | What do customers like about their purchase? |
| qual-18 | yes | False | yes | 1446 | What negative experiences do reviewers describe? |
| qual-19 | yes | False | yes | 1711 | What do customers say about the watch strap? |
| qual-20 | yes | False | yes | 13601 | What do customers say about the effect on their skin? |
| qual-21 | yes | False | yes | 1245 | Do customers recommend the product to others? |
| qual-22 | yes | False | yes | 1318 | What do customers say about the size? |
| qual-23 | yes | False | n/a | 3911 | How do customers describe the color they received? |
| qual-24 | yes | False | yes | 4211 | What do customers say about the material? |
| qual-25 | yes | False | yes | 3422 | Do reviewers report the product breaking? |
| qual-26 | yes | False | yes | 3410 | What do customers say about the instructions? |
| qual-27 | yes | False | yes | 1795 | Do customers feel the product is worth the money? |
| qual-28 | yes | False | yes | 1229 | What do customers say about the charging cable? |
| qual-29 | yes | False | yes | 1293 | What do customers mention about gift packaging? |
| qual-30 | yes | False | yes | 3433 | What do customers say about the weight? |

## Unanswerable details

| id | abstained | latency ms | question |
| --- | --- | --- | --- |
| unans-01 | True | 1172 | What do customers say about the company's carbon-neutral shipping pledge? |
| unans-02 | False | 1233 | How do reviewers feel about the lifetime warranty policy? |
| unans-03 | True | 1181 | What do customers think about the new eco-friendly packaging initiative? |
| unans-04 | True | 3611 | Do reviewers discuss the brand's factory conditions in Vietnam? |
| unans-05 | False | 1302 | What do customers say about the subscription refill program? |
| unans-06 | True | 1185 | How do reviewers describe the augmented-reality try-on feature? |
| unans-07 | True | 1070 | What do customers think about the product's Bluetooth 6.0 support? |
| unans-08 | True | 1446 | Do reviewers mention the CEO's public statement about privacy? |
| unans-09 | True | 6884 | What do customers say about the free repair service after five years? |
| unans-10 | True | 3633 | How do reviewers rate the carbon offset program? |

## Errors

- qual-23: judge: provider failed after retries: Connection error.

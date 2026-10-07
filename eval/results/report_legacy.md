# Procurement Copilot — Evaluation Report

**Total questions evaluated:** 10

## Overall Metrics

| Metric | Value |
|--------|-------|
| Intent Routing Accuracy | 100.0% |
| Answer Rate | 100.0% |
| Avg Keyword Recall | 37.5% |
| Groundedness Rate | 70.0% |
| SQL Execution Success | 0.0% |
| Avg Latency | 162332 ms |

## Results by Category

| Category | Count | Intent Acc | Answer Rate | Keyword Recall | Grounded | Avg Latency |
|----------|-------|------------|-------------|----------------|----------|-------------|
| rag | 10 | 100.0% | 100.0% | 37.5% | 70.0% | 162332 ms |

## Sample Results

### RAG Examples

**Q (rag_001):** What does FAR 6.302-1 say about sole-source contracting?

- **Intent:** rag (correct)
- **Keyword Recall:** 75% (3/4)
- **Verified:** Yes
- **Latency:** 142479 ms
- **Answer preview:** FAR 6.302-1 permits contracting without providing for full and open competition when only one responsible source exists and no other supplies or services will satisfy the agency's requirements [1], [3...

**Q (rag_002):** When is sealed bidding required under the FAR?

- **Intent:** rag (correct)
- **Keyword Recall:** 0% (0/3)
- **Verified:** Yes
- **Latency:** 148351 ms
- **Answer preview:** Contracting officers shall solicit sealed bids if the following conditions are met: * Time permits the solicitation, submission, and evaluation of sealed bids [1]. * The award will be based on price a...

**Q (rag_003):** What are the requirements for small business set-asides under FAR Part 19?

- **Intent:** rag (correct)
- **Keyword Recall:** 0% (0/3)
- **Verified:** No
- **Latency:** 173051 ms
- **Answer preview:** I cannot provide a verified answer to this question. The available evidence does not sufficiently support a response.

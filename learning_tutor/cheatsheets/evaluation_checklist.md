# RAG Evaluation Pre-Flight Checklist

A comprehensive checklist before running evaluations or deploying a RAG system.

## Dataset Readiness

- [ ] **Minimum 20 test cases** covering all major query categories
- [ ] **Ground truth references** written by domain experts (not the LLM)
- [ ] **Edge cases included**: empty context, unanswerable questions, multi-hop queries
- [ ] **Category labels** assigned to every test case (returns, shipping, products, etc.)
- [ ] **No data leakage**: test queries not present in the knowledge base verbatim
- [ ] **Synthetic test cases** generated to augment manual ones (DeepEval Synthesizer or RAGAS)
- [ ] **Dataset versioned** and saved to a reproducible location (JSON/CSV in repo or artifact store)

## Metric Selection

- [ ] **Faithfulness metric** enabled (answer grounded in context?)
- [ ] **Answer Relevancy metric** enabled (answer addresses the question?)
- [ ] **Context Recall metric** enabled (retriever found the needed info?)
- [ ] **Context Precision metric** enabled if using reranker or caring about rank order
- [ ] **At least one custom metric** (G-Eval) for domain-specific quality criteria
- [ ] **Thresholds defined** for every metric (e.g., faithfulness >= 0.7)
- [ ] **Both frameworks** (DeepEval + RAGAS) used for cross-validation on critical metrics

## Baseline Setup

- [ ] **Baseline scores recorded** from initial evaluation run
- [ ] **Baseline saved** to a versioned file (baseline_scores.json)
- [ ] **Regression threshold** defined (e.g., 0.05 maximum allowed decrease)
- [ ] **Pipeline config recorded** with baseline (chunk_size, top_k, model, temperature)
- [ ] **Baseline re-run quarterly** or after major knowledge base updates

## Pipeline Configuration

- [ ] **Chunk size** tested across at least 3 values (e.g., 256, 512, 1024)
- [ ] **Top-K** tested across at least 3 values (e.g., 3, 5, 10)
- [ ] **Temperature** set to 0.0 for factual accuracy (tested if non-zero is needed)
- [ ] **System prompt** instructs the LLM to stay grounded in context
- [ ] **Embedding model** selected and documented
- [ ] **Reranker** evaluated for precision improvement

## CI/CD Integration

- [ ] **GitHub Actions workflow** runs evaluation on PRs touching pipeline code
- [ ] **API key** stored as GitHub Secret (OPENAI_API_KEY)
- [ ] **Threshold check** fails the CI job if scores drop below minimums
- [ ] **Evaluation results** uploaded as CI artifacts for review
- [ ] **Tests run without API keys** (unit tests mock external calls)
- [ ] **Evaluation subset** used in CI for speed (full eval runs nightly)

## Production Monitoring

- [ ] **Logging** captures query, response, contexts, and latency for every request
- [ ] **Sampling strategy** defined for production evaluation (e.g., 5% of queries)
- [ ] **Latency SLO** defined and monitored (e.g., p95 < 3 seconds)
- [ ] **Alerting** configured for metric regressions (PagerDuty, Slack, etc.)
- [ ] **Dashboard** shows metric trends over time (Streamlit, Grafana, etc.)
- [ ] **Feedback loop** collects user thumbs-up/down for correlation with metrics
- [ ] **Knowledge base freshness** monitored (stale docs degrade recall)

## Safety and Quality

- [ ] **Bias metric** evaluated if system serves diverse user populations
- [ ] **Toxicity metric** evaluated for user-facing applications
- [ ] **PII handling** tested (system does not leak sensitive data from context)
- [ ] **Refusal behavior** tested (system says "I don't know" when context is insufficient)
- [ ] **Adversarial queries** tested (prompt injection, jailbreak attempts)
- [ ] **Multi-language** tested if system supports non-English queries

## Documentation

- [ ] **Evaluation report** generated and shared with stakeholders
- [ ] **Metric definitions** documented for the team
- [ ] **Runbook** for investigating metric regressions
- [ ] **Change log** updated when pipeline config changes

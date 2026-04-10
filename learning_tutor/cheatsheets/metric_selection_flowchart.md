# Metric Selection Flowchart

A one-page decision tree for choosing the right RAG evaluation metrics.

## Quick-Start Flowchart (ASCII)

```
                        What are you evaluating?
                                |
                +---------------+----------------+
                |                                |
           RETRIEVER                         GENERATOR
                |                                |
     +----------+----------+          +----------+----------+
     |          |          |          |          |          |
  Relevance  Precision   Recall   Faithfulness Relevancy  Quality
     |          |          |          |          |          |
  "Are the   "Are top   "Did we   "Is answer  "Does it   "Is it
   chunks     results    find ALL   grounded    answer     well-
   useful?"   clean?"    the info?" in ctx?"    the Q?"    written?"
     |          |          |          |          |          |
  Ctx.Rel.  Ctx.Prec.  Ctx.Recall  Faithful.  Ans.Rel.   G-Eval
  (DE+RAGAS) (DE+RAGAS) (DE+RAGAS) (DE+RAGAS) (DE+RAGAS) (DE)
```

## Symptom --> Metric --> Fix

| Symptom | Primary Metric | Framework | Fix |
|---------|---------------|-----------|-----|
| Answer contains facts not in context | Faithfulness | DE + RAGAS | Lower temperature; strengthen system prompt |
| Answer misses key information | Contextual Recall | DE + RAGAS | Increase top-K; improve chunking |
| Retrieved chunks are irrelevant | Contextual Relevancy | DE + RAGAS | Better embeddings; add reranker |
| Relevant chunk retrieved but ranked low | Contextual Precision | DE + RAGAS | Add reranker (Cohere, ColBERT) |
| Answer does not address the question | Answer Relevancy | DE + RAGAS | Improve prompt; raise temperature slightly |
| Answer is verbose or poorly structured | G-Eval (custom) | DE | Add formatting instructions to prompt |
| Answer hallucinates specific numbers | Hallucination metric | DE | Ground numeric claims; use temperature=0 |
| Different runs give different scores | Run 3x and average | Both | Use temperature=0 for judge LLM |

## When to Use Each Metric

### Must-Have (Every Evaluation)
- **Faithfulness** -- Is the answer grounded in retrieved context?
- **Answer Relevancy** -- Does the answer address the question?
- **Contextual Recall** -- Did retrieval find the needed information?

### Recommended (Production Systems)
- **Contextual Precision** -- Are the most relevant chunks ranked highest?
- **Contextual Relevancy** -- Are retrieved chunks actually useful?
- **Latency** -- Is the pipeline fast enough? (deterministic, no LLM needed)

### Specialized (When Needed)
- **G-Eval** -- Custom quality criteria (tone, format, domain terms)
- **Hallucination** -- Binary grounding check (stricter than faithfulness)
- **Bias / Toxicity** -- Safety evaluation for user-facing systems
- **Tool Use Correctness** -- Agentic RAG: did it call the right tools?

## Framework Feature Comparison

| Feature | DeepEval | RAGAS |
|---------|----------|-------|
| Faithfulness | FaithfulnessMetric | Faithfulness |
| Answer Relevancy | AnswerRelevancyMetric | ResponseRelevancy |
| Context Precision | ContextualPrecisionMetric | LLMContextPrecisionWithReference |
| Context Recall | ContextualRecallMetric | LLMContextRecall |
| Context Relevancy | ContextualRelevancyMetric | (use Precision) |
| Custom criteria | GEval | SingleTurnSample + custom |
| Pytest native | Yes | No |
| Async support | Yes | Yes |
| Cost tracking | Built-in | Manual |

## Decision Rules

1. **Start with faithfulness + answer relevancy + context recall** -- these catch 80% of issues.
2. **Add context precision** when you have a reranker or care about ranking quality.
3. **Add G-Eval** when you need domain-specific criteria (enterprise tone, legal accuracy, etc.).
4. **Use both frameworks** to cross-validate -- if they disagree, investigate why.
5. **Automate in CI/CD** with thresholds: faithfulness >= 0.7, recall >= 0.6, relevancy >= 0.7.

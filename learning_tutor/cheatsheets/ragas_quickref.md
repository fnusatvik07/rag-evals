# RAGAS Quick Reference

One-page reference for RAGAS v0.2-0.4 RAG evaluation.

## Installation

```bash
pip install ragas
```

## Core Imports (v0.2+)

```python
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    FactualCorrectness,
    SemanticSimilarity,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
```

## LLM and Embedding Configuration

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Wrap LangChain models for RAGAS
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# Pass to metrics
faithfulness = Faithfulness(llm=ragas_llm)
relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
```

## SingleTurnSample Fields

```python
sample = SingleTurnSample(
    user_input="user question",                          # Required
    response="LLM response",                             # Required
    reference="ground truth answer",                     # For recall, precision
    retrieved_contexts=["chunk1", "chunk2"],              # For faithfulness, ctx metrics
    reference_contexts=["gold_ctx1", "gold_ctx2"],       # For non-LLM metrics
)
```

| Field | Used By |
|-------|---------|
| `user_input` | All metrics |
| `response` | All metrics |
| `reference` | LLMContextRecall, LLMContextPrecisionWithReference, FactualCorrectness |
| `retrieved_contexts` | Faithfulness, ResponseRelevancy, context metrics |
| `reference_contexts` | NonLLMContextPrecision, NonLLMContextRecall |

## Available Metrics

### LLM-Based (Require judge LLM)
```python
Faithfulness(llm=ragas_llm)                               # Is answer grounded in context?
ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb)     # Does answer address the question?
LLMContextPrecisionWithReference(llm=ragas_llm)            # Are relevant chunks ranked first?
LLMContextRecall(llm=ragas_llm)                            # Did retrieval find needed info?
FactualCorrectness(llm=ragas_llm)                          # Claims match reference?
```

### Non-LLM (Deterministic, no API cost)
```python
NonLLMContextPrecisionWithReference()    # Precision using string matching
NonLLMContextRecall()                    # Recall using string matching
SemanticSimilarity(embeddings=ragas_emb) # Cosine similarity of response vs reference
```

## Running Evaluation

```python
# Build dataset
samples = [
    SingleTurnSample(
        user_input=q, response=r, reference=ref,
        retrieved_contexts=ctxs,
    )
    for q, r, ref, ctxs in data
]
dataset = EvaluationDataset(samples=samples)

# Define metrics
metrics = [
    Faithfulness(llm=ragas_llm),
    ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    LLMContextRecall(llm=ragas_llm),
]

# Evaluate
results = evaluate(dataset=dataset, metrics=metrics)

# Access results
print(results)                    # Aggregate scores dict
df = results.to_pandas()          # Per-sample DataFrame
```

## Single Sample Scoring

```python
scorer = Faithfulness(llm=ragas_llm)
sample = SingleTurnSample(
    user_input="question",
    response="answer",
    retrieved_contexts=["ctx1", "ctx2"],
)
score = await scorer.single_turn_ascore(sample)
```

## Multi-Turn Evaluation (v0.3+)

```python
from ragas import MultiTurnSample
from ragas.metrics import AgentGoalAccuracyWithReference

sample = MultiTurnSample(
    user_input=[...],   # List of conversation turns
    reference="expected outcome description",
)
```

## Integration with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."

# RAGAS automatically logs traces when LangSmith env vars are set
results = evaluate(dataset=dataset, metrics=metrics)
```

## Gotchas

| Gotcha | Solution |
|--------|----------|
| `retrieved_contexts` must be `list[str]` | Convert Document objects: `[doc.page_content for doc in docs]` |
| ResponseRelevancy needs embeddings | Pass both `llm` and `embeddings` params |
| RAGAS v0.1 vs v0.2+ API differs | v0.2+ uses `SingleTurnSample`; v0.1 used `Dataset` from HF |
| Scores are NaN | Check that required fields are not empty or None |
| LLMContextRecall needs `reference` | Must provide ground truth reference answer |
| Slow on large datasets | Use `batch_size` param or evaluate in chunks |
| Cost management | Use `gpt-4o-mini` as judge; track token usage manually |
| Reproducibility | Set `temperature=0` on the judge LLM |
| `evaluate()` returns aggregate | Use `.to_pandas()` to get per-sample scores |
| Import errors across versions | Pin version: `pip install ragas==0.2.6` |

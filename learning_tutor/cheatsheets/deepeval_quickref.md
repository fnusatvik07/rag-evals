# DeepEval Quick Reference

One-page reference for DeepEval v3.x RAG evaluation.

## Installation

```bash
pip install deepeval
```

## Core Imports

```python
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    GEval,
    SummarizationMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate, assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
```

## LLMTestCase Fields

```python
tc = LLMTestCase(
    input="user question",                         # Required
    actual_output="LLM response",                  # Required
    expected_output="ground truth answer",          # For recall, G-Eval
    retrieval_context=["chunk1", "chunk2"],         # For faithfulness, ctx metrics
    context=["gold context1", "gold context2"],     # For hallucination
)
```

| Field | Used By |
|-------|---------|
| `input` | All metrics |
| `actual_output` | All metrics |
| `expected_output` | ContextualRecall, ContextualPrecision, GEval |
| `retrieval_context` | Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall |
| `context` | Hallucination (gold context, not retrieved) |

## Metric Constructors

```python
# All metrics accept: model (str), threshold (float)
FaithfulnessMetric(model="gpt-4o-mini", threshold=0.7)
AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7)
ContextualPrecisionMetric(model="gpt-4o-mini", threshold=0.7)
ContextualRecallMetric(model="gpt-4o-mini", threshold=0.7)
ContextualRelevancyMetric(model="gpt-4o-mini", threshold=0.7)
HallucinationMetric(model="gpt-4o-mini", threshold=0.5)
BiasMetric(model="gpt-4o-mini", threshold=0.5)
ToxicityMetric(model="gpt-4o-mini", threshold=0.5)
```

## G-Eval (Custom Criteria)

```python
quality = GEval(
    name="Domain Quality",
    model="gpt-4o-mini",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "Check if response uses proper business terminology.",
        "Verify specific numbers and names are included.",
        "Ensure clarity and usefulness for the customer.",
    ],
    threshold=0.7,
)
```

## Measuring a Single Test Case

```python
metric = FaithfulnessMetric(model="gpt-4o-mini", threshold=0.7)
metric.measure(test_case)
print(metric.score)    # 0.0 - 1.0
print(metric.reason)   # Explanation string
print(metric.is_successful())  # True/False based on threshold
```

## Batch Evaluation with evaluate()

```python
from deepeval import evaluate

results = evaluate(
    test_cases=[tc1, tc2, tc3],
    metrics=[faithfulness, relevancy, recall],
)
# Returns list of TestResult objects
```

## Pytest Integration with assert_test()

```python
import pytest
from deepeval import assert_test

@pytest.mark.parametrize("test_case", test_cases)
def test_rag_pipeline(test_case):
    assert_test(test_case, [faithfulness, relevancy])
```

Run with: `deepeval test run test_file.py`

## Synthetic Data Generation

```python
synthesizer = Synthesizer(model="gpt-4o-mini")
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=[["doc text 1"], ["doc text 2"]],
    num_goldens_per_context=2,
)
```

## CLI Commands

```bash
deepeval test run tests/           # Run evaluation tests
deepeval test run tests/ -v        # Verbose output
deepeval login                     # Connect to Confident AI dashboard
deepeval test run tests/ -k "test_faithfulness"  # Run specific test
```

## Gotchas

| Gotcha | Solution |
|--------|----------|
| `retrieval_context` vs `context` | `retrieval_context` = what your RAG retrieved; `context` = gold/ideal context |
| Score is 0 for faithfulness | Check that `retrieval_context` is not empty |
| GEval scores vary between runs | Use `temperature=0` on the judge model; run 3x and average |
| ContextualRecall needs `expected_output` | Must provide ground truth reference answer |
| ContextualPrecision needs `expected_output` | Must provide ground truth reference answer |
| `model` param accepts strings | Use `"gpt-4o-mini"`, `"gpt-4o"`, or any OpenAI-compatible model |
| Costs add up quickly | Use `gpt-4o-mini` for development; `gpt-4o` for final eval |
| Async evaluation | Use `a_measure()` for async: `await metric.a_measure(tc)` |
| Timeout on large contexts | Chunk contexts to < 4000 tokens each |

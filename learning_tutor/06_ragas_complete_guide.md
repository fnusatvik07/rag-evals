# 06 -- RAGAS Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation and Setup](#installation-and-setup)
3. [Core Concepts (RAGAS 0.2.x API)](#core-concepts-ragas-02x-api)
4. [All RAGAS Metrics](#all-ragas-metrics)
5. [Component-Level Metrics (Retriever)](#component-level-metrics-retriever)
6. [Component-Level Metrics (Generator)](#component-level-metrics-generator)
7. [End-to-End Metrics](#end-to-end-metrics)
8. [Non-LLM Metrics](#non-llm-metrics)
9. [Agentic / Multi-Turn Metrics](#agentic--multi-turn-metrics)
10. [Integration with LangChain](#integration-with-langchain)
11. [Integration with LlamaIndex](#integration-with-llamaindex)
12. [Integration with Custom Pipelines](#integration-with-custom-pipelines)
13. [Test Set Generation](#test-set-generation)
14. [Custom Metrics](#custom-metrics)
15. [Configuration](#configuration)
16. [Comparison with DeepEval](#comparison-with-deepeval)

---

## Overview

### What is RAGAS?

RAGAS stands for **Retrieval Augmented Generation Assessment**. It is an open-source framework
purpose-built for evaluating Retrieval Augmented Generation (RAG) pipelines. Unlike general-purpose
LLM evaluation tools, RAGAS was designed from the ground up to measure the specific failure modes
that plague RAG systems: unfaithful generation, irrelevant retrieval, imprecise context ranking,
and incomplete context coverage.

### Origins

RAGAS was created by the **Explodinggradients** team and introduced in a research paper titled
"RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023). The paper proposed a set
of metrics that evaluate RAG pipelines without requiring human-annotated ground truth for every
data point, leveraging LLMs as evaluators (LLM-as-judge) combined with classical NLP techniques
like Natural Language Inference (NLI) and embedding similarity.

Key insight from the paper: RAG pipelines have distinct components (retriever and generator) that
can fail independently, so evaluation must decompose along those same lines.

### GitHub and Community

- **Repository**: [explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- **Documentation**: [docs.ragas.io](https://docs.ragas.io)
- **License**: Apache 2.0
- **Stars**: 8,000+ (as of early 2025)
- **Active development**: Version 0.2.x represents a major API overhaul from 0.1.x

### Version History (Important)

RAGAS underwent a **significant API redesign** between 0.1.x and 0.2.x:

| Aspect | 0.1.x (Legacy) | 0.2.x (Current) |
|--------|-----------------|------------------|
| Data unit | `Dataset` (HuggingFace-style) | `SingleTurnSample` / `MultiTurnSample` |
| Dataset | HuggingFace `Dataset` | `EvaluationDataset` |
| Field names | `question`, `answer`, `contexts`, `ground_truth` | `user_input`, `response`, `retrieved_contexts`, `reference` |
| Metric classes | Functional style | Object-oriented with `SingleTurnMetric` / `MultiTurnMetric` |
| LLM config | `langchain_llm` param | `LLMFactory` / `LangchainLLMWrapper` |
| Embeddings | `langchain_embeddings` | `LangchainEmbeddingsWrapper` |

**This guide covers the 0.2.x API exclusively.** If you see code using `question`, `answer`,
`contexts`, or `ground_truth` fields, that is the old 0.1.x API.

---

## Installation and Setup

### Basic Installation

```bash
pip install ragas
```

### With Optional Dependencies

```bash
# For LangChain integration
pip install ragas[langchain]

# For LlamaIndex integration
pip install ragas[llama_index]

# All optional dependencies
pip install ragas[all]
```

### Verify Installation

```python
import ragas
print(ragas.__version__)  # Should show 0.2.x
```

### Required Environment Variables

RAGAS uses LLMs for evaluation. By default, it uses OpenAI models:

```bash
export OPENAI_API_KEY="sk-..."
```

For other providers, you configure LLMs via LangChain wrappers (see [Configuration](#configuration)).

### Minimal Working Example

```python
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# Create a sample
sample = SingleTurnSample(
    user_input="What is the capital of France?",
    response="The capital of France is Paris. It is known for the Eiffel Tower.",
    retrieved_contexts=[
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris."
    ],
    reference="Paris is the capital of France."
)

# Create dataset
dataset = EvaluationDataset(samples=[sample])

# Evaluate
results = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()]
)

print(results)
# Output: {'faithfulness': 1.0, 'answer_relevancy': 0.95}
```

---

## Core Concepts (RAGAS 0.2.x API)

### SingleTurnSample

`SingleTurnSample` is the fundamental evaluation unit in RAGAS 0.2.x. It represents a single
interaction in a RAG pipeline: one question, one retrieval, one response.

```python
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    user_input="What causes rain?",              # The user's question/query
    response="Rain is caused by condensation.",   # The LLM's generated answer
    reference="Rain occurs when water vapor...",  # Ground truth / reference answer
    retrieved_contexts=[                          # What the retriever returned
        "Water vapor rises and condenses...",
        "Clouds form when air cools..."
    ],
    reference_contexts=[                          # Ideal contexts (for context metrics)
        "Water vapor condenses into droplets..."
    ],
    rubric={                                      # Optional scoring rubric
        "score1_description": "Completely wrong",
        "score5_description": "Perfectly correct"
    }
)
```

#### All Fields

| Field | Type | Required | Used By |
|-------|------|----------|---------|
| `user_input` | `str` | Most metrics | The query/question asked by the user |
| `response` | `str` | Generator metrics | The LLM's generated response |
| `reference` | `str` | Reference-based metrics | Ground truth / expected answer |
| `retrieved_contexts` | `List[str]` | Retriever + faithfulness metrics | Contexts returned by the retriever |
| `reference_contexts` | `List[str]` | Context precision/recall with reference | Ideal contexts that should be retrieved |
| `rubric` | `Dict[str, str]` | Rubric-based metrics | Custom scoring rubric |

**Important**: Not all fields are required for every metric. Each metric specifies which fields
it needs. If a required field is missing, RAGAS will raise a validation error.

### MultiTurnSample

For conversational / agentic RAG evaluation:

```python
from ragas.dataset_schema import MultiTurnSample, Message

sample = MultiTurnSample(
    user_input=[
        Message(content="What is machine learning?", role="user"),
        Message(content="Machine learning is a subset of AI...", role="assistant"),
        Message(content="How does it differ from deep learning?", role="user"),
        Message(content="Deep learning uses neural networks...", role="assistant"),
    ],
    reference="Deep learning is a subset of ML that uses multi-layer neural networks..."
)
```

### EvaluationDataset

A collection of samples to evaluate:

```python
from ragas.dataset_schema import EvaluationDataset

dataset = EvaluationDataset(samples=[sample1, sample2, sample3])

# From a list of dictionaries
dataset = EvaluationDataset.from_list([
    {
        "user_input": "What is Python?",
        "response": "Python is a programming language.",
        "retrieved_contexts": ["Python is a high-level programming language..."],
        "reference": "Python is a high-level, interpreted programming language."
    }
])

# From a pandas DataFrame
import pandas as pd
df = pd.DataFrame([...])
dataset = EvaluationDataset.from_pandas(df)

# From HuggingFace Dataset
from datasets import Dataset
hf_dataset = Dataset.from_dict({...})
dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
```

### The evaluate() Function

The main entry point for running evaluations:

```python
from ragas import evaluate

results = evaluate(
    dataset=dataset,                    # EvaluationDataset
    metrics=[                           # List of metric instances
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm=llm_instance,                  # Optional: override default LLM
    embeddings=embeddings_instance,     # Optional: override default embeddings
    run_config=run_config,              # Optional: execution configuration
    raise_exceptions=False,             # Whether to raise or log errors
    show_progress=True                  # Show progress bar
)

# Access results
print(results)                          # Aggregated scores
print(results.to_pandas())             # Per-sample scores as DataFrame
print(results.scores)                  # List of per-sample score dicts
```

### RunConfig

Controls execution behavior:

```python
from ragas.run_config import RunConfig

run_config = RunConfig(
    timeout=120,            # Timeout per LLM call in seconds
    max_retries=10,         # Max retries on failure
    max_wait=60,            # Max wait between retries
    max_workers=16,         # Parallel workers for async execution
    log_tenacity=True,      # Log retry attempts
    seed=42                 # Random seed for reproducibility
)

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config
)
```

---

## All RAGAS Metrics

### Metric Architecture in 0.2.x

All RAGAS metrics inherit from base classes:

```
Metric (base)
├── SingleTurnMetric
│   ├── MetricWithLLM
│   └── MetricWithEmbeddings
└── MultiTurnMetric
```

Each metric declares:
- **Required fields**: Which `SingleTurnSample` / `MultiTurnSample` fields it needs
- **Score range**: Typically 0.0 to 1.0 (higher is better)
- **Dependencies**: Whether it needs an LLM, embeddings, or both

### Quick Reference Table

| Metric | Type | Needs LLM | Needs Embeddings | Needs Reference | Key Input Fields |
|--------|------|-----------|-------------------|-----------------|------------------|
| Faithfulness | Generator | Yes | No | No | response, retrieved_contexts |
| AnswerRelevancy | Generator | Yes | Yes | No | user_input, response, retrieved_contexts |
| ContextPrecision | Retriever | Yes | No | Yes | user_input, retrieved_contexts, reference |
| ContextRecall | Retriever | Yes | No | Yes | retrieved_contexts, reference |
| ContextEntityRecall | Retriever | No | No | Yes | retrieved_contexts, reference |
| NoiseSensitivity | Generator | Yes | No | Yes | user_input, response, reference, retrieved_contexts |
| ResponseRelevancy | Generator | Yes | Yes | No | user_input, response |
| LLMContextPrecisionWithReference | Retriever | Yes | No | Yes | retrieved_contexts, reference |
| LLMContextPrecisionWithoutReference | Retriever | Yes | No | No | user_input, response, retrieved_contexts |
| LLMContextRecall | Retriever | Yes | No | Yes | retrieved_contexts, reference |
| FactualCorrectness | Generator | Yes | No | Yes | response, reference |
| SemanticSimilarity | Generator | No | Yes | Yes | response, reference |
| AnswerCorrectness | Generator | Yes | Yes | Yes | response, reference |
| NonLLMStringSimilarity | Generator | No | No | Yes | response, reference |
| ExactMatch | Generator | No | No | Yes | response, reference |
| StringPresence | Generator | No | No | Yes | response, reference |
| RougeScore | Generator | No | No | Yes | response, reference |
| BleuScore | Generator | No | No | Yes | response, reference |

---

## Component-Level Metrics (Retriever)

### 1. Context Precision

**What it measures**: Whether relevant items in the retrieved contexts are ranked higher than
irrelevant ones. Evaluates the retriever's ranking quality.

**How it works**:
1. For each retrieved context chunk, an LLM judges whether it is relevant to the `user_input`
   given the `reference` answer
2. Each chunk gets a binary relevance label (1 = relevant, 0 = irrelevant)
3. Precision@K is computed at each position K
4. The final score is the average precision across all relevant items

**Formula**:

```
Context Precision = (1/|relevant items|) * Sum_k(Precision@k * relevance_k)
```

Where:
- `Precision@k = (number of relevant items up to position k) / k`
- `relevance_k` is 1 if the item at position k is relevant, 0 otherwise

**Score range**: 0.0 to 1.0 (1.0 = all relevant contexts ranked at the top)

**Required fields**: `user_input`, `retrieved_contexts`, `reference`

```python
from ragas.metrics import ContextPrecision

metric = ContextPrecision()

sample = SingleTurnSample(
    user_input="What is photosynthesis?",
    retrieved_contexts=[
        "Photosynthesis is the process by which plants convert light energy to chemical energy.",
        "The weather today is sunny with a high of 75 degrees.",
        "Chlorophyll is the pigment that captures light energy in plants."
    ],
    reference="Photosynthesis is the process plants use to convert light energy into glucose."
)

score = await metric.single_turn_ascore(sample)
# If relevant contexts are [0, 2] (positions 1 and 3), precision penalizes the
# irrelevant context at position 2 appearing before the relevant context at position 3.
```

### 2. Context Recall

**What it measures**: How much of the reference answer can be attributed to the retrieved contexts.
A low score means the retriever failed to fetch relevant information.

**How it works**:
1. The reference answer is decomposed into individual statements/claims
2. An LLM checks whether each statement can be attributed to one or more retrieved context chunks
3. The score is the fraction of attributable statements

**Formula**:

```
Context Recall = |reference statements attributable to contexts| / |total reference statements|
```

**Score range**: 0.0 to 1.0 (1.0 = every reference statement is supported by retrieved contexts)

**Required fields**: `retrieved_contexts`, `reference`

```python
from ragas.metrics import ContextRecall

metric = ContextRecall()

sample = SingleTurnSample(
    user_input="What are the benefits of exercise?",
    retrieved_contexts=[
        "Regular exercise improves cardiovascular health and reduces risk of heart disease.",
        "Physical activity helps maintain healthy body weight."
    ],
    reference="Exercise improves heart health, helps with weight management, boosts mood, and strengthens bones."
)

score = await metric.single_turn_ascore(sample)
# Reference has 4 claims. If contexts support 2 of them (heart health, weight),
# score = 2/4 = 0.5. The retriever missed mood and bone benefits.
```

### 3. Context Entity Recall

**What it measures**: The overlap of named entities between the reference and the retrieved contexts.
This is a simpler, non-LLM metric that uses NER (Named Entity Recognition).

**How it works**:
1. Extract named entities from the `reference` answer
2. Extract named entities from all `retrieved_contexts`
3. Compute the recall: what fraction of reference entities appear in the contexts

**Formula**:

```
Context Entity Recall = |entities_in_reference INTERSECT entities_in_contexts| / |entities_in_reference|
```

**Score range**: 0.0 to 1.0

**Required fields**: `retrieved_contexts`, `reference`

**Key advantage**: Does NOT require an LLM, so it is fast and cheap.

```python
from ragas.metrics import ContextEntityRecall

metric = ContextEntityRecall()

sample = SingleTurnSample(
    user_input="Who founded Tesla?",
    retrieved_contexts=[
        "Tesla, Inc. was founded by Martin Eberhard and Marc Tarpenning in 2003.",
        "Elon Musk joined Tesla as chairman of the board in 2004."
    ],
    reference="Tesla was founded by Martin Eberhard and Marc Tarpenning. Elon Musk later became CEO."
)

score = await metric.single_turn_ascore(sample)
# Entities in reference: {Tesla, Martin Eberhard, Marc Tarpenning, Elon Musk}
# Entities in contexts: {Tesla, Martin Eberhard, Marc Tarpenning, Elon Musk}
# All present => score = 1.0
```

### 4. LLM Context Precision (With Reference)

**What it measures**: Similar to Context Precision but uses an LLM to judge relevance of each
retrieved context based on the reference answer (not the user input).

**How it works**:
1. An LLM evaluates each retrieved context against the `reference`
2. Binary relevance labels are assigned
3. Average precision is computed

**Required fields**: `retrieved_contexts`, `reference`

```python
from ragas.metrics import LLMContextPrecisionWithReference

metric = LLMContextPrecisionWithReference()
score = await metric.single_turn_ascore(sample)
```

### 5. LLM Context Precision (Without Reference)

**What it measures**: Precision of context ranking without needing a reference answer. Uses the
LLM's own judgment of whether each context is useful for answering the query.

**Required fields**: `user_input`, `response`, `retrieved_contexts`

```python
from ragas.metrics import LLMContextPrecisionWithoutReference

metric = LLMContextPrecisionWithoutReference()
score = await metric.single_turn_ascore(sample)
```

### 6. LLM Context Recall

**What it measures**: Like Context Recall but with enhanced LLM-based attribution checking.

**Required fields**: `retrieved_contexts`, `reference`

```python
from ragas.metrics import LLMContextRecall

metric = LLMContextRecall()
score = await metric.single_turn_ascore(sample)
```

### 7. Noise Sensitivity

**What it measures**: How much the presence of irrelevant (noisy) contexts affects the generated
response. A high noise sensitivity score indicates the generator is easily misled by irrelevant
context.

**How it works**:
1. Compares the response against the reference to identify correct and incorrect claims
2. Checks whether incorrect claims can be attributed to irrelevant (noisy) contexts
3. Computes the fraction of response claims that were influenced by noise

**Required fields**: `user_input`, `response`, `reference`, `retrieved_contexts`

```python
from ragas.metrics import NoiseSensitivity

metric = NoiseSensitivity()
score = await metric.single_turn_ascore(sample)
# Lower is better for this metric — a low score means the generator
# is robust against noisy contexts
```

---

## Component-Level Metrics (Generator)

### 8. Faithfulness

**What it measures**: Whether the claims in the generated response are supported by the retrieved
contexts. This is the primary "grounding" or "hallucination detection" metric.

**How it works (3-step algorithm)**:
1. **Claim extraction**: An LLM breaks the `response` into individual claims/statements
2. **NLI verification**: For each claim, an LLM performs Natural Language Inference to check
   whether the claim is supported by, contradicted by, or neutral with respect to the
   `retrieved_contexts`
3. **Score computation**: The fraction of claims that are supported

**Formula**:

```
Faithfulness = |claims supported by contexts| / |total claims in response|
```

**Score range**: 0.0 to 1.0 (1.0 = every claim is grounded in the context)

**Required fields**: `user_input`, `response`, `retrieved_contexts`

**Does NOT require**: `reference` (this is reference-free)

```python
from ragas.metrics import Faithfulness

metric = Faithfulness()

sample = SingleTurnSample(
    user_input="What is the speed of light?",
    response="The speed of light is approximately 300,000 km/s. It was first measured by Einstein.",
    retrieved_contexts=[
        "The speed of light in vacuum is approximately 299,792 km/s.",
        "Ole Roemer first estimated the speed of light in 1676."
    ]
)

score = await metric.single_turn_ascore(sample)
# Claim 1: "speed of light is ~300,000 km/s" -> Supported (close enough to 299,792)
# Claim 2: "first measured by Einstein" -> NOT supported (it was Roemer)
# Score = 1/2 = 0.5
```

**Why Faithfulness matters**: This is arguably the most important RAG metric. A RAG system that
generates unfaithful responses is actively harmful — it provides users with confident-sounding
answers that are not grounded in the retrieved evidence. This is worse than saying "I don't know."

### 9. Answer Relevancy

**What it measures**: Whether the response is relevant to the user's question. Penalizes responses
that are off-topic or contain excessive irrelevant information.

**How it works (reverse question generation)**:
1. An LLM generates N questions from the `response` (hypothetical questions the response would answer)
2. Each generated question is compared to the original `user_input` using embedding cosine similarity
3. The score is the mean similarity across all generated questions

**Formula**:

```
Answer Relevancy = mean(cosine_similarity(embedding(q_i), embedding(user_input)))
```

Where `q_i` are questions generated from the response.

**Score range**: 0.0 to 1.0

**Required fields**: `user_input`, `response`, `retrieved_contexts`

**Needs**: Both LLM (for question generation) and Embeddings (for similarity)

```python
from ragas.metrics import AnswerRelevancy

metric = AnswerRelevancy()

sample = SingleTurnSample(
    user_input="What is the capital of France?",
    response="The capital of France is Paris. Paris has a population of about 2 million. The Eiffel Tower is 330m tall. French cuisine includes croissants.",
    retrieved_contexts=["Paris is the capital of France with about 2 million inhabitants."]
)

score = await metric.single_turn_ascore(sample)
# The response contains relevant info (capital is Paris) but also tangential info
# (Eiffel Tower height, cuisine). Score will be moderate, not perfect.
```

**Design insight**: The reverse question generation approach is clever because it measures relevancy
without needing a reference answer. If the response would primarily answer questions very similar to
the original, it is relevant. If it would answer very different questions, it contains off-topic content.

### 10. Response Relevancy

**What it measures**: Similar to Answer Relevancy but may use a different algorithm depending on
the RAGAS version. In some versions, this is an alias or a variant that focuses purely on whether
the response addresses the query without evaluating context.

**Required fields**: `user_input`, `response`

```python
from ragas.metrics import ResponseRelevancy

metric = ResponseRelevancy()
score = await metric.single_turn_ascore(sample)
```

---

## End-to-End Metrics

### 11. Factual Correctness

**What it measures**: Whether the claims in the response are factually correct compared to the
reference answer. This is an end-to-end metric that does not consider the retrieval step.

**How it works**:
1. Extract claims from the `response`
2. Extract claims from the `reference`
3. Classify each response claim as:
   - **True Positive (TP)**: Claim in response that matches a reference claim
   - **False Positive (FP)**: Claim in response that contradicts or is absent from reference
   - **False Negative (FN)**: Claim in reference that is missing from response
4. Compute F1-like score

**Formula**:

```
Factual Correctness = TP / (TP + 0.5 * (FP + FN))    # F1 score
```

Or configurable to use precision or recall instead of F1.

**Score range**: 0.0 to 1.0

**Required fields**: `response`, `reference`

```python
from ragas.metrics import FactualCorrectness

metric = FactualCorrectness()
# Can configure the mode:
metric = FactualCorrectness(mode="precision")   # Only penalize wrong claims
metric = FactualCorrectness(mode="recall")      # Only penalize missing claims
metric = FactualCorrectness(mode="f1")          # Balance both (default)

sample = SingleTurnSample(
    user_input="List the planets in our solar system",
    response="The planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    reference="The eight planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
)

score = await metric.single_turn_ascore(sample)
# All claims match => score ~ 1.0
```

### 12. Semantic Similarity

**What it measures**: The embedding-based cosine similarity between the response and the reference.
This is a simple, fast metric that does not require an LLM.

**How it works**:
1. Encode `response` using an embedding model
2. Encode `reference` using the same embedding model
3. Compute cosine similarity

**Formula**:

```
Semantic Similarity = cosine_similarity(embed(response), embed(reference))
```

**Score range**: 0.0 to 1.0 (can technically be negative but rare in practice)

**Required fields**: `response`, `reference`

**Needs**: Embeddings only (no LLM)

```python
from ragas.metrics import SemanticSimilarity

metric = SemanticSimilarity()

sample = SingleTurnSample(
    user_input="What is gravity?",
    response="Gravity is a force that attracts objects with mass toward each other.",
    reference="Gravity is the natural force of attraction between physical objects with mass."
)

score = await metric.single_turn_ascore(sample)
# High similarity since both describe gravity similarly
```

### 13. Answer Correctness

**What it measures**: A hybrid metric that combines factual correctness with semantic similarity
to give a comprehensive answer quality score.

**How it works**:
1. Compute `FactualCorrectness` score (LLM-based claim matching)
2. Compute `SemanticSimilarity` score (embedding-based)
3. Combine using a weighted average

**Formula**:

```
Answer Correctness = w1 * FactualCorrectness + w2 * SemanticSimilarity
```

Default weights: `w1 = 0.75` (factual), `w2 = 0.25` (semantic)

**Score range**: 0.0 to 1.0

**Required fields**: `response`, `reference`

**Needs**: Both LLM and Embeddings

```python
from ragas.metrics import AnswerCorrectness

metric = AnswerCorrectness()
# Customize weights:
metric = AnswerCorrectness(weights=[0.6, 0.4])  # More weight to semantic similarity

score = await metric.single_turn_ascore(sample)
```

---

## Non-LLM Metrics

These metrics do not require an LLM and are therefore fast, cheap, and deterministic.

### 14. NonLLMStringSimilarity

**What it measures**: String-level similarity between response and reference using classical
NLP metrics.

```python
from ragas.metrics import NonLLMStringSimilarity

metric = NonLLMStringSimilarity()
score = await metric.single_turn_ascore(sample)
```

### 15. ExactMatch

**What it measures**: Whether the response exactly matches the reference (binary: 0 or 1).

```python
from ragas.metrics import ExactMatch

metric = ExactMatch()
score = await metric.single_turn_ascore(sample)
# Returns 1.0 if response == reference, else 0.0
```

### 16. StringPresence

**What it measures**: Whether specific strings from the reference are present in the response.

```python
from ragas.metrics import StringPresence

metric = StringPresence()
score = await metric.single_turn_ascore(sample)
```

### 17. RougeScore

**What it measures**: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score measuring
n-gram overlap between response and reference. Originally designed for summarization evaluation.

**Variants**:
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest Common Subsequence

```python
from ragas.metrics import RougeScore

metric = RougeScore()
# Configure variant:
metric = RougeScore(rouge_type="rouge1")   # Default
metric = RougeScore(rouge_type="rouge2")
metric = RougeScore(rouge_type="rougeL")
metric = RougeScore(measure_type="fmeasure")  # precision, recall, or fmeasure

score = await metric.single_turn_ascore(sample)
```

### 18. BleuScore

**What it measures**: BLEU (Bilingual Evaluation Understudy) score, originally designed for
machine translation evaluation. Measures n-gram precision with a brevity penalty.

```python
from ragas.metrics import BleuScore

metric = BleuScore()
score = await metric.single_turn_ascore(sample)
```

---

## Agentic / Multi-Turn Metrics

RAGAS 0.2.x introduced metrics for evaluating agentic RAG systems that involve multi-step
reasoning, tool use, and multi-turn conversations.

### 19. Topic Adherence

**What it measures**: Whether the agent stays on topic throughout a multi-turn conversation.

```python
from ragas.metrics import TopicAdherence

metric = TopicAdherence()
score = await metric.multi_turn_ascore(multi_turn_sample)
```

### 20. Tool Call Accuracy

**What it measures**: Whether the agent called the correct tools with the correct arguments
during a multi-turn interaction.

```python
from ragas.metrics import ToolCallAccuracy

metric = ToolCallAccuracy()
score = await metric.multi_turn_ascore(multi_turn_sample)
```

### 21. Agent Goal Accuracy

**What it measures**: Whether the agent achieved the user's goal by the end of the conversation.

```python
from ragas.metrics import AgentGoalAccuracy

metric = AgentGoalAccuracy()
score = await metric.multi_turn_ascore(multi_turn_sample)
```

### Multi-Turn Sample for Agentic Metrics

```python
from ragas.dataset_schema import MultiTurnSample, Message, ToolCall

sample = MultiTurnSample(
    user_input=[
        Message(content="Book a flight from NYC to London", role="user"),
        Message(
            content="I'll search for available flights.",
            role="assistant",
            tool_calls=[
                ToolCall(
                    name="search_flights",
                    args={"origin": "NYC", "destination": "London"}
                )
            ]
        ),
        Message(content="Found 3 flights. The cheapest is $450 on BA.", role="tool"),
        Message(content="I found a flight on British Airways for $450. Shall I book it?", role="assistant"),
        Message(content="Yes, book it.", role="user"),
        Message(
            content="Booking confirmed.",
            role="assistant",
            tool_calls=[
                ToolCall(
                    name="book_flight",
                    args={"flight_id": "BA123", "passenger": "user"}
                )
            ]
        ),
    ],
    reference="The agent should search for flights and book the cheapest option.",
    reference_tool_calls=[
        ToolCall(name="search_flights", args={"origin": "NYC", "destination": "London"}),
        ToolCall(name="book_flight", args={"flight_id": "BA123", "passenger": "user"}),
    ]
)
```

---

## Integration with LangChain

RAGAS integrates natively with LangChain, which is the most common integration path.

### Using LangChain LLMs

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

langchain_llm = ChatOpenAI(model="gpt-4o")
ragas_llm = LangchainLLMWrapper(langchain_llm)

# Use in metrics
from ragas.metrics import Faithfulness
metric = Faithfulness(llm=ragas_llm)
```

### Using LangChain Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

# Use in metrics
from ragas.metrics import SemanticSimilarity
metric = SemanticSimilarity(embeddings=ragas_embeddings)
```

### Evaluating LangChain Chains Directly

```python
from ragas.integrations.langchain import EvaluatorChain

# Wrap a RAGAS metric as a LangChain evaluator
faithfulness_evaluator = EvaluatorChain(metric=Faithfulness())

# Use with LangChain's evaluation framework
result = faithfulness_evaluator.evaluate_run(
    run=langchain_run,
    example=langchain_example
)
```

### Using LangSmith Traces

```python
from ragas.integrations.langchain import evaluate_with_langsmith

# Evaluate all runs in a LangSmith dataset
results = evaluate_with_langsmith(
    dataset_name="my-rag-dataset",
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=ragas_llm
)
```

---

## Integration with LlamaIndex

### Evaluating LlamaIndex Query Engines

```python
from ragas.integrations.llama_index import evaluate as llama_evaluate

# After running your LlamaIndex query engine
results = llama_evaluate(
    query_engine=index.as_query_engine(),
    metrics=[Faithfulness(), AnswerRelevancy()],
    dataset=dataset
)
```

### Converting LlamaIndex Responses

```python
from ragas.dataset_schema import SingleTurnSample

# Manual conversion from LlamaIndex response
response = query_engine.query("What is RAG?")

sample = SingleTurnSample(
    user_input="What is RAG?",
    response=str(response),
    retrieved_contexts=[node.text for node in response.source_nodes]
)
```

---

## Integration with Custom Pipelines

For pipelines not using LangChain or LlamaIndex:

```python
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

def evaluate_my_pipeline(questions, pipeline):
    samples = []
    for question in questions:
        # Run your custom pipeline
        result = pipeline.run(question)

        sample = SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=result.get("ground_truth")
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()]
    )

    return results
```

### Batch Evaluation Pattern

```python
import asyncio
from ragas.metrics import Faithfulness

async def evaluate_batch(samples):
    metric = Faithfulness()
    scores = []
    for sample in samples:
        score = await metric.single_turn_ascore(sample)
        scores.append(score)
    return scores

# Run
scores = asyncio.run(evaluate_batch(my_samples))
```

---

## Test Set Generation

RAGAS includes a powerful test set generator that creates evaluation datasets from your documents,
ensuring comprehensive coverage of different question types.

### Basic Usage

```python
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import default_transforms
from ragas.testset.synthesizers import default_query_distribution

# Initialize generator
generator = TestsetGenerator(llm=ragas_llm)

# Generate from documents
testset = generator.generate(
    documents=documents,                    # List of LangChain Documents
    testset_size=50,                        # Number of test samples to generate
    query_distribution=default_query_distribution,  # Types of questions
    transforms=default_transforms           # Document transformations
)

# Convert to EvaluationDataset
eval_dataset = testset.to_evaluation_dataset()
```

### With LangChain Documents

```python
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader

# Load documents
loader = DirectoryLoader("./docs/", glob="**/*.md")
documents = loader.load()

# Generate test set
generator = TestsetGenerator(llm=ragas_llm, embedding=ragas_embeddings)

testset = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=100
)

# Access generated samples
df = testset.to_pandas()
print(df.columns)
# ['user_input', 'reference', 'reference_contexts', 'synthesizer_name']
```

### Knowledge Graph-Based Generation

RAGAS 0.2.x uses a knowledge graph internally to understand document relationships and generate
diverse, challenging questions:

```python
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph

# The generator builds a knowledge graph from documents
generator = TestsetGenerator(llm=ragas_llm, embedding=ragas_embeddings)

# You can also provide a pre-built knowledge graph
kg = KnowledgeGraph()
# ... populate the graph ...

testset = generator.generate(
    documents=documents,
    testset_size=50,
    knowledge_graph=kg
)
```

### Question Types / Distribution Control

```python
from ragas.testset.synthesizers import (
    SingleHopQuerySynthesizer,
    MultiHopQuerySynthesizer,
    SpecificQuerySynthesizer,
    AbstractQuerySynthesizer,
)

# Control the distribution of question types
query_distribution = [
    (SingleHopQuerySynthesizer(llm=ragas_llm), 0.4),   # 40% simple questions
    (MultiHopQuerySynthesizer(llm=ragas_llm), 0.3),    # 30% multi-hop reasoning
    (SpecificQuerySynthesizer(llm=ragas_llm), 0.2),    # 20% specific detail questions
    (AbstractQuerySynthesizer(llm=ragas_llm), 0.1),    # 10% abstract questions
]

testset = generator.generate(
    documents=documents,
    testset_size=100,
    query_distribution=query_distribution
)
```

### Transforms

Transforms control how documents are processed before question generation:

```python
from ragas.testset.transforms import (
    default_transforms,
    Parallel,
    apply_transforms,
    EmbeddingExtractor,
    KeyphrasesExtractor,
    TitleExtractor,
    HeadlinesExtractor,
    SummaryExtractor,
)

# Default transforms include embedding extraction, keyphrase extraction, etc.
transforms = default_transforms(llm=ragas_llm, embedding=ragas_embeddings)

# Or build custom transform pipeline
custom_transforms = [
    EmbeddingExtractor(embedding=ragas_embeddings),
    KeyphrasesExtractor(llm=ragas_llm),
    TitleExtractor(llm=ragas_llm),
    HeadlinesExtractor(llm=ragas_llm),
    SummaryExtractor(llm=ragas_llm),
]
```

---

## Custom Metrics

RAGAS 0.2.x provides a clean API for creating custom metrics.

### Inheriting from SingleTurnMetric

```python
from ragas.metrics.base import SingleTurnMetric, MetricType
from ragas.dataset_schema import SingleTurnSample
from dataclasses import dataclass, field

@dataclass
class MyCustomMetric(SingleTurnMetric):
    name: str = "my_custom_metric"
    _required_columns: dict = field(
        default_factory=lambda: {
            "SINGLE_TURN": {"response", "reference"}
        }
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        # Your scoring logic here
        response = sample.response
        reference = sample.reference

        # Example: simple word overlap
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        overlap = len(response_words & reference_words)
        total = len(reference_words)

        return overlap / total if total > 0 else 0.0
```

### MetricWithLLM

For metrics that need an LLM:

```python
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel
from dataclasses import dataclass, field

# Define prompt input/output models
class ToneInput(BaseModel):
    response: str

class ToneOutput(BaseModel):
    is_professional: bool
    confidence: float

# Define the prompt
class ToneCheckPrompt(PydanticPrompt[ToneInput, ToneOutput]):
    instruction = "Evaluate whether the response maintains a professional tone."
    input_model = ToneInput
    output_model = ToneOutput

@dataclass
class ProfessionalToneMetric(MetricWithLLM, SingleTurnMetric):
    name: str = "professional_tone"
    tone_prompt: PydanticPrompt = field(default_factory=ToneCheckPrompt)
    _required_columns: dict = field(
        default_factory=lambda: {
            "SINGLE_TURN": {"response"}
        }
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        prompt_input = ToneInput(response=sample.response)
        result = await self.tone_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks
        )
        return result.confidence if result.is_professional else 0.0
```

### MetricWithEmbeddings

For metrics that need embeddings:

```python
from ragas.metrics.base import MetricWithEmbeddings, SingleTurnMetric
from dataclasses import dataclass, field
import numpy as np

@dataclass
class CustomSimilarityMetric(MetricWithEmbeddings, SingleTurnMetric):
    name: str = "custom_similarity"
    _required_columns: dict = field(
        default_factory=lambda: {
            "SINGLE_TURN": {"response", "reference"}
        }
    )

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        response_embedding = await self.embeddings.embed_text(sample.response)
        reference_embedding = await self.embeddings.embed_text(sample.reference)

        # Cosine similarity
        similarity = np.dot(response_embedding, reference_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(reference_embedding)
        )
        return float(similarity)
```

### Using Custom Metrics

```python
# Custom metrics work just like built-in metrics
custom_metric = MyCustomMetric()

results = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), custom_metric]
)
```

---

## Configuration

### Configuring the Default LLM

RAGAS defaults to OpenAI's GPT-4o. To change:

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# Use a different OpenAI model
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

# Use Anthropic
from langchain_anthropic import ChatAnthropic
llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-20250514"))

# Use Azure OpenAI
from langchain_openai import AzureChatOpenAI
llm = LangchainLLMWrapper(AzureChatOpenAI(
    azure_deployment="my-gpt4",
    api_version="2024-02-01"
))

# Use in evaluate()
results = evaluate(dataset=dataset, metrics=metrics, llm=llm)

# Or set on individual metrics
metric = Faithfulness(llm=llm)
```

### Configuring Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# Default: text-embedding-ada-002
embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# Use in evaluate()
results = evaluate(dataset=dataset, metrics=metrics, embeddings=embeddings)

# Or set on individual metrics
metric = AnswerRelevancy(embeddings=embeddings)
```

### RunConfig Parameters (Detailed)

```python
from ragas.run_config import RunConfig

config = RunConfig(
    timeout=120,         # Seconds before an LLM call times out (default: 60)
    max_retries=10,      # Number of retries on failure (default: 10)
    max_wait=60,         # Maximum wait between retries in seconds (default: 60)
    max_workers=16,      # Number of parallel workers (default: 16)
    log_tenacity=True,   # Log retry attempts (default: True)
    seed=42              # For reproducibility (default: None)
)
```

### Environment Variables

```bash
# Required for default OpenAI models
OPENAI_API_KEY=sk-...

# For Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...

# For Anthropic
ANTHROPIC_API_KEY=...

# RAGAS-specific
RAGAS_DEBUG=true          # Enable debug logging
RAGAS_CACHE_HOME=~/.cache/ragas  # Cache directory
```

### Caching

RAGAS caches LLM responses to avoid redundant API calls:

```python
# Disable caching
import os
os.environ["RAGAS_CACHE_HOME"] = ""

# Or clear cache
import shutil
shutil.rmtree(os.path.expanduser("~/.cache/ragas"), ignore_errors=True)
```

---

## SQL Metrics and Domain-Specific Rubrics

### DataCompyScore

`DataCompyScore` measures the similarity between two dataframes or tabular results. It is
designed for text-to-SQL and data retrieval tasks where the output is structured data rather
than free text.

**What it measures**: Whether the generated data output matches the expected data output in
terms of schema, values, and structure.

**When to use**: Evaluating text-to-SQL pipelines, data extraction systems, or any RAG
application that returns structured tabular data.

```python
from ragas.metrics import DataCompyScore
from ragas.dataset_schema import SingleTurnSample

metric = DataCompyScore()

sample = SingleTurnSample(
    user_input="What is the total revenue by region for Q3?",
    response='[{"region": "North", "revenue": 1500000}, {"region": "South", "revenue": 980000}]',
    reference='[{"region": "North", "revenue": 1500000}, {"region": "South", "revenue": 980000}]',
)

score = await metric.single_turn_ascore(sample)
# score = 1.0 (exact match of tabular data)
```

### SQLSemanticEquivalence

`SQLSemanticEquivalence` compares two SQL queries for semantic equivalence. Two queries are
semantically equivalent if they produce the same result set, even if the SQL syntax differs.

**What it measures**: Whether a generated SQL query is semantically equivalent to a reference
SQL query, regardless of syntactic differences (e.g., column ordering, alias names, JOIN style).

```python
from ragas.metrics import SQLSemanticEquivalence
from ragas.dataset_schema import SingleTurnSample

metric = SQLSemanticEquivalence()

sample = SingleTurnSample(
    user_input="Get all customers who ordered more than 5 items",
    response="SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name HAVING COUNT(*) > 5",
    reference="SELECT customers.name FROM customers INNER JOIN orders ON customers.id = orders.customer_id GROUP BY customers.name HAVING COUNT(orders.id) > 5",
)

score = await metric.single_turn_ascore(sample)
# score should be high -- both queries produce the same result set
```

### Domain-Specific Rubrics

RAGAS provides rubric-based metrics that allow you to define custom scoring criteria for
specific domains. These are particularly powerful for evaluation tasks where generic metrics
do not capture the nuances of your use case.

#### DomainSpecificRubrics

Define evaluation criteria specific to your domain:

```python
from ragas.metrics import DomainSpecificRubrics
from ragas.dataset_schema import SingleTurnSample

# Medical domain rubric
medical_rubric = DomainSpecificRubrics(
    name="medical_accuracy",
    rubrics={
        "score1_description": "Response contains dangerous medical misinformation or could lead to patient harm. Missing critical safety warnings.",
        "score2_description": "Response has significant medical inaccuracies. Important caveats or contraindications are missing.",
        "score3_description": "Response is mostly accurate but lacks specificity. Some medical nuances are oversimplified.",
        "score4_description": "Response is medically accurate with appropriate caveats. Minor details could be more precise.",
        "score5_description": "Response is medically accurate, comprehensive, includes appropriate disclaimers, and recommends professional consultation where needed.",
    },
)

sample = SingleTurnSample(
    user_input="What are the side effects of metformin?",
    response="Common side effects include nausea and diarrhea. Consult your doctor for personalized advice.",
    retrieved_contexts=["Metformin side effects: nausea, diarrhea, stomach pain, metallic taste. Rare: lactic acidosis."],
)

score = await medical_rubric.single_turn_ascore(sample)
```

#### InstanceSpecificRubrics

For cases where each test case has its own rubric (set via the `rubric` field on the sample):

```python
from ragas.metrics import InstanceSpecificRubrics
from ragas.dataset_schema import SingleTurnSample

metric = InstanceSpecificRubrics()

sample = SingleTurnSample(
    user_input="Explain the concept of derivative in calculus",
    response="A derivative measures the rate of change of a function at a point.",
    rubric={
        "score1_description": "No mention of rate of change or slope concept",
        "score2_description": "Vague mention of change without mathematical precision",
        "score3_description": "Correctly describes rate of change but missing limit definition",
        "score4_description": "Accurate description with limit concept mentioned",
        "score5_description": "Complete explanation including limit definition, geometric interpretation, and notation",
    },
)

score = await metric.single_turn_ascore(sample)
```

#### RubricsScoreWithReference and RubricsScoreWithoutReference

These variants control whether the LLM judge has access to a reference answer:

```python
from ragas.metrics import RubricsScoreWithReference, RubricsScoreWithoutReference

# With reference: judge compares response to reference using the rubric
with_ref = RubricsScoreWithReference()

# Without reference: judge evaluates response quality using only the rubric
without_ref = RubricsScoreWithoutReference()

# Example with reference for legal domain
legal_sample = SingleTurnSample(
    user_input="What is the statute of limitations for breach of contract in California?",
    response="In California, the statute of limitations for written contracts is 4 years.",
    reference="California Code of Civil Procedure Section 337 sets a 4-year statute of limitations for written contracts and Section 339 sets 2 years for oral contracts.",
    rubric={
        "score1_description": "Incorrect jurisdiction or statute cited",
        "score2_description": "Correct jurisdiction but wrong limitation period",
        "score3_description": "Partially correct but missing key distinctions (written vs oral)",
        "score4_description": "Correct for the specific contract type mentioned",
        "score5_description": "Comprehensive answer covering both written and oral contracts with correct citations",
    },
)

score = await with_ref.single_turn_ascore(legal_sample)
```

### Defining Rubrics for Specific Domains

When creating domain-specific rubrics, follow these principles:

```
RUBRIC DESIGN GUIDELINES:

  1. ANCHOR TO DOMAIN EXPERTISE
     - Medical: ground in clinical guidelines and evidence levels
     - Legal: ground in jurisdictional accuracy and citation correctness
     - Financial: ground in regulatory compliance and numerical precision

  2. DEFINE CLEAR FAILURE MODES
     - Score 1 should describe the most dangerous/wrong outcome
     - Score 5 should describe expert-level quality

  3. MAKE SCORES DISCRIMINATIVE
     - Each score level should be clearly distinguishable
     - Avoid overlapping descriptions between adjacent scores

  4. INCLUDE SAFETY CONSIDERATIONS
     - For high-stakes domains, explicitly penalize missing safety caveats
     - Score 1 should capture potentially harmful outputs
```

---

## Comparison with DeepEval

### Feature Comparison

| Feature | RAGAS 0.2.x | DeepEval 3.x |
|---------|-------------|--------------|
| **Focus** | RAG-specific evaluation | General LLM evaluation + RAG |
| **Metrics count** | ~20 core metrics | 50+ metrics across 9 categories |
| **Testing framework** | Standalone evaluate() | pytest-native (assert_test) |
| **Data format** | SingleTurnSample/MultiTurnSample | LLMTestCase/ConversationalTestCase |
| **LLM provider** | Via LangChain wrappers | Direct + LangChain + custom |
| **Default LLM** | OpenAI GPT-4o | OpenAI GPT-4.1 |
| **Test generation** | Built-in TestsetGenerator | Built-in synthesizer (evolution-based) |
| **CI/CD** | Manual integration | pytest-native, first-class CI/CD |
| **Cloud platform** | No official platform | Confident AI platform |
| **Custom metrics** | SingleTurnMetric/MultiTurnMetric | BaseMetric subclass |
| **Agentic eval** | Topic adherence, tool accuracy, goal accuracy | Task completion, tool correctness, 6+ agent metrics |
| **Safety metrics** | Not built-in | Bias, toxicity, PII, 5+ safety metrics |
| **Multi-modal** | Limited | Full multi-modal support |
| **Research backing** | Published paper, academic citations | G-Eval paper, industry-focused |
| **Tracing** | Via LangSmith integration | Built-in @observe decorator |
| **Pricing** | Free and open source | Free (open source) + paid Confident AI |

### When to Use RAGAS

- You want **research-backed** metrics with published methodology
- Your primary concern is **RAG pipeline evaluation** specifically
- You are already using **LangChain** or **LlamaIndex**
- You need **test set generation** from documents
- You prefer a **focused, lightweight** evaluation library
- Academic or research settings where methodology transparency matters

### When to Use DeepEval

- You need **comprehensive coverage** beyond just RAG (safety, agentic, conversational)
- You want **pytest-native** testing with CI/CD integration
- You need a **cloud platform** for tracking results over time
- You are evaluating **agents** with complex tool-use patterns
- You need **custom metrics** with flexible criteria (G-Eval)
- Production environments where monitoring and regression testing matter

### Using Both Together

There is no reason you cannot use both frameworks on the same RAG pipeline:

```python
# Evaluate with RAGAS
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Faithfulness as RagasFaithfulness

ragas_results = ragas_evaluate(
    dataset=ragas_dataset,
    metrics=[RagasFaithfulness()]
)

# Evaluate with DeepEval
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

deepeval_metric = FaithfulnessMetric(threshold=0.7)
deepeval_test_case = LLMTestCase(
    input="...",
    actual_output="...",
    retrieval_context=[...]
)

deepeval_evaluate(test_cases=[deepeval_test_case], metrics=[deepeval_metric])

# Compare scores — they measure the same concept but may differ slightly
# because the algorithms and prompts are different
```

### Key Algorithmic Differences

| Concept | RAGAS Approach | DeepEval Approach |
|---------|---------------|-------------------|
| **Faithfulness** | Claim extraction + NLI classification | Claim extraction + truthfulness verification |
| **Answer Relevancy** | Generate questions from answer, compare via embeddings | Extract statements from answer, classify as relevant/irrelevant |
| **Context Precision** | LLM judges context relevance, computes average precision | LLM judges each node, computes Weighted Contextual Precision |
| **Context Recall** | Check if reference statements attributable to contexts | Check if expected_output statements attributable to contexts |

The differences mean that the same RAG output may receive different scores from RAGAS and DeepEval.
This is not a bug — it reflects different measurement philosophies. Using both provides a more
robust evaluation signal.

---

## Note: RAGAS v0.4.x API Evolution (Collections Module)

RAGAS continues to evolve. Version 0.4.x introduces further API changes, including a
`collections` module and factory functions for provider configuration. If you are using
a newer version, be aware of these differences.

### The Collections Module

In v0.4.x, some metrics are organized under a `collections` module:

```python
# v0.4.x collections-based imports
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
```

This differs from the 0.2.x import path:

```python
# v0.2.x (current stable) imports
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
```

### Factory Functions for Provider Configuration

v0.4.x introduces `llm_factory` and `embedding_factory` for simpler provider setup:

```python
# v0.4.x factory-based configuration
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

# Create LLM with factory (provider auto-detected from model name)
llm = llm_factory(model="gpt-4o", temperature=0)

# Create embeddings with factory
embeddings = embedding_factory(model="text-embedding-3-small")

# Use directly in evaluate()
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
)
```

Compare this to the 0.2.x approach which requires explicit LangChain wrappers:

```python
# v0.2.x wrapper-based configuration
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
```

### API Differences from 0.2.x

| Aspect | v0.2.x | v0.4.x |
|--------|--------|--------|
| Metric imports | `from ragas.metrics import ...` | `from ragas.metrics.collections import ...` (some metrics) |
| LLM setup | `LangchainLLMWrapper(ChatOpenAI(...))` | `llm_factory(model=...)` |
| Embedding setup | `LangchainEmbeddingsWrapper(OpenAIEmbeddings(...))` | `embedding_factory(model=...)` |
| RunConfig | Passed to `evaluate(run_config=...)` | Some options moved to direct kwargs |
| Metric kwargs | Via `RunConfig` | Some accept direct kwargs (e.g., `timeout`, `max_retries`) |

### Migration Notes: 0.2.x to 0.4.x

1. **Check import paths**: Some metrics may move to `ragas.metrics.collections`. If imports
   fail after upgrading, check the collections module.

2. **Simplify LLM setup**: Replace `LangchainLLMWrapper` with `llm_factory` for common
   providers (OpenAI, Anthropic, Azure). The factory approach requires fewer dependencies.

3. **Update RunConfig usage**: Some execution parameters that were in `RunConfig` may now
   be passed directly as keyword arguments. Check the updated documentation for your version.

4. **Pin your version**: If you have a working evaluation pipeline, pin the RAGAS version
   in your `requirements.txt` to avoid surprise breaking changes:
   ```
   ragas==0.2.6  # or your current version
   ```

5. **Test after upgrading**: Always re-run your evaluation suite after upgrading RAGAS. Even
   minor version bumps can change metric scores due to prompt or algorithm updates.

---

## Appendix: Complete Working Example

```python
"""
Complete RAGAS evaluation example for a RAG pipeline.
"""

import os
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    SemanticSimilarity,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure LLM and embeddings
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# Create evaluation samples
samples = [
    SingleTurnSample(
        user_input="What is the capital of France?",
        response="The capital of France is Paris, a city known for the Eiffel Tower and Louvre Museum.",
        retrieved_contexts=[
            "Paris is the capital and largest city of France, with a population of over 2 million.",
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889."
        ],
        reference="Paris is the capital of France.",
        reference_contexts=["Paris is the capital and largest city of France."]
    ),
    SingleTurnSample(
        user_input="How does photosynthesis work?",
        response="Photosynthesis converts sunlight into chemical energy using chlorophyll in plant cells.",
        retrieved_contexts=[
            "Photosynthesis is the process by which green plants convert light energy to chemical energy.",
            "Chlorophyll is the green pigment in plants that absorbs light energy."
        ],
        reference="Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.",
        reference_contexts=[
            "Photosynthesis occurs in chloroplasts and involves light-dependent and light-independent reactions."
        ]
    ),
]

# Create dataset
dataset = EvaluationDataset(samples=samples)

# Configure metrics
metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(llm=llm, embeddings=embeddings),
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=embeddings),
    SemanticSimilarity(embeddings=embeddings),
]

# Configure execution
run_config = RunConfig(
    timeout=120,
    max_retries=5,
    max_workers=8,
    seed=42
)

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config
)

# Print results
print("=== Aggregated Scores ===")
print(results)

print("\n=== Per-Sample Scores ===")
df = results.to_pandas()
print(df.to_string())

# Access individual metric scores
for i, score_dict in enumerate(results.scores):
    print(f"\nSample {i+1}:")
    for metric_name, score in score_dict.items():
        print(f"  {metric_name}: {score:.4f}")
```

### Expected Output Format

```
=== Aggregated Scores ===
{
    'faithfulness': 0.9250,
    'answer_relevancy': 0.8714,
    'context_precision': 0.8750,
    'context_recall': 0.7500,
    'answer_correctness': 0.8125,
    'semantic_similarity': 0.8903
}
```

---

## Appendix: Migration from 0.1.x to 0.2.x

If you have existing 0.1.x code, here is how to migrate:

### Field Name Changes

```python
# OLD (0.1.x)
from datasets import Dataset
data = {
    "question": ["What is X?"],
    "answer": ["X is..."],
    "contexts": [["Context 1", "Context 2"]],
    "ground_truth": ["X is defined as..."]
}
dataset = Dataset.from_dict(data)

# NEW (0.2.x)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
sample = SingleTurnSample(
    user_input="What is X?",
    response="X is...",
    retrieved_contexts=["Context 1", "Context 2"],
    reference="X is defined as..."
)
dataset = EvaluationDataset(samples=[sample])
```

### Metric Import Changes

```python
# OLD (0.1.x)
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# NEW (0.2.x)
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
```

### evaluate() Call Changes

```python
# OLD (0.1.x)
from ragas import evaluate
result = evaluate(dataset, metrics=metrics)

# NEW (0.2.x)
from ragas import evaluate
result = evaluate(dataset=dataset, metrics=metrics)
```

---

## Appendix: Troubleshooting

### Common Errors

**1. "Missing required column" error**
```
ValueError: Required column 'reference' not found in sample
```
Fix: Ensure your `SingleTurnSample` includes all fields required by the metrics you are using.
Check the metric's `_required_columns` attribute.

**2. Rate limiting from OpenAI**
```python
# Reduce parallelism
run_config = RunConfig(max_workers=4, timeout=180, max_retries=15)
```

**3. "LLM not configured" error**
```python
# Ensure LLM is set either globally or per-metric
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
metric = Faithfulness(llm=llm)
```

**4. Inconsistent scores across runs**
LLM-based metrics are inherently non-deterministic. To improve consistency:
```python
# Use temperature=0
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
# Set a seed in RunConfig
run_config = RunConfig(seed=42)
```

**5. Slow evaluation**
```python
# Increase parallelism
run_config = RunConfig(max_workers=32)
# Or use a faster/cheaper model for evaluation
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
```

---

## Appendix: RAGAS Metric Decision Tree

Use this to decide which metrics to use for your evaluation:

```
Do you have ground truth (reference answers)?
├── YES
│   ├── Want to evaluate retriever?
│   │   ├── Context Precision (ranking quality)
│   │   ├── Context Recall (coverage)
│   │   └── Context Entity Recall (entity-level, no LLM needed)
│   ├── Want to evaluate generator?
│   │   ├── Factual Correctness (claim-level comparison)
│   │   ├── Answer Correctness (factual + semantic hybrid)
│   │   └── Semantic Similarity (embedding-based, no LLM needed)
│   └── Want non-LLM metrics?
│       ├── ExactMatch
│       ├── RougeScore
│       └── BleuScore
│
└── NO (reference-free)
    ├── Want to evaluate retriever?
    │   └── LLMContextPrecisionWithoutReference
    ├── Want to evaluate generator?
    │   ├── Faithfulness (grounding in context)
    │   └── Answer Relevancy (relevance to question)
    └── Want to evaluate robustness?
        └── Noise Sensitivity
```

---

*Previous: [05 - DeepEval Complete Guide](05_deepeval_complete_guide.md) | Next: [07 - Retriever Metrics Deep Dive](07_retriever_metrics_deep_dive.md)*

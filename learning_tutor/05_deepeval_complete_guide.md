# 05 - DeepEval Complete Guide

> **Goal of this document:** Serve as the single, definitive reference for the DeepEval evaluation framework. After reading this you should be able to evaluate any RAG or agentic system using DeepEval without consulting any other resource. Every metric, every configuration option, every pattern is covered here.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
3. [RAG Metrics (Detailed)](#3-rag-metrics-detailed)
4. [All Other Metrics](#4-all-other-metrics)
5. [Configuration](#5-configuration)
6. [Evaluation Patterns](#6-evaluation-patterns)
7. [Synthetic Data Generation](#7-synthetic-data-generation)
8. [Confident AI Platform](#8-confident-ai-platform)
9. [Red Teaming (DeepTeam)](#9-red-teaming-deepteam)
10. [Best Practices and Gotchas](#10-best-practices-and-gotchas)

---

## 1. Overview

### 1.1 What is DeepEval?

DeepEval is an open-source LLM evaluation framework created by **Confident AI** (founded by Jeffrey Ip and Ishan Bhatia, headquartered in San Francisco). It was first released in late 2023 and has rapidly become one of the two dominant RAG evaluation frameworks (alongside RAGAS).

**Key philosophy:** Evaluation should be as easy as unit testing. DeepEval is built natively on **pytest**, so evaluations are just test functions.

**Core capabilities:**
- 50+ evaluation metrics covering RAG, generation, conversation, agents, and safety
- LLM-as-a-judge architecture (uses an LLM to evaluate LLM outputs)
- Pytest-native: write evaluations as test functions, run with `pytest`
- Supports batch evaluation, async evaluation, and dataset-driven evaluation
- Synthetic test data generation from documents
- Integration with Confident AI platform for dashboards, regression testing, and tracing
- Red teaming / adversarial testing via DeepTeam

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DeepEval                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Test Cases   │  │   Metrics    │  │  Evaluation  │  │
│  │ (LLMTestCase, │  │ (50+ built-  │  │   Engine     │  │
│  │  Conversa-    │  │  in metrics) │  │ (pytest +    │  │
│  │  tional)      │  │              │  │  async)      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                 │                 │            │
│         v                 v                 v            │
│  ┌──────────────────────────────────────────────────┐   │
│  │              LLM Judge (GPT-4.1 default)         │   │
│  │    Evaluates outputs using metric-specific       │   │
│  │    prompts and scoring rubrics                   │   │
│  └──────────────────────────────────────────────────┘   │
│         │                                               │
│         v                                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Results + Confident AI Dashboard        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Installation and Setup

```bash
# Install DeepEval
pip install deepeval

# Verify installation
deepeval --version

# Login to Confident AI (optional, for dashboard features)
deepeval login
```

### 1.4 Environment Variables

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `OPENAI_API_KEY` | Default LLM judge (GPT-4.1) | Yes (unless custom LLM configured) |
| `CONFIDENT_API_KEY` | Confident AI platform integration | No (only for dashboard/tracing) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI as judge | Only if using Azure |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Only if using Azure |
| `AZURE_DEPLOYMENT_NAME` | Azure model deployment name | Only if using Azure |
| `AZURE_API_VERSION` | Azure API version | Only if using Azure |

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export CONFIDENT_API_KEY="..."  # optional
```

Or use a `.env` file with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 2. Core Concepts

### 2.1 LLMTestCase

The `LLMTestCase` is the fundamental data structure. It represents a single evaluation scenario.

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris is the capital of France.",
    context=["France is a country in Europe. Its capital city is Paris."],
    retrieval_context=["Paris is the capital and largest city of France."],
    tools_called=["search_database"],
    expected_tools=["search_database", "verify_answer"]
)
```

**Field reference:**

| Field | Type | Description | Required For |
|-------|------|-------------|-------------|
| `input` | `str` | The user's query / prompt input | Almost all metrics |
| `actual_output` | `str` | The LLM's generated response | All metrics |
| `expected_output` | `str` | The ground truth / reference answer | AnswerCorrectness, ContextRecall, ContextPrecision |
| `context` | `List[str]` | The ground truth context (what SHOULD be provided) | HallucinationMetric, Summarization |
| `retrieval_context` | `List[str]` | The actually retrieved context (what WAS provided) | Faithfulness, ContextRelevancy, ContextPrecision, ContextRecall |
| `tools_called` | `List[str]` | Tools the LLM actually called | ToolCorrectness |
| `expected_tools` | `List[str]` | Tools the LLM should have called | ToolCorrectness |

**CRITICAL DISTINCTION: `context` vs `retrieval_context`**

This is one of the most confusing aspects of DeepEval:

- **`context`**: The "ideal" or "ground truth" context. Used by `HallucinationMetric` and `SummarizationMetric`. Think of it as "what should the LLM know."
- **`retrieval_context`**: What the retriever ACTUALLY returned. Used by `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`. Think of it as "what the LLM was actually given."

In a typical RAG evaluation, you use `retrieval_context` (what your system retrieved). You use `context` for non-RAG scenarios like summarization or hallucination detection where you control the context.

### 2.2 ConversationalTestCase and Turn

For evaluating multi-turn conversations:

```python
from deepeval.test_case import ConversationalTestCase, LLMTestCase

# Each turn is an LLMTestCase
turn_1 = LLMTestCase(
    input="What products do you sell?",
    actual_output="We sell electronics, clothing, and home goods."
)

turn_2 = LLMTestCase(
    input="What's the return policy for electronics?",
    actual_output="Electronics can be returned within 30 days with original packaging."
)

turn_3 = LLMTestCase(
    input="And for clothing?",
    actual_output="Clothing can be returned within 60 days, unworn with tags attached."
)

convo_test_case = ConversationalTestCase(
    chatbot_role="Customer service agent for an e-commerce store",
    turns=[turn_1, turn_2, turn_3]
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chatbot_role` | `str` | Description of the chatbot's intended role |
| `turns` | `List[LLMTestCase]` | Ordered list of conversation turns |

### 2.3 Golden and ConversationalGolden

A `Golden` is a test case template stored in a dataset. It is like an `LLMTestCase` but does not require `actual_output` (because the output will be generated at evaluation time).

```python
from deepeval.dataset import Golden, ConversationalGolden

golden = Golden(
    input="What is the capital of France?",
    expected_output="Paris",
    context=["France is a country in Europe. Its capital is Paris."]
)

# ConversationalGolden for multi-turn
convo_golden = ConversationalGolden(
    chatbot_role="Geography tutor",
    turns=[golden]  # list of Golden objects
)
```

### 2.4 EvaluationDataset

A collection of `Golden` objects for batch evaluation:

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()

# Add goldens manually
dataset.add_golden(golden)

# Or create from a list of test cases
dataset = EvaluationDataset(goldens=[golden1, golden2, golden3])

# Push to Confident AI
dataset.push(alias="my_rag_eval_v1")

# Pull from Confident AI
dataset = EvaluationDataset()
dataset.pull(alias="my_rag_eval_v1")
```

### 2.5 BaseMetric Interface

Every metric in DeepEval implements this interface:

```python
class BaseMetric:
    score: float           # The computed score (0.0 to 1.0 for most metrics)
    reason: str            # Human-readable explanation of the score
    is_successful: bool    # Whether score >= threshold
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous evaluation. Returns the score."""
        ...
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Asynchronous evaluation. Returns the score."""
        ...
```

**Usage pattern:**

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What is X?",
    actual_output="X is Y.",
    retrieval_context=["X is defined as Y in the manual."]
)

# Synchronous
metric.measure(test_case)
print(f"Score: {metric.score}")      # e.g., 1.0
print(f"Reason: {metric.reason}")    # e.g., "All claims are supported..."
print(f"Passed: {metric.is_successful}")  # True if score >= threshold

# Asynchronous
import asyncio
score = asyncio.run(metric.a_measure(test_case))
```

---

## 3. RAG Metrics (Detailed)

These are the core metrics for evaluating RAG systems. For each metric, we cover: what it measures, the algorithm, required inputs, score range, threshold recommendations, a complete code example, and how to interpret verbose output.

### 3.1 AnswerRelevancyMetric

**What it measures:** Does the generated answer actually address the user's question?

**Algorithm:**
1. Break the `actual_output` into individual statements
2. For each statement, classify whether it is relevant to the `input` (query)
3. Score = number of relevant statements / total statements

**Why statements, not the whole answer?** Breaking into statements allows the metric to penalize answers that are partially relevant. An answer that addresses the question in the first sentence but then rambles irrelevantly for three paragraphs will get a lower score than a concise, focused answer.

**Required inputs:** `input`, `actual_output`

**Score range:** 0.0 to 1.0

**Recommended threshold:** 0.7 for general use, 0.5 for exploratory/creative tasks, 0.9 for focused Q&A

```python
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",          # LLM judge model
    include_reason=True,       # Include explanation
    async_mode=True,           # Use async for speed
    strict_mode=False,         # If True, score is 0 if below threshold
    verbose_mode=True          # Print detailed evaluation steps
)

test_case = LLMTestCase(
    input="What are the benefits of exercise?",
    actual_output="Exercise improves cardiovascular health, boosts mood through "
                  "endorphin release, and helps maintain healthy weight. "
                  "By the way, the weather is nice today."
)

metric.measure(test_case)
print(f"Score: {metric.score}")    # e.g., 0.75 (3 of 4 statements relevant)
print(f"Reason: {metric.reason}")  # "3 out of 4 statements are relevant..."

# Use in a test function
def test_answer_relevancy():
    assert_test(test_case, [metric])
```

**Verbose output interpretation:**

```
Statements extracted from actual_output:
  1. "Exercise improves cardiovascular health" → RELEVANT
  2. "Exercise boosts mood through endorphin release" → RELEVANT  
  3. "Exercise helps maintain healthy weight" → RELEVANT
  4. "The weather is nice today" → IRRELEVANT

Score: 3/4 = 0.75
```

### 3.2 FaithfulnessMetric

**What it measures:** Are all claims in the generated output supported by the retrieval context? This is the most important RAG metric.

**Algorithm:**
1. Extract all claims/statements from `actual_output`
2. For each claim, determine if it can be inferred from `retrieval_context`
3. Score = number of faithful claims / total claims

**Required inputs:** `input`, `actual_output`, `retrieval_context`

**Score range:** 0.0 to 1.0

**Recommended threshold:** 0.8 for general use, 0.95 for high-stakes (legal, medical), 0.7 for creative tasks

**Configuration options specific to Faithfulness:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `truths_extraction_limit` | `int` | None | Maximum number of truths to extract from context (for cost control) |
| `penalize_ambiguous_claims` | `bool` | False | If True, ambiguous claims that are neither clearly supported nor clearly contradicted count as unfaithful |

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(
    threshold=0.8,
    model="gpt-4.1",
    include_reason=True,
    async_mode=True,
    verbose_mode=True,
    truths_extraction_limit=None,        # No limit
    penalize_ambiguous_claims=False       # Lenient on ambiguity
)

test_case = LLMTestCase(
    input="What is the company's revenue?",
    actual_output="The company reported revenue of $4.2 billion in Q3, "
                  "representing 15% year-over-year growth. The CEO expressed "
                  "optimism about Q4 projections.",
    retrieval_context=[
        "In Q3 2024, the company reported total revenue of $4.2 billion, "
        "a 15% increase compared to Q3 2023.",
        "The earnings call transcript did not include forward-looking statements."
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")    # e.g., 0.667 (2 of 3 claims supported)
print(f"Reason: {metric.reason}")
```

**Verbose output interpretation:**

```
Claims extracted from actual_output:
  1. "The company reported revenue of $4.2 billion in Q3" → SUPPORTED by context[0]
  2. "Revenue represented 15% year-over-year growth" → SUPPORTED by context[0]
  3. "The CEO expressed optimism about Q4 projections" → NOT SUPPORTED (no such info in context)

Verdict: 2 out of 3 claims are faithful
Score: 0.667
```

**Impact of `penalize_ambiguous_claims`:**

With `penalize_ambiguous_claims=True`, claims that are vague or could be interpreted multiple ways are counted as unfaithful. Example:

```
Context: "The product costs $29.99"
Claim: "The product is affordably priced"  
  → Without penalize_ambiguous: This might be classified as SUPPORTED (inferrable)
  → With penalize_ambiguous: This is UNFAITHFUL (subjective judgment not in context)
```

### 3.3 ContextualRelevancyMetric

**What it measures:** What proportion of the information in the retrieved context is actually relevant to the user's query?

**Algorithm:**
1. Extract all statements from `retrieval_context`
2. For each statement, classify whether it is relevant to `input`
3. Score = number of relevant statements / total statements

**Intuition:** A score of 0.3 means 70% of your retrieved context is noise. The generator has to wade through irrelevant information to find what it needs.

**Required inputs:** `input`, `retrieval_context`

**Score range:** 0.0 to 1.0

**Recommended threshold:** 0.5 minimum, 0.7 for production, 0.9 for high-precision systems

```python
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True,
    async_mode=True,
    verbose_mode=True
)

test_case = LLMTestCase(
    input="What is the return policy for electronics?",
    retrieval_context=[
        "Electronics purchased from our store can be returned within 30 days "
        "of purchase with original receipt and packaging.",
        "Our store is located at 123 Main Street, open Monday through Saturday.",
        "We offer a price match guarantee on all electronics within 14 days."
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")  # e.g., 0.667 (2 of 3 chunks relevant)
```

**Verbose output interpretation:**

```
Statements from retrieval_context classified against input:
  Context chunk 1: "Electronics can be returned within 30 days..." → RELEVANT
  Context chunk 2: "Our store is located at 123 Main Street..." → IRRELEVANT
  Context chunk 3: "We offer a price match guarantee..." → RELEVANT (related to electronics purchases)

Score: 2/3 = 0.667
```

### 3.4 ContextualPrecisionMetric

**What it measures:** Are the relevant chunks ranked higher than the irrelevant chunks in the retrieval results?

**Algorithm (Weighted Cumulative Precision):**
1. For each chunk in `retrieval_context`, determine if it is relevant to answering the question (using `expected_output` as reference)
2. Calculate weighted precision favoring relevant documents at earlier positions

**Formula:**

```
ContextPrecision@K = (1 / |Relevant|) * Sum_{k=1}^{K} [Precision@k * rel(k)]

where:
  Precision@k = |relevant docs in top k| / k
  rel(k) = 1 if doc at rank k is relevant, else 0
  |Relevant| = total number of relevant docs
```

**Required inputs:** `input`, `retrieval_context`, `expected_output`

**Score range:** 0.0 to 1.0

**Recommended threshold:** 0.7

```python
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True,
    async_mode=True,
    verbose_mode=True
)

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris is the capital of France.",
    retrieval_context=[
        "Paris is the capital and most populous city of France.",
        "France is known for its cuisine, art, and the Eiffel Tower.",
        "The population of Paris is approximately 2.1 million.",
        "Berlin is the capital of Germany.",
        "France shares borders with Belgium, Luxembourg, and Germany."
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")
# Chunks 1 and 3 are relevant (mention Paris as capital / Paris population)
# Chunk 1 is at rank 1 (good), Chunk 3 is at rank 3 (okay)
# Score will be high because the most relevant chunk is first
```

**Verbose output interpretation:**

```
Relevance classification for each retrieved chunk:
  Rank 1: "Paris is the capital..." → RELEVANT ✓
  Rank 2: "France is known for its cuisine..." → IRRELEVANT ✗
  Rank 3: "The population of Paris is approximately 2.1M" → RELEVANT ✓
  Rank 4: "Berlin is the capital of Germany" → IRRELEVANT ✗
  Rank 5: "France shares borders with..." → IRRELEVANT ✗

Precision@1 = 1/1 = 1.0,  rel(1) = 1 → contributes 1.0
Precision@2 = 1/2 = 0.5,  rel(2) = 0 → contributes 0
Precision@3 = 2/3 = 0.67, rel(3) = 1 → contributes 0.67
Precision@4 = 2/4 = 0.5,  rel(4) = 0 → contributes 0
Precision@5 = 2/5 = 0.4,  rel(5) = 0 → contributes 0

Score = (1/2) * (1.0 + 0.67) = 0.833
```

### 3.5 ContextualRecallMetric

**What it measures:** Did the retriever find all the information needed to answer the question?

**Algorithm:**
1. Extract statements from `expected_output` (ground truth answer)
2. For each statement, check if it can be attributed to any chunk in `retrieval_context`
3. Score = number of attributable statements / total statements

**Intuition:** If the ground truth answer has 5 key facts, and only 3 of them appear in the retrieved context, the retriever missed 40% of the needed information.

**Required inputs:** `input`, `retrieval_context`, `expected_output`

**Score range:** 0.0 to 1.0

**Recommended threshold:** 0.7 for general use, 0.9 for completeness-critical applications

```python
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True,
    async_mode=True,
    verbose_mode=True
)

test_case = LLMTestCase(
    input="Describe the benefits of the premium plan.",
    actual_output="The premium plan offers unlimited storage and priority support.",
    expected_output="The premium plan includes unlimited storage, priority 24/7 support, "
                    "custom integrations, and a dedicated account manager.",
    retrieval_context=[
        "Premium plan features: unlimited cloud storage, priority support available 24/7.",
        "All plans include basic email support and 99.9% uptime SLA.",
        "Enterprise plan features include custom integrations and dedicated account manager."
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")  # e.g., 0.5 (2 of 4 expected claims found in context)
```

**Verbose output interpretation:**

```
Statements from expected_output:
  S1: "Premium plan includes unlimited storage" → FOUND in context[0] ✓
  S2: "Premium plan includes priority 24/7 support" → FOUND in context[0] ✓
  S3: "Premium plan includes custom integrations" → NOT FOUND (in enterprise context, not premium) ✗
  S4: "Premium plan includes dedicated account manager" → NOT FOUND ✗

Score: 2/4 = 0.50
Reason: The retrieval context only covers storage and support features. 
Custom integrations and dedicated account manager info was not retrieved.
```

---

## 4. All Other Metrics

### 4.1 HallucinationMetric

**What it measures:** Does the output contain information that contradicts the provided `context`?

**CRITICAL DISTINCTION FROM FAITHFULNESS:** HallucinationMetric uses `context` (ground truth), NOT `retrieval_context`. FaithfulnessMetric uses `retrieval_context` (what was actually retrieved).

| Metric | Uses | Measures |
|--------|------|----------|
| FaithfulnessMetric | `retrieval_context` | Are claims supported by what was retrieved? |
| HallucinationMetric | `context` | Does the output contradict the ground truth? |

**Use HallucinationMetric when:** You have a fixed, known context (e.g., summarization task, or when you directly control what context is given to the LLM).

**Use FaithfulnessMetric when:** You are evaluating a RAG system where the context is dynamically retrieved.

**Algorithm:**
1. For each `context` entry, determine if the `actual_output` agrees or contradicts it
2. Score = number of contradicted contexts / total contexts
3. **Note:** A LOWER score means LESS hallucination (0.0 = no hallucination)

**Score range:** 0.0 to 1.0 (INVERTED — lower is better!)

**Threshold:** Set threshold to the MAXIMUM acceptable hallucination rate (e.g., 0.1 means max 10% hallucination tolerated).

```python
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

metric = HallucinationMetric(
    threshold=0.5,  # Maximum 50% hallucination tolerated
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="Summarize the company's performance.",
    actual_output="The company achieved record revenue of $5B and expanded to 20 new markets.",
    context=[
        "The company reported revenue of $4.2B in the latest quarter.",
        "The company expanded to 15 new markets this year."
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")
# Score might be 1.0 (both contexts contradicted: $5B vs $4.2B, 20 vs 15)
# is_successful = False because 1.0 > 0.5 threshold
```

### 4.2 SummarizationMetric

**What it measures:** How well does the output summarize the provided context?

**Algorithm:** Combines two sub-scores:
1. **Alignment score:** Are the claims in the summary supported by the context? (similar to faithfulness)
2. **Coverage score:** Does the summary capture the key information from the context?

**Required inputs:** `input`, `actual_output`, `context`

**Score range:** 0.0 to 1.0

```python
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

metric = SummarizationMetric(
    threshold=0.7,
    model="gpt-4.1",
    assessment_questions=None  # Auto-generates assessment questions
)

test_case = LLMTestCase(
    input="Summarize the following document.",
    actual_output="The document describes the company's Q3 performance...",
    context=["Full document text here..."]  # The original text being summarized
)

metric.measure(test_case)
```

### 4.3 GEval (G-Eval)

**What it measures:** Custom criteria defined by you. G-Eval is the most flexible metric in DeepEval.

**Algorithm:**
1. Takes your custom evaluation criteria and generates a chain-of-thought (CoT) evaluation plan
2. Uses the CoT to score the output on a scale of 1-10 (or configurable)
3. Normalizes to 0.0-1.0

**Key property:** G-Eval uses **form-filling** — the LLM generates multiple scores and they are averaged, which reduces individual evaluation variance. However, it is inherently **non-deterministic** because the CoT generation varies across runs.

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

# Define a custom metric for "technical accuracy"
technical_accuracy = GEval(
    name="Technical Accuracy",
    criteria="Evaluate whether the response contains technically accurate "
             "information about software engineering concepts. Penalize any "
             "incorrect technical claims, misused terminology, or outdated practices.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
    model="gpt-4.1"
)

# Define a custom metric for "helpfulness"
helpfulness = GEval(
    name="Helpfulness",
    criteria="Rate how helpful and actionable the response is for the user. "
             "A helpful response directly addresses the user's question, provides "
             "clear steps or explanations, and anticipates follow-up questions.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.6
)

test_case = LLMTestCase(
    input="How do I implement retry logic in Python?",
    actual_output="You can use the tenacity library...",
    expected_output="Use exponential backoff with the tenacity library..."
)

technical_accuracy.measure(test_case)
helpfulness.measure(test_case)
```

**G-Eval parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of your custom metric |
| `criteria` | `str` | Natural language description of what to evaluate |
| `evaluation_params` | `List[LLMTestCaseParams]` | Which test case fields to use |
| `evaluation_steps` | `List[str]` | Optional explicit evaluation steps (overrides auto-generated CoT) |
| `threshold` | `float` | Minimum passing score |
| `model` | `str` | LLM judge model |

**Tip:** For reproducibility, provide explicit `evaluation_steps` instead of relying on auto-generated CoT:

```python
metric = GEval(
    name="Conciseness",
    criteria="Evaluate response conciseness",
    evaluation_steps=[
        "Count the number of sentences in the response",
        "Identify any redundant or repeated information",
        "Check if the response could be shortened without losing key information",
        "Score 10 if perfectly concise, 1 if extremely verbose"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)
```

### 4.4 DAG Metric (Directed Acyclic Graph)

**What it measures:** User-defined evaluation logic using deterministic decision trees. Unlike G-Eval (which is non-deterministic due to LLM-based CoT), DAG metrics follow a fixed evaluation path.

**When to use:** When you need deterministic, reproducible evaluation that always follows the same logic.

```python
from deepeval.metrics.dag import (
    DAGMetric,
    DeepACSNode,
    NonLLMJudgeNode, 
    LLMJudgeNode,
    BinaryJudgeNode,
    VerdictNode
)

# Define a DAG for evaluating customer service responses
dag_metric = DAGMetric(
    name="CustomerServiceQuality",
    threshold=0.5,
    tree=BinaryJudgeNode(
        criteria="Does the response directly address the customer's question?",
        yes_child=BinaryJudgeNode(
            criteria="Is the response polite and professional?",
            yes_child=VerdictNode(verdict=1.0),   # Both yes = perfect score
            no_child=VerdictNode(verdict=0.5)      # Addresses question but rude
        ),
        no_child=VerdictNode(verdict=0.0)          # Doesn't address question = fail
    )
)
```

### 4.5 BiasMetric

**What it measures:** Does the output contain biased statements related to gender, race, religion, politics, or other sensitive categories?

**Algorithm:**
1. Extract opinions and statements from the output
2. Classify each as biased or unbiased
3. Score = number of unbiased statements / total statements (higher = less bias)

**Score range:** 0.0 to 1.0 (higher = less biased)

```python
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

metric = BiasMetric(
    threshold=0.8,
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="Tell me about leadership qualities.",
    actual_output="Great leaders are decisive and strong. "
                  "Men tend to be more natural leaders than women."
)

metric.measure(test_case)
print(f"Score: {metric.score}")    # Low score due to gender bias
print(f"Reason: {metric.reason}")
```

### 4.6 ToxicityMetric

**What it measures:** Does the output contain toxic, harmful, offensive, or inappropriate content?

**Algorithm:** Similar to BiasMetric — extracts statements and classifies each for toxicity.

**Score range:** 0.0 to 1.0 (higher = less toxic)

```python
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase

metric = ToxicityMetric(
    threshold=0.9,  # Very low tolerance for toxicity
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="How do I handle a difficult coworker?",
    actual_output="Try to understand their perspective and communicate openly."
)

metric.measure(test_case)
# Expected: high score (non-toxic response)
```

### 4.7 PromptAlignmentMetric

**What it measures:** Does the output follow the instructions given in the prompt/system message?

**Required inputs:** `input`, `actual_output`, plus a list of alignment instructions

```python
from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase

metric = PromptAlignmentMetric(
    prompt_instructions=[
        "Always respond in JSON format",
        "Include a 'confidence' field with a value between 0 and 1",
        "Never mention competitor products"
    ],
    threshold=0.7,
    model="gpt-4.1"
)

test_case = LLMTestCase(
    input="What is the best CRM software?",
    actual_output='{"answer": "Our CRM offers lead tracking and analytics.", "confidence": 0.85}'
)

metric.measure(test_case)
```

### 4.8 JsonCorrectnessMetric

**What it measures:** Is the output valid JSON, and does it match an expected schema?

```python
from deepeval.metrics import JsonCorrectnessMetric
from deepeval.test_case import LLMTestCase

# With schema validation
metric = JsonCorrectnessMetric(
    expected_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    },
    threshold=1.0  # JSON must be valid and match schema
)

test_case = LLMTestCase(
    input="Extract user info from: John, 30, john@example.com",
    actual_output='{"name": "John", "age": 30, "email": "john@example.com"}'
)

metric.measure(test_case)
```

### 4.9 KnowledgeRetentionMetric

**What it measures:** In a multi-turn conversation, does the chatbot remember and correctly reference information from earlier turns?

**Required:** `ConversationalTestCase`

```python
from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase

metric = KnowledgeRetentionMetric(threshold=0.7)

convo = ConversationalTestCase(
    chatbot_role="Personal assistant",
    turns=[
        LLMTestCase(
            input="My name is Alice and I live in Boston.",
            actual_output="Nice to meet you, Alice! Boston is a great city."
        ),
        LLMTestCase(
            input="What city do I live in?",
            actual_output="You live in Boston, Alice!"  # Retained knowledge
        )
    ]
)

metric.measure(convo)
```

### 4.10 ConversationCompletenessMetric

**What it measures:** Did the chatbot successfully address all the user's needs throughout the conversation?

```python
from deepeval.metrics import ConversationCompletenessMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase

metric = ConversationCompletenessMetric(threshold=0.7)

convo = ConversationalTestCase(
    chatbot_role="Tech support agent",
    turns=[
        LLMTestCase(
            input="My printer won't connect to WiFi.",
            actual_output="I can help with that. What model is your printer?"
        ),
        LLMTestCase(
            input="It's an HP LaserJet Pro M404n.",
            actual_output="Please go to Settings > Network > Wireless Setup Wizard "
                          "and select your WiFi network."
        ),
        LLMTestCase(
            input="That worked, thanks! Also, how do I scan to PDF?",
            actual_output="Great! For scanning, use the HP Smart app..."
        )
    ]
)

metric.measure(convo)
```

### 4.11 ConversationRelevancyMetric

**What it measures:** Are the chatbot's responses relevant to the ongoing conversation?

```python
from deepeval.metrics import ConversationRelevancyMetric

metric = ConversationRelevancyMetric(threshold=0.7)
# Used the same way as ConversationCompletenessMetric with ConversationalTestCase
```

### 4.12 Agentic Metrics

#### ToolCorrectnessMetric

**What it measures:** Did the agent call the correct tools?

```python
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase

metric = ToolCorrectnessMetric(threshold=0.7)

test_case = LLMTestCase(
    input="What's the weather in NYC?",
    actual_output="The weather in NYC is 72F and sunny.",
    tools_called=["get_weather", "format_response"],
    expected_tools=["get_weather"]
)

metric.measure(test_case)
# Score considers: were expected tools called? Were unexpected tools called?
```

#### TaskCompletionMetric

**What it measures:** Did the agent successfully complete the user's task?

```python
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

metric = TaskCompletionMetric(threshold=0.7)

test_case = LLMTestCase(
    input="Book a flight from NYC to LAX for next Friday under $300.",
    actual_output="I found a flight from NYC to LAX on Friday, April 17th "
                  "for $249 on Delta. Shall I book it?",
    expected_output="A flight from NYC to LAX on the requested date for under $300."
)

metric.measure(test_case)
```

---

## 5. Configuration

### 5.1 Default LLM Judge

By default, DeepEval uses **GPT-4.1** as the evaluation judge. This is configured automatically when `OPENAI_API_KEY` is set.

### 5.2 Per-Metric Model Override

Every metric accepts a `model` parameter:

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# Use GPT-4.1 for faithfulness (needs strong reasoning)
faithfulness = FaithfulnessMetric(model="gpt-4.1", threshold=0.8)

# Use GPT-4.1-mini for answer relevancy (simpler task, save cost)
relevancy = AnswerRelevancyMetric(model="gpt-4.1-mini", threshold=0.7)
```

### 5.3 Custom LLM via DeepEvalBaseLLM

To use a non-OpenAI model (or a self-hosted model) as the judge, implement `DeepEvalBaseLLM`:

```python
from deepeval.models import DeepEvalBaseLLM
from typing import Optional

class CustomOllamaLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.1:70b"):
        self.model_name = model_name
    
    def load_model(self):
        """Load or initialize the model. Called once."""
        import ollama
        # Ollama client doesn't need explicit loading
        return None
    
    def generate(self, prompt: str, schema: Optional[dict] = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The evaluation prompt
            schema: Optional JSON schema for structured output
            
        Returns:
            The model's response as a string
        """
        import ollama
        
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json" if schema else None
        )
        return response["message"]["content"]
    
    async def a_generate(self, prompt: str, schema: Optional[dict] = None) -> str:
        """Async version of generate."""
        import ollama
        
        response = await ollama.AsyncClient().chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json" if schema else None
        )
        return response["message"]["content"]
    
    def get_model_name(self) -> str:
        """Return a string identifier for this model."""
        return f"ollama/{self.model_name}"


# Use the custom model
custom_llm = CustomOllamaLLM(model_name="llama3.1:70b")

faithfulness = FaithfulnessMetric(
    model=custom_llm,
    threshold=0.7
)
```

### 5.4 Using Azure OpenAI

```python
from deepeval.models import AzureOpenAI

azure_model = AzureOpenAI(
    model="gpt-4.1",  # or your deployment name
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview"
)

metric = FaithfulnessMetric(model=azure_model, threshold=0.8)
```

Or via environment variables:

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_DEPLOYMENT_NAME="gpt-4.1"
export AZURE_API_VERSION="2024-12-01-preview"
```

### 5.5 Using Anthropic Claude

```python
from deepeval.models import DeepEvalBaseLLM

class ClaudeLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        self.model_name = model_name
    
    def load_model(self):
        import anthropic
        self.client = anthropic.Anthropic()
        return self.client
    
    def generate(self, prompt: str, schema=None) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        import anthropic
        async_client = anthropic.AsyncAnthropic()
        message = await async_client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def get_model_name(self) -> str:
        return self.model_name

claude = ClaudeLLM()
metric = FaithfulnessMetric(model=claude, threshold=0.8)
```

### 5.6 Using Google Gemini

```python
from deepeval.models import DeepEvalBaseLLM

class GeminiLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
    
    def load_model(self):
        import google.generativeai as genai
        genai.configure()
        self.model = genai.GenerativeModel(self.model_name)
        return self.model
    
    def generate(self, prompt: str, schema=None) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text
    
    def get_model_name(self) -> str:
        return self.model_name
```

### 5.7 Using Mistral

```python
from deepeval.models import DeepEvalBaseLLM

class MistralLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.model_name = model_name
    
    def load_model(self):
        from mistralai import Mistral
        self.client = Mistral()
        return self.client
    
    def generate(self, prompt: str, schema=None) -> str:
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        from mistralai import Mistral
        async_client = Mistral()
        response = await async_client.chat.complete_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return self.model_name
```

### 5.8 Metric Options Reference

All metrics support these common options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | `float` | 0.5 | Minimum score for `is_successful` to be True |
| `model` | `str` or `DeepEvalBaseLLM` | `"gpt-4.1"` | The LLM judge model |
| `include_reason` | `bool` | True | Include human-readable reasoning |
| `async_mode` | `bool` | True | Use async LLM calls for speed |
| `strict_mode` | `bool` | False | If True, score is 0 when below threshold (binary pass/fail) |
| `verbose_mode` | `bool` | False | Print detailed evaluation steps to stdout |

**`strict_mode` behavior:**

```python
# Without strict_mode (default):
# Score = 0.6, threshold = 0.7
# → score remains 0.6, is_successful = False

# With strict_mode = True:
# Score = 0.6, threshold = 0.7
# → score becomes 0, is_successful = False

# With strict_mode = True:
# Score = 0.8, threshold = 0.7
# → score remains 0.8, is_successful = True
```

### 5.9 Custom Evaluation Templates

You can override the prompts DeepEval uses for any metric:

```python
from deepeval.metrics import FaithfulnessMetric

metric = FaithfulnessMetric(threshold=0.8)

# Override the claim extraction prompt
metric.claim_extraction_template = """
Given the following text, extract all factual claims as a JSON list.
Each claim should be a single, verifiable statement.

Text: {text}

Output format:
{{"claims": ["claim1", "claim2", ...]}}
"""

# Override the faithfulness verdict prompt  
metric.faithfulness_verdict_template = """
For each claim below, determine if it is supported by the given context.
Be strict: a claim is only supported if the context explicitly or 
strongly implies it.

Context: {context}
Claims: {claims}

For each claim, output "supported" or "unsupported".
"""
```

---

## 6. Evaluation Patterns

### 6.1 assert_test() -- Single Test Case

The simplest evaluation pattern. Evaluate one test case against one or more metrics and assert it passes:

```python
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import pytest

# Use as a pytest test
def test_rag_output():
    test_case = LLMTestCase(
        input="What is our refund policy?",
        actual_output="We offer a 30-day refund policy for all products.",
        retrieval_context=[
            "Refund Policy: All products may be returned within 30 days "
            "of purchase for a full refund."
        ]
    )
    
    faithfulness = FaithfulnessMetric(threshold=0.8)
    relevancy = AnswerRelevancyMetric(threshold=0.7)
    
    assert_test(test_case, [faithfulness, relevancy])
    # Raises AssertionError if any metric fails
```

Run with:

```bash
pytest test_rag.py -v
```

### 6.2 evaluate() -- Batch Evaluation

Evaluate multiple test cases at once and get aggregated results:

```python
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_cases = [
    LLMTestCase(
        input="What is X?",
        actual_output="X is Y.",
        retrieval_context=["X is defined as Y."]
    ),
    LLMTestCase(
        input="What is A?",
        actual_output="A is B.",
        retrieval_context=["A is defined as B."]
    ),
    # ... more test cases
]

metrics = [
    FaithfulnessMetric(threshold=0.8),
    AnswerRelevancyMetric(threshold=0.7)
]

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    run_async=True,          # Evaluate test cases in parallel
    show_indicator=True,     # Show progress bar
    print_results=True,      # Print results table
    write_cache=True,        # Cache results to avoid re-evaluation
    use_cache=False          # Don't use cached results (force fresh eval)
)

# Access results
print(f"Overall pass rate: {results.pass_rate}")
print(f"Total tests: {results.total}")
print(f"Passed: {results.passed}")
print(f"Failed: {results.failed}")

# Access individual test results
for test_result in results.test_results:
    print(f"Input: {test_result.input}")
    for metric_result in test_result.metrics_data:
        print(f"  {metric_result.name}: {metric_result.score} "
              f"({'PASS' if metric_result.success else 'FAIL'})")
```

### 6.3 deepeval test run -- CLI Execution

Run evaluations from the command line:

```bash
# Run all test files
deepeval test run test_rag.py

# Run with verbose output
deepeval test run test_rag.py -v

# Run specific tests
deepeval test run test_rag.py::test_faithfulness

# Run and push results to Confident AI
deepeval test run test_rag.py --confident
```

### 6.4 Parametrized Tests with pytest

Use pytest's parametrize decorator for data-driven testing:

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_data = [
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
        "context": ["Paris is the capital and largest city of France."]
    },
    {
        "input": "What is Python?",
        "output": "Python is a programming language created by Guido van Rossum.",
        "context": ["Python is a high-level programming language designed by Guido van Rossum."]
    },
    {
        "input": "What is the speed of light?",
        "output": "The speed of light is approximately 300,000 km/s.",
        "context": ["Light travels at exactly 299,792,458 metres per second in vacuum."]
    },
]

@pytest.mark.parametrize("data", test_data, ids=[d["input"][:30] for d in test_data])
def test_faithfulness(data):
    test_case = LLMTestCase(
        input=data["input"],
        actual_output=data["output"],
        retrieval_context=data["context"]
    )
    metric = FaithfulnessMetric(threshold=0.7)
    assert_test(test_case, [metric])
```

### 6.5 Dataset-Driven Evaluation

Load test data from a dataset:

```python
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# Load dataset from Confident AI
dataset = EvaluationDataset()
dataset.pull(alias="production_eval_v2")

# Convert goldens to test cases (you provide actual_output from your RAG system)
test_cases = []
for golden in dataset.goldens:
    # Run your RAG system to get the actual output
    actual_output = my_rag_system.query(golden.input)
    
    test_case = golden.to_test_case(actual_output=actual_output)
    test_cases.append(test_case)

# Evaluate
results = evaluate(
    test_cases=test_cases,
    metrics=[
        FaithfulnessMetric(threshold=0.8),
        AnswerRelevancyMetric(threshold=0.7)
    ]
)
```

### 6.6 Component-Level Evaluation with Tracing

DeepEval supports tracing individual RAG components using `@observe` and `update_current_span`:

```python
from deepeval.tracing import observe, update_current_span, TraceType

@observe(type=TraceType.RETRIEVER, name="vector_search")
def retrieve_documents(query: str):
    """Your retrieval logic here."""
    results = vector_store.search(query, top_k=5)
    
    # Report retriever-specific data
    update_current_span(
        input=query,
        output=[doc.text for doc in results],
        metadata={
            "top_k": 5,
            "index": "main_index",
            "num_results": len(results)
        }
    )
    return results

@observe(type=TraceType.LLM, name="generate_answer")
def generate_answer(query: str, context: list):
    """Your generation logic here."""
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    
    update_current_span(
        input=prompt,
        output=answer,
        metadata={
            "model": "gpt-4.1",
            "tokens_used": response.usage.total_tokens
        }
    )
    return answer

@observe(type=TraceType.AGENT, name="rag_pipeline")
def rag_pipeline(query: str):
    """Full RAG pipeline with tracing."""
    docs = retrieve_documents(query)
    context = [doc.text for doc in docs]
    answer = generate_answer(query, context)
    return answer, context
```

---

## 7. Synthetic Data Generation

DeepEval includes a `Synthesizer` for generating evaluation test cases from documents.

### 7.1 Synthesizer Class

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer(
    model="gpt-4.1",       # Model for generating questions
    async_mode=True         # Generate in parallel
)
```

### 7.2 generate_goldens_from_docs()

Generate test cases from raw documents:

```python
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

synthesizer = Synthesizer(model="gpt-4.1")

# From raw text documents
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[
        "data/company_policy.pdf",
        "data/product_manual.txt",
        "data/faq.docx"
    ],
    max_goldens_per_document=10,    # Max questions per document
    chunk_size=1024,                 # How to chunk documents
    chunk_overlap=200                # Overlap between chunks
)

# Save to dataset
dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="synthetic_eval_v1")

print(f"Generated {len(goldens)} test cases")
for g in goldens[:3]:
    print(f"Q: {g.input}")
    print(f"A: {g.expected_output}")
    print(f"Context: {g.context[0][:100]}...")
    print()
```

### 7.3 generate_goldens_from_contexts()

Generate test cases from pre-chunked contexts:

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()

contexts = [
    ["The company was founded in 2010 by John Smith in San Francisco."],
    ["Our flagship product, DataFlow, processes up to 1M events per second."],
    ["The free tier includes 100 API calls per day. The pro tier is $49/month."]
]

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    max_goldens_per_context=3,
    include_expected_output=True
)

for g in goldens:
    print(f"Q: {g.input}")
    print(f"Expected: {g.expected_output}")
```

### 7.4 generate_goldens_from_scratch()

Generate test cases without any source documents (useful for testing general capabilities):

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()

goldens = synthesizer.generate_goldens_from_scratch(
    subject="Python programming",
    task="Answer technical questions about Python",
    num_goldens=20
)
```

### 7.5 Controlling Generation Quality

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    Evolution,
    EvolutionConfig,
    FiltrationConfig
)

synthesizer = Synthesizer(
    model="gpt-4.1",
    # Control how questions evolve in complexity
    evolutions={
        Evolution.SIMPLE: 0.3,           # 30% simple questions
        Evolution.REASONING: 0.3,        # 30% reasoning questions
        Evolution.MULTI_CONTEXT: 0.2,    # 20% multi-context questions
        Evolution.CONDITIONAL: 0.2       # 20% conditional questions
    },
    # Filter out low-quality generations
    filtration_config=FiltrationConfig(
        max_quality_retries=3,           # Retry up to 3 times for quality
        critic_model="gpt-4.1"          # Model to judge quality
    )
)
```

---

## 8. Confident AI Platform

### 8.1 What It Adds

Confident AI is the companion SaaS platform for DeepEval. While DeepEval (the open-source library) handles evaluation logic, Confident AI adds:

| Feature | Description |
|---------|-------------|
| **Dashboards** | Visual display of metrics over time, per-metric drill-downs |
| **Regression Testing** | Automatically compare evaluation runs, detect regressions |
| **Dataset Management** | Store, version, and share evaluation datasets (push/pull) |
| **Tracing** | Visualize the full RAG pipeline trace for each query |
| **Collaboration** | Share results and datasets with team members |
| **Alerts** | Get notified when metrics drop below thresholds |

### 8.2 Login and API Key Setup

```bash
# Login via CLI (opens browser)
deepeval login

# Or set API key directly
export CONFIDENT_API_KEY="your-api-key-here"
```

### 8.3 Pushing and Pulling Datasets

```python
from deepeval.dataset import EvaluationDataset

# Push a dataset
dataset = EvaluationDataset(goldens=[...])
dataset.push(alias="my_eval_dataset_v1")

# Pull a dataset
dataset = EvaluationDataset()
dataset.pull(alias="my_eval_dataset_v1")

# List available datasets
# (via Confident AI web dashboard)
```

### 8.4 Sending Test Run Results

```bash
# Automatically send results to Confident AI
deepeval test run test_rag.py --confident

# Or programmatically
from deepeval import evaluate

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    # Results are automatically sent if CONFIDENT_API_KEY is set
)
```

---

## 9. Red Teaming (DeepTeam)

### 9.1 What It Is

DeepTeam is DeepEval's red teaming / adversarial testing module. It generates adversarial inputs designed to make your LLM fail in specific ways (produce toxic content, leak system prompts, bypass safety guardrails, etc.).

### 9.2 Available Vulnerability Types

| Vulnerability | Description |
|--------------|-------------|
| **Prompt Injection** | Attempts to override system instructions |
| **Jailbreaking** | Attempts to bypass safety guardrails |
| **PII Leakage** | Attempts to extract personal information |
| **Intellectual Property** | Attempts to extract proprietary information |
| **Toxicity** | Attempts to generate toxic, harmful, or offensive content |
| **Bias** | Attempts to generate biased outputs |
| **Misinformation** | Attempts to generate false information |
| **Illegal Activity** | Attempts to generate instructions for illegal activities |
| **Excessive Agency** | Attempts to make the LLM take unauthorized actions |
| **Hallucination** | Prompts designed to make the LLM fabricate information |
| **Data Poisoning** | Attempts to corrupt the LLM's behavior |
| **System Prompt Leakage** | Attempts to make the LLM reveal its system prompt |

### 9.3 How to Use It

```python
from deepeval.red_teaming import RedTeamer, AttackEnhancement
from deepeval.vulnerability import (
    PromptInjection,
    Jailbreaking,
    PIILeakage,
    Toxicity,
    Bias
)

# Initialize red teamer
red_teamer = RedTeamer(
    model="gpt-4.1",
    target_model_callback=my_llm_callback  # Function that calls your LLM
)

# Define callback for your system
def my_llm_callback(prompt: str) -> str:
    """Your LLM system that will be tested."""
    response = my_rag_system.query(prompt)
    return response

# Run red teaming
results = red_teamer.scan(
    vulnerabilities=[
        PromptInjection(),
        Jailbreaking(),
        PIILeakage(),
        Toxicity(),
        Bias()
    ],
    attack_enhancements=[
        AttackEnhancement.BASE64,          # Encode attacks in base64
        AttackEnhancement.ROT13,           # ROT13 encoding
        AttackEnhancement.LEETSPEAK,       # L33t speak encoding
        AttackEnhancement.PROMPT_INJECTION, # Nested prompt injection
        AttackEnhancement.JAILBREAK_LINEAR  # Linear jailbreak escalation
    ],
    attacks_per_vulnerability=10   # Number of attack attempts per type
)

# Analyze results
print(f"Total attacks: {results.total}")
print(f"Successful attacks: {results.successful}")
print(f"Vulnerability breakdown:")
for vuln_result in results.vulnerability_results:
    print(f"  {vuln_result.vulnerability}: "
          f"{vuln_result.successful}/{vuln_result.total} attacks succeeded")
```

---

## 10. Best Practices and Gotchas

### 10.1 The `input` Field Trap

**The most common DeepEval mistake:** Putting the full prompt template (with system instructions, context, etc.) in the `input` field.

**WRONG:**

```python
# DO NOT DO THIS
test_case = LLMTestCase(
    input="""You are a helpful assistant. Use the following context to answer.
    
Context: {context}

Question: What is the refund policy?

Answer in a concise manner.""",
    actual_output="We offer 30-day refunds.",
    retrieval_context=["Refund policy: 30 day returns accepted."]
)
```

**RIGHT:**

```python
# Do this instead
test_case = LLMTestCase(
    input="What is the refund policy?",  # Just the user's question
    actual_output="We offer 30-day refunds.",
    retrieval_context=["Refund policy: 30 day returns accepted."]
)
```

**Why this matters:** The `AnswerRelevancyMetric` checks if the output is relevant to the `input`. If `input` contains system instructions, the metric will check if the output is relevant to those instructions, not the user's question. This inflates or distorts scores.

**Rule:** `input` should contain ONLY the user's query/question. System prompts, instructions, and context formatting belong elsewhere.

### 10.2 HallucinationMetric vs FaithfulnessMetric Confusion

This is the second most common mistake:

| Scenario | Correct Metric | Uses |
|----------|---------------|------|
| Evaluating a RAG system's output against retrieved context | **FaithfulnessMetric** | `retrieval_context` |
| Checking if a summary contradicts the source text | **HallucinationMetric** | `context` |
| Checking if an LLM response contradicts known ground truth | **HallucinationMetric** | `context` |
| Checking if an LLM response is grounded in what the retriever found | **FaithfulnessMetric** | `retrieval_context` |

**Key difference in scoring:**
- **FaithfulnessMetric:** Higher score = more faithful (good). Score of 1.0 means all claims are supported.
- **HallucinationMetric:** Higher score = more hallucination (bad). Score of 0.0 means no hallucination.

**This inverted scoring is a common source of confusion.** Double-check which metric you are using and what the score direction means.

### 10.3 Cost Management

LLM-as-judge evaluation can be expensive at scale. Here are strategies to manage costs:

| Strategy | Savings | Tradeoff |
|----------|---------|----------|
| Use GPT-4.1-mini for simpler metrics | 50-80% | Slightly less accurate for nuanced judgments |
| Cache evaluation results | Avoids re-evaluation | May miss regressions if cache is stale |
| Reduce test set size for CI/CD | Proportional to size reduction | Less statistical power |
| Use async evaluation | Time savings (not cost) | May hit rate limits |
| Batch similar test cases | Marginal savings | Complexity |
| Reserve GPT-4.1 for Faithfulness only | ~50% overall | Other metrics may be slightly less accurate |

**Cost estimation formula:**

```python
def estimate_evaluation_cost(
    num_test_cases: int,
    num_metrics: int,
    avg_input_tokens_per_call: int = 3000,
    avg_output_tokens_per_call: int = 500,
    llm_calls_per_metric: int = 2,
    input_price_per_1k: float = 0.002,   # GPT-4.1-mini pricing
    output_price_per_1k: float = 0.008
):
    total_calls = num_test_cases * num_metrics * llm_calls_per_metric
    input_cost = total_calls * avg_input_tokens_per_call * input_price_per_1k / 1000
    output_cost = total_calls * avg_output_tokens_per_call * output_price_per_1k / 1000
    total = input_cost + output_cost
    return {
        "total_llm_calls": total_calls,
        "estimated_cost_usd": round(total, 2),
        "input_cost": round(input_cost, 2),
        "output_cost": round(output_cost, 2)
    }

# Example: 200 test cases, 4 metrics, using GPT-4.1-mini
cost = estimate_evaluation_cost(200, 4)
print(cost)
# {'total_llm_calls': 1600, 'estimated_cost_usd': 16.0, ...}

# With GPT-4.1 (roughly 10x more expensive)
cost_gpt4 = estimate_evaluation_cost(
    200, 4,
    input_price_per_1k=0.002,
    output_price_per_1k=0.008
)
```

### 10.4 Non-Determinism Handling

LLM-as-judge metrics are inherently non-deterministic. The same test case can receive different scores on different runs.

**Practical strategies:**

```python
import numpy as np
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def evaluate_with_confidence(test_cases, metrics, n_runs=3):
    """
    Run evaluation multiple times and compute mean +/- std for each metric.
    """
    all_scores = {metric.__class__.__name__: [] for metric in metrics}
    
    for run in range(n_runs):
        results = evaluate(
            test_cases=test_cases,
            metrics=[m.__class__(threshold=m.threshold) for m in metrics],
            print_results=False
        )
        
        for test_result in results.test_results:
            for metric_data in test_result.metrics_data:
                all_scores[metric_data.name].append(metric_data.score)
    
    # Compute statistics
    for metric_name, scores in all_scores.items():
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{metric_name}: {mean:.3f} +/- {std:.3f} "
              f"(min={min(scores):.3f}, max={max(scores):.3f})")
    
    return all_scores
```

**Expected variance by metric:**

| Metric | Typical Std Dev (across runs) | Notes |
|--------|------------------------------|-------|
| FaithfulnessMetric | 0.02-0.08 | Relatively stable |
| AnswerRelevancyMetric | 0.03-0.10 | Moderate variance |
| ContextualPrecisionMetric | 0.02-0.05 | Stable (binary relevance judgments) |
| ContextualRecallMetric | 0.03-0.08 | Moderate variance |
| GEval | 0.05-0.15 | Higher variance (CoT generation varies) |
| BiasMetric | 0.05-0.12 | Can vary depending on edge cases |

### 10.5 Version Pinning

DeepEval is actively developed and metrics can change between versions. Always pin your version:

```bash
# In requirements.txt
deepeval==3.1.1

# Or with a range
deepeval>=3.0,<4.0
```

**Why this matters:**
- Metric algorithms may be updated (changing scores for the same inputs)
- New required parameters may be added
- Default models may change
- Score normalization may change

**Best practice:** When starting a new evaluation project, pin to a specific version. When upgrading, re-run your full test suite and compare scores to detect any metric drift.

### 10.6 Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `OPENAI_API_KEY not set` | Missing API key | Set `OPENAI_API_KEY` environment variable |
| `retrieval_context is required` | Metric needs `retrieval_context` but test case does not have it | Add `retrieval_context` to your `LLMTestCase` |
| `expected_output is required` | Metric needs ground truth | Add `expected_output` or use a metric that doesn't need it |
| `Rate limit exceeded` | Too many parallel API calls | Reduce concurrency, add delays, or use batch API |
| `Context length exceeded` | Very long contexts exceed judge model's window | Truncate context, use a model with larger context window |
| `Invalid JSON in response` | Judge LLM returned malformed JSON | Retry, or use a more capable model as judge |
| `Score is None` | Evaluation failed silently | Check `verbose_mode=True` for details, check API connectivity |

### 10.7 Testing Strategy: A Complete Example

Here is a complete, production-ready evaluation setup:

```python
# file: tests/test_rag_evaluation.py

import pytest
import json
import os
from pathlib import Path
from deepeval import assert_test, evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase


# ── Configuration ──────────────────────────────────────

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4.1-mini")
FAITHFULNESS_THRESHOLD = 0.8
RELEVANCY_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.7
CONTEXT_RELEVANCY_THRESHOLD = 0.5


# ── Load Test Data ─────────────────────────────────────

def load_test_cases():
    """Load test cases from JSON file."""
    test_data_path = Path(__file__).parent / "data" / "eval_dataset.json"
    with open(test_data_path) as f:
        data = json.load(f)
    
    test_cases = []
    for item in data:
        # Run your RAG system to get actual output and retrieval context
        from my_rag_system import query_rag
        result = query_rag(item["query"])
        
        tc = LLMTestCase(
            input=item["query"],
            actual_output=result["answer"],
            expected_output=item.get("expected_answer"),
            retrieval_context=result["retrieved_chunks"]
        )
        test_cases.append(tc)
    
    return test_cases


# ── Metrics ────────────────────────────────────────────

@pytest.fixture
def faithfulness_metric():
    return FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True
    )

@pytest.fixture
def relevancy_metric():
    return AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True
    )

@pytest.fixture
def context_precision_metric():
    return ContextualPrecisionMetric(
        threshold=CONTEXT_PRECISION_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True
    )

@pytest.fixture
def context_recall_metric():
    return ContextualRecallMetric(
        threshold=CONTEXT_RECALL_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True
    )

@pytest.fixture
def context_relevancy_metric():
    return ContextualRelevancyMetric(
        threshold=CONTEXT_RELEVANCY_THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
        async_mode=True
    )


# ── Individual Metric Tests ────────────────────────────

TEST_CASES = load_test_cases()

@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc.input[:50] for tc in TEST_CASES])
def test_faithfulness(test_case, faithfulness_metric):
    assert_test(test_case, [faithfulness_metric])

@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc.input[:50] for tc in TEST_CASES])
def test_answer_relevancy(test_case, relevancy_metric):
    assert_test(test_case, [relevancy_metric])

@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc.input[:50] for tc in TEST_CASES])
def test_context_precision(test_case, context_precision_metric):
    if test_case.expected_output:  # Only run if ground truth available
        assert_test(test_case, [context_precision_metric])

@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc.input[:50] for tc in TEST_CASES])
def test_context_recall(test_case, context_recall_metric):
    if test_case.expected_output:  # Only run if ground truth available
        assert_test(test_case, [context_recall_metric])


# ── Batch Evaluation ───────────────────────────────────

def test_batch_evaluation():
    """Run all metrics on all test cases and verify aggregate pass rate."""
    metrics = [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=JUDGE_MODEL),
        AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=JUDGE_MODEL),
        ContextualRelevancyMetric(threshold=CONTEXT_RELEVANCY_THRESHOLD, model=JUDGE_MODEL),
    ]
    
    results = evaluate(
        test_cases=TEST_CASES,
        metrics=metrics,
        run_async=True,
        print_results=True
    )
    
    # Assert minimum aggregate pass rate
    assert results.pass_rate >= 0.8, (
        f"Overall pass rate {results.pass_rate:.2%} is below 80% threshold. "
        f"Passed: {results.passed}/{results.total}"
    )
```

Run the tests:

```bash
# Run all RAG evaluation tests
pytest tests/test_rag_evaluation.py -v --tb=short

# Run only faithfulness tests
pytest tests/test_rag_evaluation.py -k "faithfulness" -v

# Run batch evaluation only
pytest tests/test_rag_evaluation.py::test_batch_evaluation -v

# Run with Confident AI integration
deepeval test run tests/test_rag_evaluation.py --confident
```

### 10.8 Metric Selection Quick Reference

| I want to check... | Use this metric | Requires |
|--------------------|-----------------|----------|
| Is the answer grounded in the retrieved docs? | `FaithfulnessMetric` | `actual_output`, `retrieval_context` |
| Does the answer address the question? | `AnswerRelevancyMetric` | `input`, `actual_output` |
| Are retrieved chunks relevant? | `ContextualRelevancyMetric` | `input`, `retrieval_context` |
| Are relevant chunks ranked first? | `ContextualPrecisionMetric` | `input`, `retrieval_context`, `expected_output` |
| Did we find all needed info? | `ContextualRecallMetric` | `input`, `retrieval_context`, `expected_output` |
| Is the answer factually correct? | `AnswerCorrectnessMetric` | `actual_output`, `expected_output` |
| Does the output contradict known facts? | `HallucinationMetric` | `actual_output`, `context` |
| Is the summary good? | `SummarizationMetric` | `actual_output`, `context` |
| Custom criteria | `GEval` | Configurable |
| Deterministic eval logic | `DAGMetric` | Configurable |
| Is output biased? | `BiasMetric` | `actual_output` |
| Is output toxic? | `ToxicityMetric` | `actual_output` |
| Does output follow instructions? | `PromptAlignmentMetric` | `input`, `actual_output`, instructions |
| Is output valid JSON? | `JsonCorrectnessMetric` | `actual_output`, schema |
| Does chatbot remember context? | `KnowledgeRetentionMetric` | `ConversationalTestCase` |
| Did chatbot complete the task? | `ConversationCompletenessMetric` | `ConversationalTestCase` |
| Did agent use right tools? | `ToolCorrectnessMetric` | `tools_called`, `expected_tools` |
| Did agent complete the task? | `TaskCompletionMetric` | `input`, `actual_output`, `expected_output` |

### 10.9 Comparison with RAGAS

| Feature | DeepEval | RAGAS |
|---------|----------|-------|
| **Primary interface** | pytest-native | Standalone / LangChain |
| **Number of metrics** | 50+ | ~10-15 core |
| **Agentic evaluation** | Yes (ToolCorrectness, TaskCompletion) | Limited |
| **Conversational evaluation** | Yes (multi-turn metrics) | Limited |
| **Red teaming** | Yes (DeepTeam) | No |
| **Synthetic data generation** | Yes (Synthesizer) | Yes (TestsetGenerator) |
| **Platform/dashboard** | Confident AI | Ragas app / third-party |
| **Default LLM judge** | GPT-4.1 | GPT-4o |
| **Custom LLM support** | DeepEvalBaseLLM | LangChain LLMs |
| **Async support** | Native | Via LangChain |
| **CI/CD integration** | pytest + CLI | Separate integration needed |
| **Learning curve** | Moderate | Lower for basic use |
| **Community size** | Growing rapidly | Large, established |
| **Open source license** | Apache 2.0 | Apache 2.0 |

### 10.10 Troubleshooting Checklist

When your evaluation produces unexpected results, work through this checklist:

1. **Check `input` field:** Does it contain ONLY the user query? (Not the full prompt template)
2. **Check `context` vs `retrieval_context`:** Are you using the right field for your metric?
3. **Check score direction:** Remember HallucinationMetric is inverted (lower = better)
4. **Enable verbose mode:** Set `verbose_mode=True` to see exactly what the metric is doing
5. **Check the judge model:** Is the judge model capable enough? (GPT-4.1-mini may struggle with nuanced faithfulness checks)
6. **Check context length:** Are your contexts being truncated by the judge model's context window?
7. **Run multiple times:** Is the score stable across runs? If variance is high, increase test set size or use multiple runs
8. **Check DeepEval version:** Did you recently upgrade? Metric behavior may have changed
9. **Check API connectivity:** Ensure your OpenAI API key is valid and has sufficient quota
10. **Check threshold vs score:** A "failed" test might still have a reasonable score — adjust thresholds appropriately

---

## Appendix A: Complete Metric Parameter Reference

### FaithfulnessMetric

```python
FaithfulnessMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    truths_extraction_limit: int | None = None,
    penalize_ambiguous_claims: bool = False
)
```

### AnswerRelevancyMetric

```python
AnswerRelevancyMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### ContextualRelevancyMetric

```python
ContextualRelevancyMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### ContextualPrecisionMetric

```python
ContextualPrecisionMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### ContextualRecallMetric

```python
ContextualRecallMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### HallucinationMetric

```python
HallucinationMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### GEval

```python
GEval(
    name: str,
    criteria: str,
    evaluation_params: list[LLMTestCaseParams],
    evaluation_steps: list[str] | None = None,
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### BiasMetric

```python
BiasMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

### ToxicityMetric

```python
ToxicityMetric(
    threshold: float = 0.5,
    model: str | DeepEvalBaseLLM = "gpt-4.1",
    include_reason: bool = True,
    async_mode: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False
)
```

---

## Appendix B: DeepEval CLI Reference

| Command | Description |
|---------|-------------|
| `deepeval --version` | Show installed version |
| `deepeval login` | Authenticate with Confident AI |
| `deepeval test run <file>` | Run evaluation tests |
| `deepeval test run <file> -v` | Verbose output |
| `deepeval test run <file> --confident` | Run and push results to Confident AI |
| `deepeval test run <file> -k "keyword"` | Run tests matching keyword |
| `deepeval test run <file> -n 4` | Run with 4 parallel workers |
| `deepeval synthesize` | Generate synthetic test data |

---

## Appendix C: Environment Variable Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for default judge | Yes (unless custom LLM) |
| `CONFIDENT_API_KEY` | Confident AI platform key | No |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | Only for Azure |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL | Only for Azure |
| `AZURE_DEPLOYMENT_NAME` | Azure deployment name | Only for Azure |
| `AZURE_API_VERSION` | Azure API version | Only for Azure |
| `DEEPEVAL_RESULTS_FOLDER` | Custom folder for test results | No |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | Disable telemetry (`"YES"`) | No |

---

**Next:** [06 - RAGAS Complete Guide](06_ragas_complete_guide.md) -- The complete reference for the RAGAS evaluation framework.

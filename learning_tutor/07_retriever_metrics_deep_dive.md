# 07 -- Retriever Metrics Deep Dive

## Table of Contents

1. [Why Retriever Evaluation Matters](#why-retriever-evaluation-matters)
2. [What Makes a Good Retrieval](#what-makes-a-good-retrieval)
3. [Traditional IR Metrics (Background)](#traditional-ir-metrics-background)
4. [DeepEval Retriever Metrics](#deepeval-retriever-metrics)
5. [RAGAS Retriever Metrics](#ragas-retriever-metrics)
6. [Cross-Framework Comparison](#cross-framework-comparison)
7. [Practical Guide: Diagnosing Retriever Problems](#practical-guide-diagnosing-retriever-problems)
8. [Worked Example with Sample Data](#worked-example-with-sample-data)

---

## Why Retriever Evaluation Matters

### Garbage In, Garbage Out

The retriever is the **foundation** of every RAG pipeline. No matter how powerful your generator
LLM is, it cannot produce high-quality answers from low-quality context. This principle cannot
be overstated:

- If the retriever returns **irrelevant** chunks, the generator will either hallucinate (make
  things up) or produce off-topic responses
- If the retriever **misses** key information, the generator cannot include it in the answer
- If the retriever returns relevant chunks but **ranks them poorly**, the generator may
  prioritize less relevant information (especially with limited context windows)

A well-tuned generator on bad retrievals will produce confidently wrong answers. A mediocre
generator on excellent retrievals will still produce useful answers. This asymmetry means that
**retriever quality has higher leverage than generator quality** in most RAG systems.

### The Three Retriever Failure Modes

1. **Irrelevant retrieval** (low precision): The retriever returns chunks that are not related
   to the query. This wastes context window space and can mislead the generator.

2. **Incomplete retrieval** (low recall): The retriever misses chunks that contain the answer
   or key supporting evidence. The generator cannot include information it never received.

3. **Poor ranking** (low ranking quality): The retriever returns relevant chunks but buries
   them below irrelevant ones. Since LLMs pay more attention to early context (the "lost in
   the middle" problem), this effectively degrades answer quality.

Each of these failure modes is measured by different metrics, and diagnosing which failure mode
is dominant tells you exactly what to fix in your RAG pipeline.

### The Retriever Optimization Stack

Understanding retriever metrics helps you optimize each component of the retrieval stack:

| Component | What It Does | Metrics That Evaluate It |
|-----------|-------------|--------------------------|
| **Embedding model** | Encodes queries and documents into vectors | Context Recall, Contextual Recall |
| **Chunking strategy** | Splits documents into retrievable units | Contextual Relevancy, Context Entity Recall |
| **Top-K selection** | How many chunks to retrieve | Contextual Relevancy (precision), Context Recall |
| **Reranker** | Re-orders retrieved chunks by relevance | Context Precision, Contextual Precision |
| **Hybrid search** | Combines vector + keyword search | All retriever metrics (holistic improvement) |

---

## What Makes a Good Retrieval

Before diving into metrics, establish what "good" means for a retriever:

### The Ideal Retrieval

For a query Q and a knowledge base K, the ideal retrieval R* would:

1. **Include all chunks** from K that contain information needed to answer Q (high recall)
2. **Exclude all chunks** from K that do not contain useful information for Q (high precision)
3. **Rank the most relevant chunks first** (high ranking quality)
4. **Be efficient** in the number of chunks returned (the smallest R* that satisfies points 1-3)

### Real-World Tradeoffs

In practice, these goals conflict:

- **Precision vs. Recall**: Retrieving more chunks (higher K) increases recall but decreases
  precision. Retrieving fewer chunks does the opposite.
- **Speed vs. Quality**: More sophisticated retrieval (reranking, hybrid search) improves quality
  but adds latency and cost.
- **Chunk size vs. Granularity**: Larger chunks provide more context but may include more
  irrelevant content. Smaller chunks are more focused but may miss surrounding context.

Retriever metrics help you find the optimal tradeoff for your specific use case.

---

## Traditional IR Metrics (Background)

Before RAG, Information Retrieval (IR) research developed a rich set of metrics. Understanding
these provides the foundation for RAG-specific retriever metrics.

### Precision@K

**Definition**: Of the top-K retrieved documents, what fraction is relevant?

**Formula**:

```
Precision@K = |{relevant documents in top-K}| / K
```

**Example**:
- Query: "Python programming tutorials"
- Top-5 results: [relevant, irrelevant, relevant, relevant, irrelevant]
- Precision@5 = 3/5 = 0.6

**Interpretation**: 60% of the retrieved documents are relevant. The other 40% is noise.

**Limitation**: Does not consider the order of results. [R, I, I, I, I] and [I, I, I, I, R]
both have Precision@5 = 0.2, but the first is clearly better for a user.

### Recall@K

**Definition**: Of all relevant documents in the corpus, what fraction appears in the top-K?

**Formula**:

```
Recall@K = |{relevant documents in top-K}| / |{all relevant documents in corpus}|
```

**Example**:
- Total relevant documents in corpus: 10
- Relevant documents in top-5: 3
- Recall@5 = 3/10 = 0.3

**Interpretation**: The retriever found 30% of all relevant documents.

**Limitation**: Requires knowing the total number of relevant documents in the corpus (expensive
to compute and often requires human annotation).

### Mean Reciprocal Rank (MRR)

**Definition**: The average of the reciprocal of the rank of the first relevant document across
all queries.

**Formula**:

```
MRR = (1/|Q|) * Sum_i(1 / rank_i)
```

Where `rank_i` is the position of the first relevant result for query i.

**Example**:
- Query 1: first relevant result at position 1 -> 1/1 = 1.0
- Query 2: first relevant result at position 3 -> 1/3 = 0.333
- Query 3: first relevant result at position 2 -> 1/2 = 0.5
- MRR = (1.0 + 0.333 + 0.5) / 3 = 0.611

**Interpretation**: On average, the first relevant result appears around position 1.6. Higher MRR
means relevant results appear earlier.

### Normalized Discounted Cumulative Gain (NDCG)

**Definition**: Measures the quality of ranking, giving higher weight to relevant documents at
higher positions.

**Formula**:

```
DCG@K = Sum_{i=1}^{K} (relevance_i / log2(i+1))
IDCG@K = DCG@K of the ideal ranking
NDCG@K = DCG@K / IDCG@K
```

**Example**:
- Relevance scores: [3, 0, 2, 1, 0] (positions 1-5)
- DCG@5 = 3/log2(2) + 0/log2(3) + 2/log2(4) + 1/log2(5) + 0/log2(6)
        = 3/1 + 0 + 2/2 + 1/2.32 + 0
        = 3 + 0 + 1 + 0.43
        = 4.43
- Ideal ranking: [3, 2, 1, 0, 0]
- IDCG@5 = 3/1 + 2/1.58 + 1/2 + 0 + 0 = 3 + 1.26 + 0.5 = 4.76
- NDCG@5 = 4.43 / 4.76 = 0.93

**Interpretation**: The ranking is 93% as good as the ideal ranking. NDCG handles graded
relevance (not just binary) and is sensitive to ranking position.

### Mean Average Precision (MAP)

**Definition**: The mean of Average Precision across all queries.

**Formula**:

```
AP(q) = (1/|relevant docs|) * Sum_{k=1}^{n}(Precision@k * relevance_k)
MAP = (1/|Q|) * Sum_{q in Q}(AP(q))
```

**Example**:
- 4 relevant documents total
- Results: [R, I, R, I, R, I, I, R] (R=relevant, I=irrelevant)
- Precision@1 = 1/1 * 1 = 1.0
- Precision@3 = 2/3 * 1 = 0.667
- Precision@5 = 3/5 * 1 = 0.6
- Precision@8 = 4/8 * 1 = 0.5
- AP = (1.0 + 0.667 + 0.6 + 0.5) / 4 = 0.692

**Interpretation**: Rewards relevant documents appearing earlier. Higher MAP means relevant
documents are consistently ranked higher.

### Hit Rate (Hit@K)

**Definition**: The fraction of queries where at least one relevant document appears in the top-K.

**Formula**:

```
Hit@K = |{queries with at least one relevant result in top-K}| / |{all queries}|
```

**Example**: Out of 100 queries, 85 have at least one relevant document in the top-5.
Hit@5 = 85/100 = 0.85.

**Interpretation**: 85% of queries find at least one useful result. Simple but useful for
understanding basic retrieval reliability.

### How Traditional IR Metrics Relate to RAG

In the RAG context:
- "Documents" become "context chunks"
- "Queries" are user questions
- "Relevance" means "contains information needed to answer the question"

Traditional IR metrics can be applied to RAG retrievers, but RAG introduces additional
considerations:
1. **The generator adds noise**: Even with perfect retrieval, the generator may hallucinate
2. **Relevance is answer-dependent**: A chunk may be topically related but not contain the
   specific fact needed
3. **Context interaction matters**: Multiple chunks may need to be combined to form a complete answer

This is why RAG-specific metrics like those in DeepEval and RAGAS exist.

---

## DeepEval Retriever Metrics

### 1. ContextualRelevancyMetric

**Purpose**: Measures how relevant the retrieved context is to the user's query. Evaluates whether
the retriever is returning useful information.

**Type**: LLM-as-judge, single-turn, **referenceless** (no expected_output needed)

**What it targets in your pipeline**: Chunk size, top-K parameter, embedding model quality

**Algorithm (step-by-step)**:
1. An LLM extracts all individual **statements** from the `retrieval_context`
2. For each statement, the LLM classifies it as **relevant** or **irrelevant** to the `input`
3. The score is computed as the ratio of relevant statements

**Formula**:

```
Contextual Relevancy = Number of Relevant Statements / Total Number of Statements
```

**Required test case fields**:
- `input` (user query)
- `actual_output` (LLM response)
- `retrieval_context` (retrieved chunks)

**Does NOT require**: `expected_output`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing score |
| `model` | str/DeepEvalBaseLLM | gpt-4.1 | Judge LLM |
| `include_reason` | bool | True | Output explanation |
| `strict_mode` | bool | False | Binary scoring (1 or 0) |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | ContextualRelevancyTemplate | default | Custom prompts |

**Code example**:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

metric = ContextualRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="What is the return policy for electronics?",
    actual_output="Electronics can be returned within 30 days with receipt.",
    retrieval_context=[
        "All electronics purchases can be returned within 30 days with a valid receipt for a full refund.",
        "Our store is located at 123 Main Street and is open Monday through Saturday.",
        "Extended warranties are available for all electronics at an additional cost.",
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")    # e.g., 0.67 (2 of 3 contexts are relevant)
print(f"Reason: {metric.reason}")
```

**Step-by-step calculation example**:

Given the retrieval_context above, the LLM would extract statements from each context chunk:

Chunk 1: "Electronics can be returned within 30 days with receipt for full refund"
- Statement: "Electronics returns within 30 days with receipt" -> RELEVANT to return policy query

Chunk 2: "Store is at 123 Main Street, open Monday-Saturday"
- Statement: "Store location is 123 Main Street" -> IRRELEVANT
- Statement: "Store hours are Monday-Saturday" -> IRRELEVANT

Chunk 3: "Extended warranties available for electronics at additional cost"
- Statement: "Extended warranties available" -> IRRELEVANT (related to electronics but not return policy)

Score = 1 relevant / 4 total = 0.25

(Note: Actual scoring depends on how the LLM extracts and classifies statements. Some implementations
evaluate at the chunk level rather than statement level.)

**Interpretation guide**:

| Score | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | Almost all retrieved content is relevant | Retriever is working well |
| 0.7-0.9 | Most content relevant, some noise | Consider reducing top-K or improving chunking |
| 0.5-0.7 | Mixed relevance | Review embedding model and chunking strategy |
| 0.3-0.5 | More noise than signal | Significant retriever problems |
| 0.0-0.3 | Retrieved context is mostly irrelevant | Fundamental retrieval failure |

---

### 2. ContextualPrecisionMetric

**Purpose**: Measures whether relevant retrieval nodes are **ranked higher** than irrelevant ones.
This specifically evaluates the **ordering/ranking** quality of the retrieval.

**Type**: LLM-as-judge, single-turn, **reference-based** (needs expected_output)

**What it targets in your pipeline**: Reranker quality, ranking algorithm

**Algorithm (step-by-step)**:
1. An LLM judge evaluates each node/chunk in `retrieval_context` for relevance to the `input`,
   using `expected_output` as ground truth reference
2. Each node receives a binary relevance verdict (relevant or irrelevant)
3. Weighted Cumulative Precision (WCP) is computed

**Formula (Weighted Cumulative Precision)**:

```
WCP = (1/R) * Sum_{k=1}^{n} (Precision@k * r_k)

Where:
  R = total number of relevant nodes
  k = position (1-indexed)
  n = total number of nodes
  r_k = 1 if node at position k is relevant, 0 otherwise
  Precision@k = (number of relevant nodes in positions 1..k) / k
```

**Why WCP instead of simple precision?**:
- Simple precision treats [Relevant, Irrelevant, Relevant] and [Irrelevant, Relevant, Relevant]
  equally (both have precision = 2/3)
- WCP penalizes the second case because a relevant document was "pushed down" by an irrelevant one
- This matters because LLMs pay more attention to context that appears earlier ("lost in the middle")

**Required test case fields**:
- `input` (user query)
- `actual_output` (LLM response)
- `expected_output` (ground truth answer)
- `retrieval_context` (retrieved chunks, in order)

**Parameters**: Same as ContextualRelevancyMetric (threshold, model, include_reason, strict_mode,
async_mode, verbose_mode, evaluation_template using ContextualPrecisionTemplate)

**Code example**:

```python
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="What year was the Eiffel Tower built?",
    actual_output="The Eiffel Tower was built in 1889.",
    expected_output="The Eiffel Tower was constructed in 1889 for the World's Fair.",
    retrieval_context=[
        "The Eiffel Tower is a popular tourist destination in Paris.",        # Irrelevant to "what year"
        "The Eiffel Tower was completed in 1889 for the World's Fair.",       # Relevant
        "Gustave Eiffel's company designed and built the tower.",             # Somewhat relevant
        "The tower is 330 meters tall and was the tallest structure until 1930.", # Irrelevant to "what year"
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
```

**Step-by-step calculation example**:

Assume the LLM judges relevance as: [0, 1, 1, 0] (positions 1-4)

R = 2 (total relevant nodes)

Position 1 (irrelevant, r_1=0): Precision@1 = 0/1 = 0 -> contributes 0 * 0 = 0
Position 2 (relevant, r_2=1):   Precision@2 = 1/2 = 0.5 -> contributes 0.5 * 1 = 0.5
Position 3 (relevant, r_3=1):   Precision@3 = 2/3 = 0.67 -> contributes 0.67 * 1 = 0.67
Position 4 (irrelevant, r_4=0): Precision@4 = 2/4 = 0.5 -> contributes 0.5 * 0 = 0

WCP = (1/2) * (0 + 0.5 + 0.67 + 0) = (1/2) * 1.17 = 0.585

Now compare if the relevant chunks were at positions 1 and 2 instead: [1, 1, 0, 0]

Position 1 (relevant, r_1=1):   Precision@1 = 1/1 = 1.0 -> contributes 1.0
Position 2 (relevant, r_2=1):   Precision@2 = 2/2 = 1.0 -> contributes 1.0
Position 3 (irrelevant, r_3=0): contributes 0
Position 4 (irrelevant, r_4=0): contributes 0

WCP = (1/2) * (1.0 + 1.0) = 1.0 (perfect score)

This demonstrates how WCP rewards relevant documents being ranked higher.

---

### 3. ContextualRecallMetric

**Purpose**: Measures how much of the **expected answer** can be attributed to the retrieved
context. If the retriever missed key information, Contextual Recall will be low.

**Type**: LLM-as-judge, single-turn, **reference-based** (needs expected_output)

**What it targets in your pipeline**: Embedding model quality, knowledge base coverage

**Algorithm (step-by-step)**:
1. An LLM decomposes the `expected_output` into individual **statements/claims**
2. For each statement, the LLM checks whether it can be **attributed** to any node in
   `retrieval_context`
3. The score is the fraction of attributable statements

**Formula**:

```
Contextual Recall = Number of Attributable Statements / Total Number of Statements
```

Note: Statements are extracted from `expected_output` (not `actual_output`), because we are
measuring retriever quality against the ideal answer, not the generated answer.

**Required test case fields**:
- `input` (user query)
- `actual_output` (LLM response)
- `expected_output` (ground truth answer)
- `retrieval_context` (retrieved chunks)

**Parameters**: Same structure (threshold=0.5, model=gpt-4.1, include_reason, strict_mode,
async_mode, verbose_mode, evaluation_template using ContextualRecallTemplate)

**Code example**:

```python
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="What are the health benefits of green tea?",
    actual_output="Green tea has antioxidants and may help with weight loss.",
    expected_output="Green tea contains antioxidants called catechins that can reduce inflammation, "
                    "support weight management, improve brain function, and lower the risk of heart disease.",
    retrieval_context=[
        "Green tea is rich in catechins, a type of antioxidant that helps reduce inflammation.",
        "Studies show that green tea extract can boost metabolism and support weight management.",
        "The history of green tea dates back to ancient China over 4000 years ago.",
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
```

**Step-by-step calculation example**:

Statements extracted from expected_output:
1. "Green tea contains antioxidants called catechins" -> Can be attributed to chunk 1. ATTRIBUTABLE.
2. "Catechins can reduce inflammation" -> Can be attributed to chunk 1. ATTRIBUTABLE.
3. "Green tea supports weight management" -> Can be attributed to chunk 2. ATTRIBUTABLE.
4. "Green tea improves brain function" -> NOT found in any context. NOT ATTRIBUTABLE.
5. "Green tea lowers the risk of heart disease" -> NOT found in any context. NOT ATTRIBUTABLE.

Score = 3/5 = 0.6

**Interpretation**: The retriever captured 60% of the information needed for the ideal answer.
It missed information about brain function and heart disease — these facts exist somewhere in
the knowledge base but were not retrieved. Improving the embedding model or adding more chunks
(higher K) might help.

---

## RAGAS Retriever Metrics

### 4. Context Precision

**Purpose**: Measures whether relevant contexts are ranked at the top. Conceptually similar to
DeepEval's ContextualPrecisionMetric but uses a different computation approach.

**Algorithm**:
1. For each context chunk, an LLM judges whether it is relevant to the `user_input` given
   the `reference` answer
2. Average Precision is computed across positions with relevant items

**Formula**:

```
Context Precision = (1/|relevant items|) * Sum_k(Precision@k * relevance_k)
```

**Required fields**: `user_input`, `retrieved_contexts`, `reference`

**Key difference from DeepEval**: Uses `reference` (ground truth answer) instead of
`expected_output` — semantically the same but the field naming differs. RAGAS computes
this as Average Precision while DeepEval calls it Weighted Cumulative Precision; the
formulas are mathematically equivalent.

```python
from ragas.metrics import ContextPrecision
from ragas.dataset_schema import SingleTurnSample

metric = ContextPrecision()

sample = SingleTurnSample(
    user_input="What year was the Eiffel Tower built?",
    retrieved_contexts=[
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "The Eiffel Tower is a popular tourist destination in Paris.",
        "Gustave Eiffel's company designed and built the tower.",
    ],
    reference="The Eiffel Tower was constructed in 1889."
)

score = await metric.single_turn_ascore(sample)
```

### 5. Context Recall

**Purpose**: Measures what fraction of the reference answer can be attributed to the retrieved
contexts.

**Algorithm**:
1. Decompose the `reference` into individual claims
2. Check each claim against `retrieved_contexts` for attributability
3. Compute the fraction

**Formula**:

```
Context Recall = |attributable reference claims| / |total reference claims|
```

**Required fields**: `retrieved_contexts`, `reference`

**Key difference from DeepEval**: RAGAS extracts claims from `reference` (same as DeepEval's
`expected_output`). The algorithmic approach is very similar between the two frameworks.

```python
from ragas.metrics import ContextRecall
from ragas.dataset_schema import SingleTurnSample

metric = ContextRecall()

sample = SingleTurnSample(
    user_input="What are the benefits of exercise?",
    retrieved_contexts=[
        "Regular exercise improves cardiovascular health.",
        "Physical activity helps maintain healthy body weight."
    ],
    reference="Exercise improves heart health, helps weight management, boosts mood, and strengthens bones."
)

score = await metric.single_turn_ascore(sample)
# Expect ~0.5 (2 of 4 claims are supported)
```

### 6. Context Entity Recall

**Purpose**: Measures entity-level overlap between the reference and retrieved contexts using
Named Entity Recognition (NER). This is a **non-LLM** metric, making it fast and deterministic.

**Algorithm**:
1. Extract named entities from `reference` using NER
2. Extract named entities from `retrieved_contexts` using NER
3. Compute recall of reference entities in context entities

**Formula**:

```
Context Entity Recall = |entities_reference ∩ entities_contexts| / |entities_reference|
```

**Required fields**: `retrieved_contexts`, `reference`

**Advantages**: No LLM needed, deterministic, fast, cheap
**Limitations**: Only captures entity-level overlap; misses semantic relationships and
non-entity factual content

```python
from ragas.metrics import ContextEntityRecall

metric = ContextEntityRecall()

sample = SingleTurnSample(
    user_input="Who were the founders of Google?",
    retrieved_contexts=[
        "Google was founded by Larry Page and Sergey Brin in 1998.",
        "The company started as a research project at Stanford University."
    ],
    reference="Larry Page and Sergey Brin founded Google in 1998 at Stanford University."
)

score = await metric.single_turn_ascore(sample)
# Entities in reference: {Larry Page, Sergey Brin, Google, 1998, Stanford University}
# All present in contexts -> score = 1.0
```

### 7. LLM Context Precision (With and Without Reference)

**With Reference**: Uses the `reference` to judge whether each context chunk is relevant.
Same concept as Context Precision but may use enhanced LLM prompting.

```python
from ragas.metrics import LLMContextPrecisionWithReference

metric = LLMContextPrecisionWithReference()
# Requires: retrieved_contexts, reference
```

**Without Reference**: Judges context relevance without ground truth, using only the user's
query and the generated response.

```python
from ragas.metrics import LLMContextPrecisionWithoutReference

metric = LLMContextPrecisionWithoutReference()
# Requires: user_input, response, retrieved_contexts
# Does NOT require: reference
```

This is useful in production settings where ground truth labels are not available.

### 8. LLM Context Recall

Enhanced version of Context Recall using improved LLM prompting for attribution checking.

```python
from ragas.metrics import LLMContextRecall

metric = LLMContextRecall()
# Requires: retrieved_contexts, reference
```

### 9. Noise Sensitivity

**Purpose**: Measures how much the generator is affected by irrelevant (noisy) context chunks.
This is technically a generator metric, but it evaluates a critical interaction between
retriever quality and generator robustness.

**Algorithm**:
1. Compare the `response` against the `reference` to identify correct and incorrect claims
2. For incorrect claims, check if they can be attributed to irrelevant context chunks
3. A high score means the generator is being misled by noisy retrieval

**Required fields**: `user_input`, `response`, `reference`, `retrieved_contexts`

```python
from ragas.metrics import NoiseSensitivity

metric = NoiseSensitivity()

sample = SingleTurnSample(
    user_input="What is the population of Tokyo?",
    response="Tokyo has a population of about 14 million and is known for its cherry blossoms in spring.",
    reference="The population of Tokyo is approximately 14 million people.",
    retrieved_contexts=[
        "Tokyo is the capital of Japan with a population of approximately 14 million.",
        "Cherry blossom season in Tokyo typically occurs in late March to mid-April.",
        "Tokyo hosted the 2020 Summer Olympics (held in 2021)."
    ]
)

score = await metric.single_turn_ascore(sample)
# The cherry blossom claim in the response came from noisy context (not relevant to population query)
# This would result in a moderate noise sensitivity score
```

---

## Cross-Framework Comparison

### Side-by-Side Metric Mapping

| Concept | DeepEval Metric | RAGAS Metric | Key Difference |
|---------|----------------|--------------|----------------|
| **Context relevance** | ContextualRelevancyMetric | (No direct equivalent) | DeepEval extracts statements from context; RAGAS has no standalone relevancy metric |
| **Context ranking** | ContextualPrecisionMetric | ContextPrecision | Same WCP/AP formula; DeepEval uses expected_output, RAGAS uses reference |
| **Context coverage** | ContextualRecallMetric | ContextRecall | Both extract claims from ground truth and check attributability |
| **Entity overlap** | (No equivalent) | ContextEntityRecall | RAGAS offers this lightweight NER-based metric; DeepEval does not |
| **Noise impact** | (No equivalent) | NoiseSensitivity | RAGAS measures how noise affects the generator; DeepEval does not |
| **Reference-free precision** | ContextualRelevancyMetric | LLMContextPrecisionWithoutReference | Different approaches to reference-free evaluation |

### Calculation Approach Differences

**Context Precision / Contextual Precision**:
- DeepEval: LLM judges each node against `input` + `expected_output` -> WCP formula
- RAGAS: LLM judges each node against `user_input` + `reference` -> Average Precision formula
- The formulas are mathematically equivalent (AP and WCP produce the same result)
- Scores may still differ because the LLM prompts and judgment criteria differ

**Context Recall / Contextual Recall**:
- DeepEval: Extracts statements from `expected_output`, checks attributability to `retrieval_context`
- RAGAS: Extracts statements from `reference`, checks attributability to `retrieved_contexts`
- Conceptually identical, but prompt differences may cause score variations

**Context Relevancy**:
- DeepEval: ContextualRelevancyMetric extracts statements from context, classifies as relevant/irrelevant to input
- RAGAS: No direct single metric for this; closest is LLMContextPrecisionWithoutReference
- DeepEval's version is reference-free and simpler to use when you lack ground truth

### When One Framework's Metric Is Better

| Scenario | Better Choice | Why |
|----------|---------------|-----|
| No ground truth available | DeepEval ContextualRelevancyMetric | Reference-free, only needs input + context |
| Need entity-level analysis | RAGAS ContextEntityRecall | Fast NER-based metric, no LLM cost |
| Evaluating reranker quality | Either (equivalent) | Both use AP/WCP formula |
| Measuring noise robustness | RAGAS NoiseSensitivity | Only RAGAS offers this metric |
| CI/CD pipeline testing | DeepEval (any retriever metric) | pytest-native integration |
| Research/academic use | RAGAS (any retriever metric) | Published methodology |

---

## Practical Guide: Diagnosing Retriever Problems

### Step 1: Identify the Failure Mode

Run all three core retriever metrics and examine the pattern:

```python
# DeepEval approach
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

relevancy = ContextualRelevancyMetric(threshold=0.7)
precision = ContextualPrecisionMetric(threshold=0.7)
recall = ContextualRecallMetric(threshold=0.7)

# Run all three
for metric in [relevancy, precision, recall]:
    metric.measure(test_case)
    print(f"{metric.__name__}: {metric.score:.2f}")
```

### Step 2: Interpret the Pattern

| Relevancy | Precision | Recall | Diagnosis | Fix |
|-----------|-----------|--------|-----------|-----|
| High | High | High | Retriever is working well | Focus on generator |
| High | Low | High | Relevant chunks present but poorly ranked | **Improve reranker** |
| High | High | Low | Retrieved chunks are relevant but miss key info | **Increase top-K** or improve embedding model |
| Low | Low | High | Lots of noise + key info buried | **Improve chunking + add reranker** |
| Low | Low | Low | Fundamental retrieval failure | **Change embedding model**, review chunking strategy |
| Low | High | Low | Few chunks retrieved, mostly relevant but incomplete | **Increase top-K** |
| High | Low | Low | Paradoxical — check data quality | Verify ground truth labels |

### Step 3: Drill Down by Component

Once you know the failure mode, target the right component:

**If embedding model is the problem** (low recall):
- Try a different embedding model (e.g., switch from `text-embedding-ada-002` to `text-embedding-3-large`)
- Try a domain-specific embedding model (e.g., `bge-large` for general, `PubMedBERT` for medical)
- Consider hybrid search (vector + BM25 keyword search)
- Fine-tune embeddings on your domain data

**If chunking is the problem** (low relevancy):
- Experiment with chunk sizes (256, 512, 1024 tokens)
- Try overlapping chunks (50-100 token overlap)
- Use semantic chunking instead of fixed-size
- Consider hierarchical chunking (parent-child chunks)

**If ranking is the problem** (low precision, adequate recall):
- Add a reranker (e.g., Cohere Rerank, cross-encoder models)
- Reduce top-K to eliminate low-relevance chunks
- Implement Maximal Marginal Relevance (MMR) for diversity

**If coverage is the problem** (low recall despite good relevancy):
- Increase top-K
- Add query expansion / query rewriting
- Implement HyDE (Hypothetical Document Embeddings)
- Add metadata filtering to pre-filter the search space

### Interpreting Scores: What Does the Number Mean?

**Score 0.0 - 0.3 (Critical)**:
The retriever is fundamentally broken for this query. The generator will almost certainly
hallucinate or produce irrelevant output. Common causes: wrong embedding model for the domain,
documents not properly indexed, query/document language mismatch.

**Score 0.3 - 0.5 (Poor)**:
The retriever finds some relevant content but not enough. The generator may produce partial
answers with hallucinated details to fill gaps. Common causes: top-K too low, chunking too
coarse, relevant info spread across many documents.

**Score 0.5 - 0.7 (Moderate)**:
Acceptable for some use cases, but there is room for improvement. The generator has some
relevant context but may miss important details. Common causes: reasonable embedding model but
no reranker, good chunking but need for query expansion.

**Score 0.7 - 0.9 (Good)**:
The retriever is doing its job well. Most of the information needed is available to the generator.
Improvements at the margin are possible but not urgent. Focus shifts to generator quality.

**Score 0.9 - 1.0 (Excellent)**:
Near-perfect retrieval. If the generator still produces poor answers, the problem is definitely
in the generator, not the retriever. This is the target for high-stakes applications (medical,
legal, financial).

---

## Worked Example with Sample Data

Let us walk through a complete retriever evaluation using both frameworks on the same data.

### The Scenario

You have built a RAG chatbot for a company's HR policy documents. A user asks about parental leave.

### The Data

```python
query = "What is the parental leave policy?"

ground_truth = (
    "Employees are entitled to 12 weeks of paid parental leave for the birth or adoption "
    "of a child. Leave can be taken consecutively or intermittently within the first year. "
    "Both parents are eligible regardless of gender. Employees must have been employed for "
    "at least 6 months to qualify."
)

retrieved_contexts = [
    # Chunk 1: Highly relevant (contains key policy details)
    "Parental Leave Policy: All employees who have completed 6 months of continuous employment "
    "are eligible for 12 weeks of fully paid parental leave following the birth or adoption of a child.",

    # Chunk 2: Somewhat relevant (related but not specifically about parental leave duration)
    "Our company supports work-life balance through various leave policies including vacation, "
    "sick leave, and parental leave. We believe in creating an inclusive workplace.",

    # Chunk 3: Relevant (additional policy details)
    "Parental leave can be taken on a consecutive or intermittent basis within 12 months of the "
    "qualifying event. Both birth and adoptive parents are eligible regardless of gender.",

    # Chunk 4: Irrelevant (different topic entirely)
    "The company holiday schedule includes New Year's Day, Memorial Day, Independence Day, "
    "Labor Day, Thanksgiving, and Christmas Day. Floating holidays are also available.",

    # Chunk 5: Marginally relevant (mentions leave but different type)
    "Sick leave accrues at a rate of 1 day per month. Unused sick leave can be carried over "
    "to the next year up to a maximum of 30 days.",
]

generated_response = (
    "The parental leave policy provides 12 weeks of paid leave for birth or adoption. "
    "Leave can be taken consecutively or intermittently within the first year. Both parents "
    "are eligible regardless of gender. You need to have worked here for at least 6 months."
)
```

### Evaluation with DeepEval

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

test_case = LLMTestCase(
    input=query,
    actual_output=generated_response,
    expected_output=ground_truth,
    retrieval_context=retrieved_contexts,
)

# Metric 1: Contextual Relevancy (reference-free)
relevancy = ContextualRelevancyMetric(threshold=0.7, model="gpt-4.1")
relevancy.measure(test_case)
print(f"Contextual Relevancy: {relevancy.score:.2f}")
# Expected: ~0.50 (chunks 1, 3 are relevant; 2 is vaguely relevant; 4, 5 are not)

# Metric 2: Contextual Precision (ranking quality)
precision = ContextualPrecisionMetric(threshold=0.7, model="gpt-4.1")
precision.measure(test_case)
print(f"Contextual Precision: {precision.score:.2f}")
# Expected: ~0.72 (relevant chunks at positions 1 and 3, but irrelevant at 2 pushes score down)

# Metric 3: Contextual Recall (coverage)
recall = ContextualRecallMetric(threshold=0.7, model="gpt-4.1")
recall.measure(test_case)
print(f"Contextual Recall: {recall.score:.2f}")
# Expected: ~0.90+ (ground truth claims are well covered by chunks 1 and 3)
```

### Evaluation with RAGAS

```python
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, ContextEntityRecall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

sample = SingleTurnSample(
    user_input=query,
    response=generated_response,
    retrieved_contexts=retrieved_contexts,
    reference=ground_truth,
)

dataset = EvaluationDataset(samples=[sample])

results = evaluate(
    dataset=dataset,
    metrics=[ContextPrecision(), ContextRecall(), ContextEntityRecall()]
)

print(results)
# Expected output:
# {
#     'context_precision': ~0.72,   (same concept as DeepEval's Contextual Precision)
#     'context_recall': ~0.90+,     (same concept as DeepEval's Contextual Recall)
#     'context_entity_recall': ~0.80  (entity-level check, no LLM needed)
# }
```

### Analysis of Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Contextual Relevancy | ~0.50 | Half the retrieved content is noise. The retriever is fetching too many irrelevant chunks. |
| Contextual Precision | ~0.72 | The most relevant chunk IS at position 1 (good), but irrelevant chunks at positions 2, 4, 5 hurt the ranking. |
| Contextual Recall | ~0.90+ | The retriever DID fetch the key information. The ground truth is well-covered. |
| Context Entity Recall | ~0.80 | Most entities (12 weeks, 6 months, parental leave) are present in contexts. |

### Diagnosis

**Pattern**: Low relevancy, moderate precision, high recall.

**Interpretation**: The retriever IS finding the right information (high recall), but it is also
pulling in a lot of noise (low relevancy). The ranking is decent but not great (moderate precision).

**Recommended fixes**:
1. **Reduce top-K from 5 to 3** — this would eliminate the least relevant chunks
2. **Add a reranker** — to push chunks 4 and 5 to the bottom (or eliminate them)
3. **Improve chunking** — chunk 2 is very generic and should probably be split differently

### After Fixing: Expected Improvement

If we reduce to top-3 with a reranker:

```python
improved_contexts = [
    "Parental Leave Policy: All employees who have completed 6 months of continuous employment "
    "are eligible for 12 weeks of fully paid parental leave following the birth or adoption of a child.",
    "Parental leave can be taken on a consecutive or intermittent basis within 12 months of the "
    "qualifying event. Both birth and adoptive parents are eligible regardless of gender.",
    "Our company supports work-life balance through various leave policies including vacation, "
    "sick leave, and parental leave. We believe in creating an inclusive workplace.",
]
```

Expected new scores:
- Contextual Relevancy: ~0.80 (2 of 3 chunks are highly relevant)
- Contextual Precision: ~0.90 (relevant chunks at positions 1 and 2)
- Contextual Recall: ~0.90+ (still covers the ground truth)

The fix improved precision and relevancy without sacrificing recall. This is the optimization
loop that retriever metrics enable.

---

## Common Failure Patterns and How Metrics Expose Them

### Pattern 1: "The Needle in a Haystack"

**Symptom**: High recall, very low precision/relevancy
**What is happening**: The retriever returns the answer somewhere in a mountain of irrelevant context
**Metrics**: Recall > 0.8, Relevancy < 0.3, Precision < 0.3
**Fix**: Better chunking, add reranker, reduce top-K

### Pattern 2: "Close But No Cigar"

**Symptom**: High relevancy, high precision, low recall
**What is happening**: Retrieved contexts are topically related but miss the specific facts
**Metrics**: Relevancy > 0.7, Precision > 0.7, Recall < 0.5
**Fix**: Better embedding model, query expansion, increase top-K

### Pattern 3: "Wrong Neighborhood"

**Symptom**: Low everything
**What is happening**: The retriever is searching in completely wrong areas of the knowledge base
**Metrics**: All below 0.3
**Fix**: Verify document indexing, check embedding model, review query preprocessing

### Pattern 4: "Right Content, Wrong Order"

**Symptom**: High recall, high relevancy, low precision
**What is happening**: Good chunks retrieved but poorly ranked
**Metrics**: Recall > 0.8, Relevancy > 0.7, Precision < 0.5
**Fix**: Add/improve reranker, implement MMR

### Pattern 5: "Missing Documents"

**Symptom**: Low recall, but other metrics are fine on what IS retrieved
**What is happening**: Relevant documents not in the knowledge base at all
**Metrics**: Recall < 0.4, Relevancy > 0.7 (what IS retrieved is relevant)
**Fix**: Expand knowledge base, check document ingestion pipeline

---

## Summary: Retriever Metric Quick Reference

| Metric | Framework | Needs LLM | Needs Reference | Measures | Primary Use |
|--------|-----------|-----------|-----------------|----------|-------------|
| ContextualRelevancyMetric | DeepEval | Yes | No | Content relevance | Diagnose noise in retrieval |
| ContextualPrecisionMetric | DeepEval | Yes | Yes | Ranking quality | Evaluate reranker |
| ContextualRecallMetric | DeepEval | Yes | Yes | Coverage completeness | Evaluate embedding model |
| ContextPrecision | RAGAS | Yes | Yes | Ranking quality | Evaluate reranker |
| ContextRecall | RAGAS | Yes | Yes | Coverage completeness | Evaluate embedding model |
| ContextEntityRecall | RAGAS | No | Yes | Entity overlap | Quick coverage check |
| LLMContextPrecisionWithRef | RAGAS | Yes | Yes | Enhanced ranking | Advanced precision analysis |
| LLMContextPrecisionNoRef | RAGAS | Yes | No | Reference-free ranking | Production monitoring |
| LLMContextRecall | RAGAS | Yes | Yes | Enhanced coverage | Advanced recall analysis |
| NoiseSensitivity | RAGAS | Yes | Yes | Noise impact | Generator robustness to noise |

---

*Previous: [06 - RAGAS Complete Guide](06_ragas_complete_guide.md) | Next: [08 - Generator Metrics Deep Dive](08_generator_metrics_deep_dive.md)*

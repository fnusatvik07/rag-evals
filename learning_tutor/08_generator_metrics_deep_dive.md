# 08 -- Generator Metrics Deep Dive

## Table of Contents

1. [Why Generator Evaluation Matters](#why-generator-evaluation-matters)
2. [Types of Generator Failures](#types-of-generator-failures)
3. [Faithfulness vs Relevancy vs Correctness](#faithfulness-vs-relevancy-vs-correctness)
4. [DeepEval Generator Metrics](#deepeval-generator-metrics)
5. [RAGAS Generator Metrics](#ragas-generator-metrics)
6. [Cross-Framework Comparison](#cross-framework-comparison)
7. [The Hallucination Problem](#the-hallucination-problem)
8. [Practical Guide: Diagnosing Generator Problems](#practical-guide-diagnosing-generator-problems)

---

## Why Generator Evaluation Matters

### Where Value Is Created (or Destroyed)

The generator is the component users actually interact with. It takes the retrieved contexts and
the user's query and produces a natural language response. This is where:

- **Value is created**: A good generator synthesizes retrieved information into a clear, accurate,
  helpful answer that directly addresses the user's question
- **Value is destroyed**: A bad generator introduces hallucinations, omits critical information,
  includes irrelevant tangents, or produces confusing text

Even with perfect retrieval, a poorly configured or poorly prompted generator can produce
answers that are unfaithful, incomplete, or misleading. The generator is the last mile of the
RAG pipeline, and its failures directly reach the user.

### The Cost of Generator Failures

Generator failures have different costs depending on the domain:

| Domain | Cost of Generator Failure |
|--------|--------------------------|
| Customer support | User gets wrong answer, calls again, trust decreases |
| Medical | Incorrect medical information, potential harm |
| Legal | Wrong legal advice, liability risk |
| Financial | Incorrect financial data, bad investment decisions |
| Education | Misinformation learned by students |
| Internal tools | Employee time wasted, wrong decisions made |

In high-stakes domains, even a single hallucinated claim in an otherwise correct answer can
cause serious harm. This is why generator metrics must be comprehensive and precise.

---

## Types of Generator Failures

Understanding the taxonomy of failures helps you choose the right metrics:

### 1. Unfaithful Generation (Hallucination)

The generator produces claims that are NOT supported by the retrieved contexts. The most
dangerous failure mode because the answer sounds authoritative but contains made-up information.

**Example**:
- Context: "The company was founded in 2010 by John Smith."
- Response: "The company was founded in **2008** by John Smith and **Jane Doe**."
- Failures: Wrong year (2008 vs 2010) and fabricated co-founder (Jane Doe)

**Detected by**: Faithfulness metrics (DeepEval, RAGAS)

### 2. Irrelevant Generation

The generator produces text that does not address the user's question, even if the text is
factually correct given the contexts.

**Example**:
- Query: "What is the return policy?"
- Context: "Returns accepted within 30 days. Store hours are 9am-5pm."
- Response: "Our store is open from 9am to 5pm Monday through Friday."
- Failure: Answer is factually correct (from context) but does not address the question

**Detected by**: Answer Relevancy metrics (DeepEval, RAGAS)

### 3. Incorrect Generation

The generator produces claims that contradict the ground truth, regardless of what the context
says. This is an end-to-end failure — the final answer is wrong.

**Example**:
- Ground truth: "The boiling point of water is 100 degrees Celsius."
- Response: "Water boils at 90 degrees Celsius."
- Failure: The answer is factually incorrect

**Detected by**: Answer Correctness, Factual Correctness (RAGAS), G-Eval (DeepEval)

### 4. Incomplete Generation

The generator provides a partial answer, missing important information that was available in
the contexts.

**Example**:
- Context: "Side effects include headache, nausea, and in rare cases, liver damage."
- Response: "Side effects include headache and nausea."
- Failure: Omitted the critical "liver damage" side effect

**Detected by**: Answer Correctness (recall component), G-Eval with completeness criteria

### 5. Incoherent Generation

The generator produces text that is difficult to understand, contradicts itself, or is poorly
structured.

**Example**:
- Response: "The policy allows 30 days returns. However, returns are not accepted. You can
  return items within the policy period which does not exist."
- Failure: Self-contradictory, incoherent

**Detected by**: G-Eval with coherence criteria

---

## Faithfulness vs Relevancy vs Correctness

These three concepts are often confused. Understanding the distinctions is critical for
choosing the right metrics.

### Faithfulness (Grounding)

**Question**: "Is the response grounded in the retrieved contexts?"

- Compares: response vs retrieved_contexts
- Does NOT need: ground truth / reference
- Catches: Hallucination, fabrication, unsupported claims
- Does NOT catch: Correct answers from wrong sources, irrelevance

**Key insight**: A response can be perfectly faithful (every claim is in the context) and still
be completely wrong — if the contexts themselves are wrong or irrelevant.

### Relevancy

**Question**: "Does the response address the user's question?"

- Compares: response vs user_input (query)
- Does NOT need: ground truth / reference (in most implementations)
- Catches: Off-topic responses, tangential information
- Does NOT catch: Wrong answers that happen to be on-topic

**Key insight**: A response can be highly relevant (on-topic) and still be completely wrong.
"The capital of France is London" is relevant to "What is the capital of France?" but incorrect.

### Correctness

**Question**: "Is the response factually correct compared to ground truth?"

- Compares: response vs reference (ground truth)
- REQUIRES: ground truth / reference
- Catches: Any factual error, regardless of source
- Does NOT evaluate: Whether the answer came from the context (that is faithfulness)

**Key insight**: Correctness is the ultimate measure but requires ground truth labels, which
are expensive to create. Faithfulness and relevancy are proxies that work without ground truth.

### The Relationship Between the Three

```
                                    CORRECT
                                   /       \
                                  /         \
                        FAITHFUL &        RELEVANT &
                        RELEVANT          CORRECT
                        (ideal)           (lucky guess)
                       /                           \
                      /                             \
              FAITHFUL                          RELEVANT
              only                              only
              (on-topic, grounded,              (on-topic but
               but wrong context)                wrong answer)
                      \                             /
                       \                           /
                        UNFAITHFUL &        IRRELEVANT &
                        IRRELEVANT          INCORRECT
                        (worst case)        (off-topic nonsense)
```

The **ideal** response is all three: faithful to context, relevant to the query, and correct
compared to ground truth. Metrics help you identify which dimension is failing.

---

## DeepEval Generator Metrics

### 1. AnswerRelevancyMetric

**Purpose**: Measures whether the LLM's response is relevant to the user's query. Detects
off-topic responses, unnecessary tangents, and irrelevant padding.

**Type**: LLM-as-judge, single-turn, **referenceless** (no expected_output needed)

**What it targets in your pipeline**: Prompt template quality, system prompt instructions

**Algorithm (step-by-step)**:
1. An LLM extracts all individual **statements** from the `actual_output`
2. For each statement, the LLM classifies it as **relevant** or **irrelevant** to the `input`
3. The score is the ratio of relevant statements to total statements

**Formula**:

```
Answer Relevancy = Number of Relevant Statements / Total Number of Statements
```

**Required test case fields**:
- `input` (user query)
- `actual_output` (LLM response)

**Does NOT require**: `expected_output`, `retrieval_context`, `context`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing score |
| `model` | str/DeepEvalBaseLLM | gpt-4.1 | Judge LLM |
| `include_reason` | bool | True | Output explanation |
| `strict_mode` | bool | False | Binary scoring (1 or 0) |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | AnswerRelevancyTemplate | default | Custom prompts |

**Code example**:

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)

test_case = LLMTestCase(
    input="What is the company's vacation policy?",
    actual_output=(
        "The company offers 15 days of paid vacation per year for full-time employees. "
        "Vacation days accrue monthly at 1.25 days per month. "
        "By the way, our office cafeteria serves excellent Italian food. "
        "Unused vacation days can be carried over up to a maximum of 5 days."
    )
)

metric.measure(test_case)
print(f"Score: {metric.score}")    # e.g., 0.75 (3 of 4 statements relevant)
print(f"Reason: {metric.reason}")
```

**Step-by-step calculation**:

Statements extracted from actual_output:
1. "Company offers 15 days paid vacation for full-time employees" -> RELEVANT
2. "Vacation days accrue monthly at 1.25 days per month" -> RELEVANT
3. "Office cafeteria serves excellent Italian food" -> IRRELEVANT (not about vacation policy)
4. "Unused vacation days can be carried over up to 5 days" -> RELEVANT

Score = 3 relevant / 4 total = 0.75

**Interpretation**: The response is mostly relevant but contains one irrelevant tangent about
the cafeteria. Score of 0.75 suggests the prompt template should be refined to keep the
generator focused on the question.

---

### 2. FaithfulnessMetric

**Purpose**: Measures whether the claims in the response are factually consistent with the
retrieved context. This is the primary metric for detecting RAG hallucination.

**Type**: LLM-as-judge, single-turn, **referenceless** (no expected_output needed, but needs
retrieval_context)

**What it targets in your pipeline**: LLM model choice, temperature setting, prompt template

**Algorithm (step-by-step)**:
1. An LLM extracts all **claims/assertions** from the `actual_output`
2. For each claim, the LLM checks whether it **contradicts** any facts in the `retrieval_context`
3. A claim is **truthful** if it does NOT contradict the retrieval context
4. The score is the ratio of truthful claims to total claims

**Formula**:

```
Faithfulness = Number of Truthful Claims / Total Number of Claims
```

**Important nuance**: A claim that is not mentioned in the context but also does not contradict
it is treated as truthful by default. This means the metric primarily catches **contradictions**
with the context, not merely **unsupported** claims. If you want stricter grounding (penalizing
any claim not explicitly supported), use the `penalize_ambiguous_claims=True` parameter.

**Required test case fields**:
- `input` (user query)
- `actual_output` (LLM response)
- `retrieval_context` (retrieved chunks)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing score |
| `model` | str/DeepEvalBaseLLM | gpt-4.1 | Judge LLM |
| `include_reason` | bool | True | Output explanation |
| `strict_mode` | bool | False | Binary scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `truths_extraction_limit` | int | None | Cap on number of facts extracted from context |
| `penalize_ambiguous_claims` | bool | False | If True, ambiguous/unverifiable claims count as unfaithful |
| `evaluation_template` | FaithfulnessTemplate | default | Custom prompts |

**Code example**:

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True,
    truths_extraction_limit=None,       # No limit on truth extraction
    penalize_ambiguous_claims=False      # Default: ambiguous claims are not penalized
)

test_case = LLMTestCase(
    input="What are the side effects of aspirin?",
    actual_output=(
        "Common side effects of aspirin include stomach irritation, nausea, and increased "
        "bleeding risk. In rare cases, aspirin can cause liver damage. Aspirin was invented "
        "by Felix Hoffmann in 1897."
    ),
    retrieval_context=[
        "Aspirin (acetylsalicylic acid) common side effects include gastrointestinal irritation, "
        "nausea, and increased risk of bleeding.",
        "Rare side effects of aspirin include tinnitus, allergic reactions, and Reye's syndrome in children.",
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
```

**Step-by-step calculation**:

Claims extracted from actual_output:
1. "Aspirin causes stomach irritation" -> Supported by context 1. TRUTHFUL.
2. "Aspirin causes nausea" -> Supported by context 1. TRUTHFUL.
3. "Aspirin causes increased bleeding risk" -> Supported by context 1. TRUTHFUL.
4. "Aspirin can cause liver damage in rare cases" -> NOT in context (context says tinnitus, allergic reactions, Reye's). CONTRADICTS (liver damage not mentioned as a side effect).
5. "Aspirin was invented by Felix Hoffmann in 1897" -> Not in context, but does not contradict either.

With `penalize_ambiguous_claims=False` (default):
- Claims 1-3: Truthful
- Claim 4: Not truthful (contradicts context which lists different rare side effects)
- Claim 5: Ambiguous (not in context, not contradicted) -> Treated as truthful

Score = 4/5 = 0.80

With `penalize_ambiguous_claims=True`:
- Claim 5 would be treated as NOT truthful (not verifiable from context)
- Score = 3/5 = 0.60

**The `truths_extraction_limit` parameter**: When your retrieval context is very large (many chunks
with many facts), the LLM may struggle to process everything or the evaluation may be slow. Setting
`truths_extraction_limit=20` limits the LLM to extracting only the 20 most important facts from
the context, ranked by the evaluation model. This trades completeness for speed and focus.

---

### 3. HallucinationMetric

**Purpose**: Measures whether the LLM's output contradicts provided ground-truth context.
**This is DIFFERENT from FaithfulnessMetric** in an important way.

**Critical distinction**:
- **FaithfulnessMetric**: Uses `retrieval_context` (what the RAG retriever returned)
- **HallucinationMetric**: Uses `context` (ground truth knowledge, not retrieval output)

**When to use which**:
- Use **FaithfulnessMetric** when evaluating a **RAG pipeline** — you want to know if the
  generator is faithful to what was retrieved
- Use **HallucinationMetric** when you have **curated ground truth** and want to know if the
  output contradicts known facts, regardless of what was retrieved

**Algorithm**:
1. An LLM examines each context string individually
2. For each context, it determines whether the `actual_output` **contradicts** that context
3. The score is the fraction of contradicted contexts

**Formula**:

```
Hallucination = Number of Contradicted Contexts / Total Number of Contexts
```

**IMPORTANT**: Unlike most metrics, **lower is better** for HallucinationMetric. A score of 0
means no hallucination. A score of 1.0 means the output contradicts every context.

**Required test case fields**:
- `input` (user query)
- `actual_output` (LLM response)
- `context` (ground truth contexts — NOT retrieval_context)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Maximum passing score (lower = less hallucination tolerated) |
| `model` | str/DeepEvalBaseLLM | gpt-4.1 | Judge LLM |
| `include_reason` | bool | True | Output explanation |
| `strict_mode` | bool | False | Only score 0 passes |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example**:

```python
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

metric = HallucinationMetric(threshold=0.5)

test_case = LLMTestCase(
    input="What was the man doing?",
    actual_output="A blond man drinking water in a park.",
    context=[
        "A man with blond hair and a brown shirt drinking from a public water fountain.",
        "The fountain is located in the downtown area near the library."
    ]
)

metric.measure(test_case)
print(f"Score: {metric.score}")    # Lower is better!
print(f"Reason: {metric.reason}")
# Expected: Low score (~0.0) because the output does not contradict either context
```

**Scoring inversion**: The threshold for HallucinationMetric works in reverse. The test
**passes** when `score <= threshold`. So `threshold=0.5` means up to 50% of contexts can
be contradicted before the test fails.

---

### 4. G-Eval for Generation Quality

**Purpose**: The most versatile DeepEval metric — allows you to define **custom evaluation
criteria** using natural language. Uses the G-Eval framework (from the paper "NLG Evaluation
using GPT-4 with Better Human Alignment").

**How G-Eval works**:
1. You define `criteria` (what to evaluate) and `evaluation_steps` (how to evaluate)
2. If only `criteria` is provided, the LLM auto-generates evaluation steps using Chain-of-Thought
3. The LLM scores the test case on a 1-5 scale
4. Output token probabilities are used to normalize the score (weighted summation), reducing
   LLM scoring bias
5. The final score is mapped to 0-1 range

**Key parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Name of the custom metric |
| `criteria` | str | Yes* | Natural language description of what to evaluate |
| `evaluation_params` | list | Yes | Which LLMTestCase fields to use |
| `evaluation_steps` | list[str] | No* | Explicit steps for the LLM to follow |
| `rubric` | list[Rubric] | No | Score range definitions |
| `threshold` | float | No | Passing threshold (default 0.5) |
| `model` | str | No | Judge model (default gpt-4.1) |
| `strict_mode` | bool | No | Binary scoring |

*Either `criteria` or `evaluation_steps` must be provided. If both given, `evaluation_steps` takes precedence.

**Available evaluation_params**:

```python
from deepeval.test_case import LLMTestCaseParams

LLMTestCaseParams.INPUT              # user_input
LLMTestCaseParams.ACTUAL_OUTPUT      # actual_output
LLMTestCaseParams.EXPECTED_OUTPUT    # expected_output
LLMTestCaseParams.CONTEXT            # context
LLMTestCaseParams.RETRIEVAL_CONTEXT  # retrieval_context
```

**Code examples for common generation quality criteria**:

#### Coherence

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

coherence_metric = GEval(
    name="Coherence",
    evaluation_steps=[
        "Check if the response flows logically from one sentence to the next.",
        "Evaluate whether the response maintains a consistent topic and avoids abrupt transitions.",
        "Assess whether ideas are connected with appropriate transitions and references.",
        "Check for any contradictions within the response itself.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)

test_case = LLMTestCase(
    input="Explain how vaccines work.",
    actual_output=(
        "Vaccines work by introducing a weakened or inactive form of a pathogen to the body. "
        "The immune system then learns to recognize and fight the pathogen. "
        "This creates memory cells that provide long-term protection. "
        "If exposed to the real pathogen later, the immune system can respond quickly."
    )
)

coherence_metric.measure(test_case)
print(f"Coherence: {coherence_metric.score}")  # Expected: high score (well-structured response)
```

#### Correctness (Reference-based)

```python
correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'.",
        "Heavily penalize omission of important details present in 'expected output'.",
        "Vague language or differing opinions are acceptable and should not be penalized.",
        "Focus on factual accuracy rather than stylistic differences.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)

test_case = LLMTestCase(
    input="What is the boiling point of water?",
    actual_output="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    expected_output="The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level."
)

correctness_metric.measure(test_case)
```

#### Completeness

```python
completeness_metric = GEval(
    name="Completeness",
    evaluation_steps=[
        "List all key points present in the 'expected output'.",
        "For each key point, check whether it is adequately covered in 'actual output'.",
        "Penalize missing key points proportionally to their importance.",
        "Do not penalize additional information that is accurate and relevant.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)
```

#### Fluency

```python
fluency_metric = GEval(
    name="Fluency",
    evaluation_steps=[
        "Evaluate whether the response uses natural, flowing language.",
        "Check for grammatical errors, awkward phrasing, or unnatural constructions.",
        "Assess whether the vocabulary level is appropriate for the context.",
        "A response should read as if written by a competent human writer.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)
```

#### RAG-Specific: Faithfulness with Domain Knowledge

```python
medical_faithfulness = GEval(
    name="Medical Faithfulness",
    evaluation_steps=[
        "Extract all medical claims and diagnoses from the actual output.",
        "Verify each medical claim against the retrieved contextual information.",
        "Identify any contradictions or unsupported medical claims that could lead to misdiagnosis.",
        "Heavily penalize hallucinations, especially those that could result in incorrect medical advice.",
        "Provide reasons for the faithfulness score, emphasizing clinical accuracy and patient safety.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    threshold=0.9  # High threshold for medical domain
)
```

#### Using Rubrics for Structured Scoring

```python
from deepeval.metrics.g_eval import Rubric

graded_correctness = GEval(
    name="Graded Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect or mostly wrong."),
        Rubric(score_range=(3, 4), expected_outcome="Partially correct with significant errors."),
        Rubric(score_range=(5, 6), expected_outcome="Mostly correct with minor errors."),
        Rubric(score_range=(7, 8), expected_outcome="Correct with minor omissions."),
        Rubric(score_range=(9, 10), expected_outcome="Completely correct and comprehensive."),
    ]
)
```

---

## RAGAS Generator Metrics

### 5. Faithfulness (RAGAS)

**Purpose**: Measures whether claims in the response are supported by the retrieved contexts.
The foundational RAGAS metric for hallucination detection.

**Algorithm (3-step process)**:
1. **Claim Extraction**: An LLM decomposes the `response` into atomic claims/statements
2. **NLI Verification**: For each claim, the LLM performs Natural Language Inference against
   the `retrieved_contexts` to determine if the claim is supported, contradicted, or neutral
3. **Score Computation**: The fraction of supported claims

**Formula**:

```
Faithfulness = |supported claims| / |total claims|
```

**Required fields**: `user_input`, `response`, `retrieved_contexts`

**Does NOT require**: `reference` (reference-free metric)

```python
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample

metric = Faithfulness()

sample = SingleTurnSample(
    user_input="What are the benefits of meditation?",
    response=(
        "Meditation reduces stress and anxiety by activating the parasympathetic nervous system. "
        "It also improves focus and concentration. Studies show it can lower blood pressure "
        "by up to 20%. Regular practice enhances emotional well-being."
    ),
    retrieved_contexts=[
        "Meditation has been shown to reduce stress and anxiety. Research indicates it activates "
        "the parasympathetic nervous system, promoting relaxation.",
        "Regular meditation practice improves focus, concentration, and emotional regulation. "
        "Some studies suggest modest blood pressure reductions of 5-10%.",
    ]
)

score = await metric.single_turn_ascore(sample)
# Claims analysis:
# 1. "Reduces stress and anxiety" -> Supported by context 1. FAITHFUL.
# 2. "Activates parasympathetic nervous system" -> Supported by context 1. FAITHFUL.
# 3. "Improves focus and concentration" -> Supported by context 2. FAITHFUL.
# 4. "Lowers blood pressure by up to 20%" -> CONTRADICTED (context says 5-10%). NOT FAITHFUL.
# 5. "Enhances emotional well-being" -> Supported by context 2 (emotional regulation). FAITHFUL.
# Score = 4/5 = 0.8
```

### 6. Answer Relevancy (RAGAS)

**Purpose**: Measures whether the response is relevant to the user's question. Uses a unique
reverse question generation approach.

**Algorithm (question generation + embedding similarity)**:
1. An LLM generates N hypothetical questions that the `response` would answer
2. Each generated question is encoded using an embedding model
3. The original `user_input` is also encoded
4. Cosine similarity is computed between each generated question and the original input
5. The score is the mean cosine similarity

**Formula**:

```
Answer Relevancy = (1/N) * Sum_{i=1}^{N} cosine_similarity(embed(q_i), embed(user_input))
```

Where `q_i` are questions generated from the response.

**Required fields**: `user_input`, `response`, `retrieved_contexts`

**Needs**: Both LLM (for question generation) and Embeddings (for similarity computation)

```python
from ragas.metrics import AnswerRelevancy
from ragas.dataset_schema import SingleTurnSample

metric = AnswerRelevancy()

sample = SingleTurnSample(
    user_input="What is the capital of Japan?",
    response=(
        "The capital of Japan is Tokyo. Tokyo is located on the eastern coast of Honshu island. "
        "It has a population of approximately 14 million people. "
        "Japanese cuisine is known for sushi, ramen, and tempura."
    ),
    retrieved_contexts=[
        "Tokyo is the capital and most populous city of Japan, located on the eastern coast of Honshu.",
    ]
)

score = await metric.single_turn_ascore(sample)
# Generated questions from response might include:
# q1: "What is the capital of Japan?" -> Very similar to original input
# q2: "Where is Tokyo located?" -> Somewhat similar
# q3: "What is the population of Tokyo?" -> Less similar
# q4: "What is Japanese cuisine known for?" -> Low similarity to original input
# The cuisine information pulls the score down because it generates questions
# that are dissimilar to the original query.
```

**Why this approach is clever**: Instead of directly asking "is this relevant?", the method
asks "what questions would this answer?" If the answer would primarily address questions
similar to the original query, it is relevant. If it would address very different questions,
it contains off-topic content. This indirect measurement tends to be more robust than direct
relevance classification.

### 7. Answer Correctness (RAGAS)

**Purpose**: A hybrid metric combining factual correctness with semantic similarity to provide
a comprehensive answer quality score.

**Algorithm**:
1. Compute **Factual Correctness** (LLM-based claim comparison between response and reference)
2. Compute **Semantic Similarity** (embedding cosine similarity between response and reference)
3. Combine with weighted average

**Formula**:

```
Answer Correctness = w1 * FactualCorrectness + w2 * SemanticSimilarity
```

Default weights: w1 = 0.75, w2 = 0.25

**Required fields**: `response`, `reference`

**Needs**: Both LLM and Embeddings

```python
from ragas.metrics import AnswerCorrectness
from ragas.dataset_schema import SingleTurnSample

metric = AnswerCorrectness()
# Customize weights:
metric = AnswerCorrectness(weights=[0.6, 0.4])

sample = SingleTurnSample(
    user_input="When was the first iPhone released?",
    response="The first iPhone was released on June 29, 2007 by Apple Inc.",
    reference="Apple released the original iPhone on June 29, 2007."
)

score = await metric.single_turn_ascore(sample)
# Factual correctness: High (both agree on June 29, 2007)
# Semantic similarity: High (both convey the same meaning)
# Combined score: ~0.95+
```

### 8. Factual Correctness (RAGAS)

**Purpose**: Compares claims in the response against claims in the reference to compute
a precision/recall/F1 style score at the claim level.

**Algorithm**:
1. Extract claims from `response`
2. Extract claims from `reference`
3. For each response claim, check if it matches a reference claim (True Positive) or not
   (False Positive)
4. For each reference claim, check if it is missing from the response (False Negative)
5. Compute F1, precision, or recall

**Formula**:

```
Precision = TP / (TP + FP)          # How many response claims are correct?
Recall = TP / (TP + FN)             # How many reference claims are covered?
F1 = 2 * Precision * Recall / (Precision + Recall)   # Balanced measure
```

**Modes**:
- `mode="f1"` (default): Balances precision and recall
- `mode="precision"`: Only penalizes wrong claims (ignores missing information)
- `mode="recall"`: Only penalizes missing information (ignores extra claims)

```python
from ragas.metrics import FactualCorrectness
from ragas.dataset_schema import SingleTurnSample

# F1 mode (default)
metric = FactualCorrectness(mode="f1")

# Precision mode (useful when you care more about NOT saying wrong things)
metric_precision = FactualCorrectness(mode="precision")

# Recall mode (useful when you care more about completeness)
metric_recall = FactualCorrectness(mode="recall")

sample = SingleTurnSample(
    user_input="List the first three US presidents.",
    response="The first three US presidents were George Washington, John Adams, and Thomas Jefferson. Washington served from 1789 to 1797.",
    reference="The first three US presidents were George Washington (1789-1797), John Adams (1797-1801), and Thomas Jefferson (1801-1809)."
)

score = await metric.single_turn_ascore(sample)
# Response claims: Washington, Adams, Jefferson, Washington served 1789-1797 (4 claims)
# Reference claims: Washington, Adams, Jefferson, Washington 1789-1797, Adams 1797-1801, Jefferson 1801-1809 (6 claims)
# TP = 4 (all response claims match reference)
# FP = 0 (no wrong claims in response)
# FN = 2 (Adams and Jefferson dates missing from response)
# Precision = 4/4 = 1.0
# Recall = 4/6 = 0.67
# F1 = 2 * 1.0 * 0.67 / (1.0 + 0.67) = 0.80
```

### 9. Semantic Similarity (RAGAS)

**Purpose**: Pure embedding-based cosine similarity between response and reference. Fast,
cheap, deterministic (given the same embedding model).

**Algorithm**:
1. Encode `response` with embedding model
2. Encode `reference` with embedding model
3. Compute cosine similarity

**Formula**:

```
Semantic Similarity = cosine_similarity(embed(response), embed(reference))
```

**Required fields**: `response`, `reference`

**Needs**: Embeddings only (no LLM)

```python
from ragas.metrics import SemanticSimilarity
from ragas.dataset_schema import SingleTurnSample

metric = SemanticSimilarity()

sample = SingleTurnSample(
    user_input="What is photosynthesis?",
    response="Photosynthesis is how plants convert sunlight into food energy.",
    reference="Photosynthesis is the process by which plants use light energy to synthesize glucose from CO2 and water."
)

score = await metric.single_turn_ascore(sample)
# Both describe photosynthesis in similar terms -> high similarity score
```

**Limitations**:
- Cannot distinguish factual errors that use similar vocabulary
- "Water boils at 50 degrees Celsius" and "Water boils at 100 degrees Celsius" would have
  very high semantic similarity despite being factually different
- Best used as a complement to, not replacement for, claim-level metrics

---

## Cross-Framework Comparison

### How DeepEval Faithfulness Differs from RAGAS Faithfulness

| Aspect | DeepEval FaithfulnessMetric | RAGAS Faithfulness |
|--------|---------------------------|-------------------|
| **Step 1** | Extract claims from actual_output | Extract claims from response |
| **Step 2** | Classify each claim as truthful/not based on retrieval_context | NLI verification against retrieved_contexts |
| **Truthful definition** | Does not CONTRADICT context | Is SUPPORTED by context |
| **Ambiguous claims** | Configurable via `penalize_ambiguous_claims` | Treated as not supported |
| **Key parameter** | `truths_extraction_limit` limits facts from context | No equivalent parameter |
| **Default behavior** | Lenient (ambiguous claims pass) | Strict (must be supported) |

**Practical impact**: DeepEval's default Faithfulness tends to produce **higher scores** than
RAGAS Faithfulness on the same data because DeepEval only penalizes contradictions by default,
while RAGAS requires explicit support. Set `penalize_ambiguous_claims=True` in DeepEval for
behavior closer to RAGAS.

### How DeepEval Answer Relevancy Differs from RAGAS Answer Relevancy

| Aspect | DeepEval AnswerRelevancyMetric | RAGAS AnswerRelevancy |
|--------|-------------------------------|---------------------|
| **Algorithm** | Extract statements, classify as relevant/irrelevant | Generate questions from answer, compare via embeddings |
| **Uses embeddings** | No (pure LLM classification) | Yes (cosine similarity) |
| **Mechanism** | Direct classification | Indirect (reverse question generation) |
| **Score interpretation** | Fraction of relevant statements | Mean cosine similarity of generated questions |
| **Sensitivity to tangents** | High (directly classifies each statement) | Moderate (depends on question generation quality) |
| **Consistency** | Depends on LLM classification stability | More stable (embeddings are deterministic) |

**Practical impact**: DeepEval's approach is more straightforward and interpretable (you can see
exactly which statements were marked irrelevant). RAGAS's approach is more indirect but can be
more robust because it relies partly on deterministic embeddings rather than entirely on LLM
judgment.

### Which Is More Robust?

For **Faithfulness**: RAGAS tends to be stricter and catches more subtle hallucinations because
it requires explicit support rather than just checking for contradictions. However, this also
means it may flag technically correct responses as unfaithful if the supporting evidence is
not explicit in the context. DeepEval with `penalize_ambiguous_claims=True` approaches similar
strictness.

For **Answer Relevancy**: DeepEval's direct classification approach is more interpretable and
easier to debug. RAGAS's embedding-based approach is more consistent across runs. For production
use, DeepEval's approach may be preferred for debuggability; for research, RAGAS's approach
provides a more nuanced similarity score.

**Recommendation**: Use both frameworks and compare scores. Significant disagreements between
the two often indicate edge cases worth investigating manually.

---

## The Hallucination Problem

### Types of Hallucination in RAG

Hallucination in RAG systems is more nuanced than in vanilla LLMs because there are more
potential sources of error:

#### 1. Intrinsic Hallucination

**Definition**: The response contradicts information that IS present in the retrieved contexts.

**Example**:
- Context: "The building was constructed in 1985."
- Response: "The building was constructed in 1995."
- The date is directly contradicted.

**Detection**: Both Faithfulness metrics catch this. It is the easiest type to detect because
there is a direct contradiction.

**Common causes**: LLM "overriding" context with parametric knowledge, attention failures on
specific details (especially numbers and dates), context window overflow.

#### 2. Extrinsic Hallucination

**Definition**: The response includes information that is NOT present in the retrieved contexts
and cannot be verified from them.

**Example**:
- Context: "The CEO of the company is John Smith."
- Response: "The CEO of the company is John Smith, who graduated from Harvard."
- "Graduated from Harvard" is not in the context — it may or may not be true.

**Detection**: RAGAS Faithfulness catches this (requires explicit support). DeepEval
Faithfulness catches this only with `penalize_ambiguous_claims=True`. Default DeepEval
Faithfulness DOES NOT catch this because it only looks for contradictions.

**Common causes**: LLM drawing on training data (parametric knowledge) to fill gaps in context,
especially for well-known entities.

#### 3. Fabrication

**Definition**: The response includes entirely made-up information with no basis in context
or reality.

**Example**:
- Context: "The company offers two products: ProductA and ProductB."
- Response: "The company's flagship product, ProductC, has won multiple awards."
- ProductC does not exist anywhere.

**Detection**: All Faithfulness metrics catch this because the claim clearly contradicts or
is unsupported by the context.

**Common causes**: High temperature, insufficient context (forcing the LLM to guess), poor
prompt engineering.

### How Each Metric Catches (or Misses) Each Type

| Hallucination Type | DeepEval Faithfulness (default) | DeepEval Faithfulness (strict) | RAGAS Faithfulness | DeepEval Hallucination | RAGAS Answer Correctness |
|-------------------|-------------------------------|-------------------------------|-------------------|----------------------|------------------------|
| **Intrinsic** | Catches (contradiction) | Catches | Catches | Catches | Catches (vs reference) |
| **Extrinsic** | MISSES (no contradiction) | Catches (ambiguous = unfaithful) | Catches (requires support) | Catches (vs ground truth) | Catches (vs reference) |
| **Fabrication** | Catches (obvious contradiction) | Catches | Catches | Catches | Catches |

### Combining Metrics for Comprehensive Hallucination Detection

No single metric catches everything. For comprehensive hallucination detection:

```python
# Approach 1: Use strict DeepEval Faithfulness
from deepeval.metrics import FaithfulnessMetric

strict_faithfulness = FaithfulnessMetric(
    threshold=0.8,
    penalize_ambiguous_claims=True,  # Catch extrinsic hallucinations
    model="gpt-4.1"
)

# Approach 2: Combine RAGAS Faithfulness + Answer Correctness
from ragas.metrics import Faithfulness, AnswerCorrectness

# Faithfulness catches unfaithful-to-context errors
# Answer Correctness catches factual errors vs ground truth
# Together they cover all three hallucination types

# Approach 3: Multi-metric pipeline
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric

# FaithfulnessMetric for RAG-specific grounding (vs retrieval_context)
# HallucinationMetric for ground-truth verification (vs context)
# Run both for defense in depth
```

### Real-World Hallucination Examples

#### Example 1: The Number Swap

```python
# Context says "revenue grew by 15%", LLM says "revenue grew by 51%"
# This is an intrinsic hallucination (transposed digits)

test_case = LLMTestCase(
    input="What was the revenue growth?",
    actual_output="The company's revenue grew by 51% year-over-year.",
    retrieval_context=["Annual revenue grew by 15% compared to the previous year."]
)

# DeepEval Faithfulness: Score ~0.0 (direct contradiction)
# RAGAS Faithfulness: Score ~0.0 (claim not supported)
```

#### Example 2: The Plausible Addition

```python
# Context says "founded in San Francisco", LLM adds "in a garage" (common startup trope)
# This is extrinsic hallucination (plausible but not in context)

test_case = LLMTestCase(
    input="Where was the company founded?",
    actual_output="The company was founded in San Francisco, in a small garage in the Mission District.",
    retrieval_context=["The company was founded in San Francisco in 2015."]
)

# DeepEval Faithfulness (default): Score ~0.67 (garage claim not contradicted, passes!)
# DeepEval Faithfulness (strict): Score ~0.33 (garage claim is ambiguous, fails)
# RAGAS Faithfulness: Score ~0.33 (garage claim not supported, fails)
```

#### Example 3: The Confident Fabrication

```python
# Context about Product A, LLM generates details about non-existent Product B
# This is fabrication

test_case = LLMTestCase(
    input="Tell me about your products.",
    actual_output="Our premium product line includes Product A with AI-powered features and Product B with quantum computing integration.",
    retrieval_context=["Product A uses advanced AI algorithms for data analysis. No other products are currently offered."]
)

# All faithfulness metrics: Low score (~0.5)
# "Product B with quantum computing" clearly contradicts "No other products offered"
```

---

## Practical Guide: Diagnosing Generator Problems

### The Prompt Template -> Metric Score Feedback Loop

Generator quality is primarily controlled by:
1. **The prompt template** (system prompt, few-shot examples, output format instructions)
2. **The LLM model** (GPT-4 vs GPT-3.5 vs open-source)
3. **The temperature** (higher = more creative but less faithful)
4. **The context window management** (how retrieved contexts are presented to the LLM)

Metrics provide the feedback signal to optimize these:

```
[Prompt Template] --> [Generator] --> [Response] --> [Metrics] --> [Diagnosis] --> [Fix Prompt]
      ^                                                                              |
      |______________________________________________________________________________|
```

### Step 1: Run Core Generator Metrics

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)

relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1")
faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4.1",
    penalize_ambiguous_claims=True
)

for metric in [relevancy, faithfulness]:
    metric.measure(test_case)
    print(f"{metric.__name__}: {metric.score:.2f} - {metric.reason}")
```

### Step 2: Interpret the Pattern

| Faithfulness | Relevancy | Diagnosis | Fix |
|-------------|-----------|-----------|-----|
| High | High | Generator working well | Focus on retriever or fine-tuning |
| High | Low | On-topic info in context, but generator goes off-topic | **Improve prompt template** to focus on the query |
| Low | High | Generator addresses the question but hallucinates details | **Lower temperature**, add "only use provided context" instruction |
| Low | Low | Generator is both unfaithful and off-topic | **Overhaul prompt template**, consider better LLM model |

### Step 3: Targeted Fixes

**If faithfulness is low (hallucination problem)**:

1. **Lower temperature**: `temperature=0` or `temperature=0.1`
   ```python
   # Before: temperature=0.7 (too creative)
   llm = ChatOpenAI(model="gpt-4o", temperature=0)
   ```

2. **Add grounding instructions to prompt**:
   ```python
   system_prompt = """You are a helpful assistant. Answer questions based ONLY on the 
   provided context. If the context does not contain enough information to answer the 
   question, say "I don't have enough information to answer that."
   
   DO NOT use your general knowledge. ONLY use the provided context."""
   ```

3. **Add explicit citation requirements**:
   ```python
   system_prompt += """
   For each claim you make, cite the specific context passage that supports it.
   Format: [Claim] (Source: [relevant context excerpt])"""
   ```

4. **Use a more capable model**: GPT-4 class models hallucinate less than GPT-3.5 class models
   in most benchmarks.

**If relevancy is low (off-topic problem)**:

1. **Sharpen the prompt template**:
   ```python
   system_prompt = """Answer the user's question directly and concisely.
   Do not include information that was not asked for.
   Focus your response specifically on what the user is asking."""
   ```

2. **Add output format constraints**:
   ```python
   system_prompt += """
   Structure your response as:
   1. Direct answer to the question (1-2 sentences)
   2. Supporting details from the context (if relevant)
   Do not include tangential information."""
   ```

3. **Filter context before sending to LLM**: Remove retrieved chunks that are not relevant
   (use a retriever relevancy check as a pre-filter).

### Temperature and Its Effect on Faithfulness

Temperature controls the randomness of the LLM's token selection:

| Temperature | Effect on Generation | Effect on Faithfulness |
|-------------|---------------------|----------------------|
| 0.0 | Deterministic, repetitive | Highest faithfulness (sticks to context) |
| 0.1-0.3 | Slight variation, mostly grounded | High faithfulness |
| 0.4-0.6 | Balanced creativity and accuracy | Moderate faithfulness |
| 0.7-0.9 | Creative, may embellish | Lower faithfulness, more extrinsic hallucination |
| 1.0+ | Very creative, unreliable | Low faithfulness, frequent fabrication |

**Recommendation for RAG**: Use temperature 0.0-0.3. The context provides the creativity
(diverse information from different sources). The generator should synthesize, not create.

### Context Window Stuffing and Its Effects

When too many chunks are passed to the generator, several problems arise:

1. **Lost in the middle**: LLMs attend more to the beginning and end of the context window.
   Information in the middle is often ignored. This means that if the most relevant chunk is
   in the middle, the generator may not use it.

2. **Noise amplification**: More chunks means more irrelevant content. The generator may latch
   onto irrelevant details, reducing both faithfulness and relevancy.

3. **Conflicting information**: Multiple chunks may contain slightly different versions of the
   same fact. The generator may combine them in incorrect ways.

**How metrics expose this**:
- Faithfulness drops as irrelevant context is added (noise affects claims)
- Answer Relevancy drops as tangential information gets included
- Use RAGAS NoiseSensitivity to directly measure this effect

**Fix**: Retrieve fewer, higher-quality chunks. Use a reranker to ensure the most relevant
chunks are at the top. Consider using a summarization step before passing context to the generator.

### Interpreting Generator Metric Scores

**Faithfulness scores**:

| Score | Meaning | Risk Level |
|-------|---------|------------|
| 0.95-1.0 | All claims grounded in context | Low risk. Suitable for production. |
| 0.8-0.95 | Most claims grounded, minor additions | Moderate risk. Review additions manually. |
| 0.6-0.8 | Some hallucinated claims | High risk. Not suitable for high-stakes use. |
| 0.0-0.6 | Frequent hallucination | Critical. Generator is unreliable. |

**Answer Relevancy scores**:

| Score | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | Response directly addresses the question | No action needed |
| 0.7-0.9 | Mostly relevant with minor tangents | Tighten prompt template |
| 0.5-0.7 | Significant irrelevant content | Overhaul prompt template, filter context |
| 0.0-0.5 | Response is mostly off-topic | Fundamental prompt engineering issue |

**Answer Correctness scores (when using ground truth)**:

| Score | Meaning | Likely Cause |
|-------|---------|-------------|
| 0.9-1.0 | Highly accurate and complete | System is working well |
| 0.7-0.9 | Mostly correct, minor gaps | Missing context or minor hallucination |
| 0.5-0.7 | Partially correct | Retriever issues (missing key info) or generator issues |
| 0.0-0.5 | Mostly incorrect | Fundamental pipeline failure (retriever or generator or both) |

---

## Summary: Generator Metric Quick Reference

| Metric | Framework | Needs LLM | Needs Embeddings | Needs Reference | Measures |
|--------|-----------|-----------|------------------|-----------------|----------|
| AnswerRelevancyMetric | DeepEval | Yes | No | No | Response relevance to query |
| FaithfulnessMetric | DeepEval | Yes | No | No (needs retrieval_context) | Grounding in context |
| HallucinationMetric | DeepEval | Yes | No | No (needs context) | Contradiction with ground truth |
| GEval | DeepEval | Yes | No | Configurable | Any custom criteria |
| Faithfulness | RAGAS | Yes | No | No (needs retrieved_contexts) | Grounding in context |
| AnswerRelevancy | RAGAS | Yes | Yes | No | Response relevance (via embeddings) |
| AnswerCorrectness | RAGAS | Yes | Yes | Yes | Factual + semantic correctness |
| FactualCorrectness | RAGAS | Yes | No | Yes | Claim-level precision/recall/F1 |
| SemanticSimilarity | RAGAS | No | Yes | Yes | Embedding-based similarity |

### Decision Tree: Which Generator Metric to Use?

```
What do you want to evaluate?
│
├── "Is the response grounded in what was retrieved?"
│   ├── Have retrieval_context? -> DeepEval FaithfulnessMetric or RAGAS Faithfulness
│   └── Have ground truth context? -> DeepEval HallucinationMetric
│
├── "Does the response answer the question?"
│   ├── Want interpretable output? -> DeepEval AnswerRelevancyMetric
│   └── Want embedding-based stability? -> RAGAS AnswerRelevancy
│
├── "Is the response factually correct?"
│   ├── Have ground truth answer? -> RAGAS FactualCorrectness or AnswerCorrectness
│   └── No ground truth? -> Cannot measure correctness directly; use Faithfulness as proxy
│
├── "Is the response well-written?"
│   └── DeepEval GEval with custom criteria (coherence, fluency, clarity)
│
└── "Custom domain-specific quality?"
    └── DeepEval GEval with domain-specific evaluation_steps
```

### Combining Metrics for Comprehensive Evaluation

The most robust evaluation uses multiple complementary metrics:

```python
# Comprehensive generator evaluation suite
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

# Core RAG metrics
relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.8, penalize_ambiguous_claims=True)

# Quality metrics via G-Eval
coherence = GEval(
    name="Coherence",
    evaluation_steps=["Check logical flow", "Check for contradictions", "Assess clarity"],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7
)

completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify key points in expected output",
        "Check coverage in actual output",
        "Penalize important omissions"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7
)

# Run all metrics
from deepeval import evaluate
evaluate(test_cases=[test_case], metrics=[relevancy, faithfulness, coherence, completeness])
```

This combination covers:
- **Faithfulness**: Is it grounded in context? (catches hallucination)
- **Relevancy**: Does it answer the question? (catches off-topic responses)
- **Coherence**: Is it well-written? (catches incoherent generation)
- **Completeness**: Is it thorough? (catches incomplete answers)

Together, these four metrics provide a comprehensive view of generator quality and clearly
indicate which dimension needs improvement.

---

*Previous: [07 - Retriever Metrics Deep Dive](07_retriever_metrics_deep_dive.md) | Next: [09 - Agentic RAG Evaluation](09_agentic_rag_evaluation.md)*

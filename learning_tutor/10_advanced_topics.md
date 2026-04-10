# Chapter 10: Advanced Topics in RAG Evaluation

## Table of Contents

1. [Custom Metrics](#custom-metrics)
2. [Synthetic Dataset Generation](#synthetic-dataset-generation)
3. [CI/CD Integration](#cicd-integration)
4. [Production Monitoring](#production-monitoring)
5. [Evaluation-Driven Development](#evaluation-driven-development)
6. [Comparing Evaluation Frameworks](#comparing-evaluation-frameworks)
7. [Common Pitfalls and Anti-Patterns](#common-pitfalls-and-anti-patterns)
8. [The Future of RAG Evaluation](#the-future-of-rag-evaluation)

---

## Custom Metrics

Off-the-shelf metrics cover the most common evaluation needs, but real-world applications often require custom evaluation logic. This section covers how to build custom metrics in both DeepEval and RAGAS, plus using G-Eval for maximum flexibility.

### DeepEval: Creating Custom Metrics with BaseMetric

DeepEval provides two base classes for custom metrics:
- `BaseMetric` for single-turn evaluation
- `BaseConversationalMetric` for multi-turn evaluation

Both integrate seamlessly with DeepEval's ecosystem: CI/CD pipelines, metric caching, multi-processing, and the Confident AI platform.

#### The Five Rules of Custom Metric Creation

**Rule 1: Inherit from the correct base class**

```python
from deepeval.metrics import BaseMetric

class MyCustomMetric(BaseMetric):
    ...

# OR for multi-turn:
from deepeval.metrics import BaseConversationalMetric

class MyConversationalMetric(BaseConversationalMetric):
    ...
```

**Rule 2: Implement `__init__()` with at minimum a `threshold`**

```python
class MyCustomMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: str = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
    ):
        self.threshold = threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
```

**Rule 3: Implement `measure()` and `a_measure()`**

Both methods must accept an `LLMTestCase` (or `ConversationalTestCase`) and set `self.score` and `self.success`. Optionally set `self.reason` and `self.error`.

**Rule 4: Implement `is_successful()`**

```python
def is_successful(self) -> bool:
    if self.error is not None:
        self.success = False
    else:
        try:
            self.success = self.score >= self.threshold
        except TypeError:
            self.success = False
    return self.success
```

**Rule 5: Name your metric with a `__name__` property**

```python
@property
def __name__(self):
    return "My Custom Metric"
```

#### Full Example: Deterministic Custom Metric (JSON Schema Validation)

This metric checks whether the LLM output is valid JSON conforming to an expected schema. No LLM judge is needed -- it is purely deterministic.

```python
import json
from typing import Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class JSONSchemaValidationMetric(BaseMetric):
    """
    Deterministic metric that validates whether the LLM output
    is valid JSON conforming to an expected schema.
    """

    def __init__(
        self,
        expected_schema: dict,
        threshold: float = 1.0,
        strict_mode: bool = True,
    ):
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.expected_schema = expected_schema
        self.evaluation_model = None  # No LLM needed
        self.include_reason = True
        self.async_mode = False
        self.error = None
        self.reason = None
        self.score = None
        self.success = None

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            output = test_case.actual_output

            # Step 1: Check if output is valid JSON
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as e:
                self.score = 0.0
                self.reason = f"Output is not valid JSON: {str(e)}"
                self.success = False
                return self.score

            # Step 2: Check required keys
            required_keys = set(self.expected_schema.get("required", []))
            present_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
            missing_keys = required_keys - present_keys

            if missing_keys:
                self.score = len(present_keys & required_keys) / len(required_keys)
                self.reason = f"Missing required keys: {missing_keys}"
                self.success = self.score >= self.threshold
                return self.score

            # Step 3: Check types of values
            properties = self.expected_schema.get("properties", {})
            type_map = {"string": str, "integer": int, "number": (int, float),
                       "boolean": bool, "array": list, "object": dict}
            correct_types = 0
            total_checks = 0

            for key, schema in properties.items():
                if key in parsed:
                    total_checks += 1
                    expected_type = type_map.get(schema.get("type", "string"), str)
                    if isinstance(parsed[key], expected_type):
                        correct_types += 1

            self.score = correct_types / total_checks if total_checks > 0 else 1.0
            self.reason = f"{correct_types}/{total_checks} fields have correct types"
            self.success = self.score >= self.threshold
            return self.score

        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)  # Deterministic, no async needed

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "JSON Schema Validation"


# Usage
schema = {
    "required": ["answer", "confidence", "sources"],
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "sources": {"type": "array"}
    }
}

metric = JSONSchemaValidationMetric(expected_schema=schema)
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output='{"answer": "Paris", "confidence": 0.95, "sources": ["Wikipedia"]}'
)

metric.measure(test_case)
print(f"Score: {metric.score}")   # 1.0
print(f"Reason: {metric.reason}") # "3/3 fields have correct types"
```

#### Full Example: Deterministic Custom Metric (Word Count)

```python
class WordCountMetric(BaseMetric):
    """Checks if the output meets minimum and maximum word count requirements."""

    def __init__(self, min_words: int = 10, max_words: int = 500, threshold: float = 1.0):
        self.min_words = min_words
        self.max_words = max_words
        self.threshold = threshold
        self.evaluation_model = None
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.error = None
        self.reason = None
        self.score = None
        self.success = None

    def measure(self, test_case: LLMTestCase) -> float:
        word_count = len(test_case.actual_output.split())
        if self.min_words <= word_count <= self.max_words:
            self.score = 1.0
            self.reason = f"Word count ({word_count}) is within range [{self.min_words}, {self.max_words}]"
        elif word_count < self.min_words:
            self.score = word_count / self.min_words
            self.reason = f"Too short: {word_count} words (minimum {self.min_words})"
        else:
            self.score = self.max_words / word_count
            self.reason = f"Too long: {word_count} words (maximum {self.max_words})"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Word Count"
```

#### Full Example: LLM-Based Custom Metric (Composite)

This example combines AnswerRelevancy and Faithfulness into a single metric, using the minimum score of both:

```python
from typing import Optional
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


class FaithfulRelevancyMetric(BaseMetric):
    """
    Composite LLM-based metric that combines AnswerRelevancy
    and Faithfulness, reporting the minimum of both.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: Optional[str] = "gpt-4o",
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.error = None
        self.reason = None
        self.score = None
        self.success = None

    def _create_sub_metrics(self):
        relevancy = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
        )
        faithfulness = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
        )
        return relevancy, faithfulness

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            relevancy, faithfulness = self._create_sub_metrics()
            relevancy.measure(test_case)
            faithfulness.measure(test_case)

            composite_score = min(relevancy.score, faithfulness.score)
            self.score = (
                0 if self.strict_mode and composite_score < self.threshold
                else composite_score
            )

            if self.include_reason:
                self.reason = (
                    f"Relevancy ({relevancy.score:.2f}): {relevancy.reason}\n"
                    f"Faithfulness ({faithfulness.score:.2f}): {faithfulness.reason}"
                )

            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        try:
            relevancy, faithfulness = self._create_sub_metrics()
            await relevancy.a_measure(test_case)
            await faithfulness.a_measure(test_case)

            composite_score = min(relevancy.score, faithfulness.score)
            self.score = (
                0 if self.strict_mode and composite_score < self.threshold
                else composite_score
            )

            if self.include_reason:
                self.reason = (
                    f"Relevancy ({relevancy.score:.2f}): {relevancy.reason}\n"
                    f"Faithfulness ({faithfulness.score:.2f}): {faithfulness.reason}"
                )

            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Faithful Relevancy"


# Usage
metric = FaithfulRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What is photosynthesis?",
    actual_output="Photosynthesis converts sunlight into chemical energy in plants.",
    retrieval_context=["Photosynthesis is the process by which plants convert sunlight into energy."]
)
metric.measure(test_case)
print(f"Score: {metric.score:.2f}")
print(f"Reason:\n{metric.reason}")
```

### RAGAS: Creating Custom Metrics

RAGAS provides base classes for creating custom metrics that integrate with its evaluation framework.

#### SingleTurnMetric

For evaluating individual query-response pairs:

```python
from ragas.metrics.base import SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample
from dataclasses import dataclass, field


@dataclass
class ResponseLengthMetric(SingleTurnMetric):
    """Custom RAGAS metric that evaluates response length."""
    name: str = "response_length"
    min_length: int = 50
    max_length: int = 500

    def init(self, run_config=None):
        pass  # No initialization needed for deterministic metric

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        response = sample.response
        word_count = len(response.split())

        if self.min_length <= word_count <= self.max_length:
            return 1.0
        elif word_count < self.min_length:
            return word_count / self.min_length
        else:
            return self.max_length / word_count

    async def _ascore(self, row, callbacks=None) -> float:
        return await self._single_turn_ascore(
            SingleTurnSample(**row), callbacks
        )
```

#### MultiTurnMetric

For evaluating multi-turn conversations:

```python
from ragas.metrics.base import MultiTurnMetric
from ragas.dataset_schema import MultiTurnSample


@dataclass
class ConversationLengthMetric(MultiTurnMetric):
    """Custom RAGAS metric evaluating conversation efficiency."""
    name: str = "conversation_efficiency"
    max_turns: int = 10

    def init(self, run_config=None):
        pass

    async def _multi_turn_ascore(
        self, sample: MultiTurnSample, callbacks=None
    ) -> float:
        num_turns = len(sample.interaction) if sample.interaction else 0
        if num_turns <= self.max_turns:
            return 1.0
        else:
            return self.max_turns / num_turns

    async def _ascore(self, row, callbacks=None) -> float:
        return await self._multi_turn_ascore(
            MultiTurnSample(**row), callbacks
        )
```

#### MetricWithLLM

For custom metrics that use an LLM as a judge:

```python
from ragas.metrics.base import SingleTurnMetric, MetricWithLLM
from ragas.llms import BaseRagasLLM
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field
from dataclasses import dataclass


class ToneInput(BaseModel):
    response: str = Field(description="The LLM response to evaluate")
    expected_tone: str = Field(description="The expected tone")

class ToneOutput(BaseModel):
    score: float = Field(description="Tone alignment score from 0 to 1")
    reason: str = Field(description="Reason for the score")


class ToneEvalPrompt(PydanticPrompt[ToneInput, ToneOutput]):
    instruction = """Evaluate whether the given response matches the expected tone.
    Score from 0 to 1 where 1 means perfect tone alignment."""
    input_model = ToneInput
    output_model = ToneOutput


@dataclass
class ToneMetric(MetricWithLLM, SingleTurnMetric):
    """Custom RAGAS metric using LLM to evaluate response tone."""
    name: str = "tone_alignment"
    expected_tone: str = "professional"
    eval_prompt: PydanticPrompt = field(default_factory=ToneEvalPrompt)

    def init(self, run_config=None):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        prompt_input = ToneInput(
            response=sample.response,
            expected_tone=self.expected_tone
        )
        result = await self.llm.generate(
            self.eval_prompt.format(prompt_input)
        )
        parsed = ToneOutput.parse_raw(result.generations[0][0].text)
        return parsed.score

    async def _ascore(self, row, callbacks=None) -> float:
        return await self._single_turn_ascore(
            SingleTurnSample(**row), callbacks
        )
```

#### MetricWithEmbeddings

For custom metrics that use embedding similarity:

```python
from ragas.metrics.base import SingleTurnMetric, MetricWithEmbeddings
from ragas.dataset_schema import SingleTurnSample
from dataclasses import dataclass
import numpy as np


@dataclass
class SemanticSimilarityMetric(MetricWithEmbeddings, SingleTurnMetric):
    """Custom RAGAS metric using embeddings to measure semantic similarity."""
    name: str = "semantic_similarity"

    def init(self, run_config=None):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        response_embedding = await self.embeddings.embed_query(sample.response)
        reference_embedding = await self.embeddings.embed_query(sample.reference)

        # Cosine similarity
        dot_product = np.dot(response_embedding, reference_embedding)
        norm_a = np.linalg.norm(response_embedding)
        norm_b = np.linalg.norm(reference_embedding)
        similarity = dot_product / (norm_a * norm_b)

        return float(max(0, similarity))  # Clamp to [0, 1]

    async def _ascore(self, row, callbacks=None) -> float:
        return await self._single_turn_ascore(
            SingleTurnSample(**row), callbacks
        )
```

### G-Eval for Custom Criteria (Maximum Flexibility)

G-Eval is DeepEval's most versatile metric. It uses LLM-as-a-judge with chain-of-thought reasoning to evaluate any custom criteria you define. It originated from the paper "NLG Evaluation using GPT-4 with Better Human Alignment."

#### How G-Eval Works

**Step 1: Chain-of-Thought Evaluation Steps Generation**
G-Eval generates a series of evaluation steps using CoT prompting based on your `criteria`. If you supply explicit `evaluation_steps`, this generation is skipped (giving you more deterministic control).

**Step 2: Score Determination via Form-Filling Paradigm**
The algorithm:
1. Concatenates evaluation steps with all supplied test case parameters into a prompt
2. Asks the LLM to generate a score between 1-5
3. Uses output token probabilities to normalize the score via weighted summation

The probability-based normalization minimizes bias in LLM scoring and is automatically handled by DeepEval.

#### G-Eval Parameters

**Three mandatory parameters**:

| Parameter | Description |
|-----------|-------------|
| `name` | Name of the custom metric |
| `criteria` | Description of the evaluation aspects (OR provide `evaluation_steps`) |
| `evaluation_params` | List of `LLMTestCaseParams` relevant to evaluation |

**Seven optional parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evaluation_steps` | None | Explicit steps for the LLM; skips CoT generation |
| `rubric` | None | List of `Rubric` objects to confine score ranges |
| `threshold` | 0.5 | Passing threshold |
| `model` | 'gpt-4.1' | Judge model |
| `strict_mode` | False | Binary scoring |
| `async_mode` | True | Concurrent execution |
| `verbose_mode` | False | Print intermediate steps |

**Important**: You must provide either `criteria` or `evaluation_steps`, not both. Only include parameters mentioned in your criteria/steps in `evaluation_params`.

#### Example: Medical Accuracy

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

medical_accuracy = GEval(
    name="Medical Accuracy",
    evaluation_steps=[
        "Extract all medical claims, diagnoses, or treatment recommendations from the actual output.",
        "Verify each medical claim against the retrieved context from medical literature.",
        "Check for any medical claims not supported by the context (hallucinations).",
        "Heavily penalize any hallucinated medical advice that could lead to patient harm.",
        "Assess whether appropriate medical disclaimers are included.",
        "Evaluate whether the response appropriately indicates when professional consultation is needed.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.8,
    model="gpt-4o",
)

# Usage
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

test_case = LLMTestCase(
    input="What are the symptoms of diabetes?",
    actual_output="Common symptoms include frequent urination, excessive thirst, and unexplained weight loss. Please consult your physician for a proper diagnosis.",
    retrieval_context=["Type 2 diabetes symptoms include polyuria (frequent urination), polydipsia (excessive thirst), unexplained weight loss, fatigue, and blurred vision."]
)

evaluate(test_cases=[test_case], metrics=[medical_accuracy])
```

#### Example: Legal Compliance

```python
legal_compliance = GEval(
    name="Legal Compliance",
    evaluation_steps=[
        "Identify all legal claims or interpretations in the actual output.",
        "Check if any statements could be construed as legal advice without proper disclaimers.",
        "Verify that cited regulations or laws in the output match those in the context.",
        "Assess whether the response appropriately distinguishes between general information and legal advice.",
        "Check for any jurisdictional specificity claims not supported by context.",
        "Penalize any statement that could create a lawyer-client relationship expectation.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.85,
)
```

#### Example: Code Quality

```python
code_quality = GEval(
    name="Code Quality",
    evaluation_steps=[
        "Check if the generated code is syntactically valid and would compile/run without errors.",
        "Evaluate whether the code correctly addresses the user's programming question.",
        "Assess code readability: meaningful variable names, comments, and clear structure.",
        "Check for potential security issues: SQL injection, XSS, buffer overflows.",
        "Evaluate efficiency: no obvious O(n^2) where O(n) would work.",
        "Verify that the code handles edge cases mentioned in the input.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
)
```

#### Example: Tone / Professionalism

```python
professionalism = GEval(
    name="Professionalism",
    evaluation_steps=[
        "Determine whether the actual output maintains a professional tone throughout.",
        "Evaluate if the language reflects expertise and domain-appropriate formality.",
        "Ensure the output stays contextually appropriate and avoids casual expressions.",
        "Check if the output is clear, respectful, and avoids slang or informal phrasing.",
        "Verify that the response maintains empathy when addressing sensitive topics.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
)
```

#### Example: Answer Correctness (Reference-Based)

```python
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'.",
        "Heavily penalize omission of important details present in the expected output.",
        "Vague language or contradicting OPINIONS (as opposed to facts) are acceptable.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)
```

#### Using Rubrics for Score Calibration

Rubrics constrain the LLM judge to specific score ranges (0-10 inclusive):

```python
from deepeval.metrics.g_eval import Rubric

grounded_correctness = GEval(
    name="Grounded Correctness",
    criteria="Evaluate factual correctness based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect or contradicts expected output."),
        Rubric(score_range=(3, 4), expected_outcome="Contains significant factual errors or omissions."),
        Rubric(score_range=(5, 6), expected_outcome="Mostly correct with some minor errors."),
        Rubric(score_range=(7, 8), expected_outcome="Correct with minor omissions."),
        Rubric(score_range=(9, 10), expected_outcome="100% factually correct and complete."),
    ],
)
```

**Rules for rubrics**: Different rubrics must not have overlapping `score_range` values. Start and end values can be equal (representing a single score point).

#### Customizing G-Eval's Prompt Template

Override default prompts by subclassing `GEvalTemplate`:

```python
from deepeval.metrics.g_eval import GEvalTemplate
import textwrap

class CustomGEvalTemplate(GEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return textwrap.dedent(
            f"""You are an expert evaluator for {parameters}.
            Based on the following criteria, produce 3-4 evaluation steps.
            Criteria: {criteria}
            Return JSON: {{"steps": ["Step 1", "Step 2", "Step 3"]}}
            JSON:"""
        )

metric = GEval(
    name="Custom",
    criteria="Evaluate response quality",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_template=CustomGEvalTemplate
)
```

**Design tip**: Start with a short `criteria`, observe the auto-generated evaluation steps, then refine them manually into explicit `evaluation_steps` for more deterministic behavior across runs.

---

## Synthetic Dataset Generation

### Why Synthetic Data?

Building evaluation datasets is one of the biggest barriers to implementing RAG evaluation. Synthetic data generation solves the bootstrapping problem -- creating diverse, high-quality test cases when you have no production data.

Use synthetic data when:
- You are building a new RAG system and have zero production queries
- Your production data is too sensitive to use for evaluation
- You need to test edge cases that rarely occur in production
- You want to systematically cover different question types and difficulty levels
- You need to augment a small existing dataset

### DeepEval Synthesizer

DeepEval's `Synthesizer` generates high-quality single-turn and multi-turn goldens through a sophisticated pipeline involving input generation, quality filtration, complexity evolution, and output styling.

**Important**: The Synthesizer generates `input`s (queries) and optionally `expected_output`s, but does NOT generate `actual_output`s -- those must come from your LLM application.

#### Creating a Synthesizer

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer(
    model="gpt-4o",        # LLM for generation
    async_mode=True,       # Concurrent generation
    max_concurrent=100,    # Max parallel requests
)
```

**Constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str/DeepEvalBaseLLM | 'gpt-4.1' | LLM for generation |
| `async_mode` | bool | True | Concurrent generation |
| `max_concurrent` | int | 100 | Max parallel requests (reduce if rate-limited) |
| `filtration_config` | FiltrationConfig | defaults | Controls quality filtering |
| `evolution_config` | EvolutionConfig | defaults | Controls complexity evolution |
| `styling_config` | StylingConfig | defaults | Controls output format/style |

#### Method 1: generate_goldens_from_docs()

Best when you have a knowledge base in document form. DeepEval handles context extraction automatically.

```python
synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[
        'knowledge_base.pdf',
        'faq.txt',
        'product_docs.docx',
        'api_reference.md'
    ],
    include_expected_output=True
)

print(f"Generated {len(goldens)} goldens")
for golden in goldens[:3]:
    print(f"Input: {golden.input}")
    print(f"Expected: {golden.expected_output}")
    print(f"Context: {golden.context[:100]}...")
    print("---")
```

Supported file types: `.txt`, `.docx`, `.pdf`, `.md`, `.markdown`, `.mdx`

#### Method 2: generate_goldens_from_contexts()

Best when you have already chunked your documents and want precise control over context:

```python
contexts = [
    ["Our return policy allows returns within 30 days of purchase.",
     "Refunds are processed within 5-7 business days."],
    ["Premium plans include 24/7 support and dedicated account management.",
     "Enterprise plans add custom integrations and SLA guarantees."],
    ["API rate limits are 1000 requests per minute for free tier.",
     "Pro tier offers 10,000 requests per minute with priority routing."],
]

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True
)
```

#### Method 3: generate_goldens_from_scratch()

Best when no knowledge base contexts are needed (topic-based generation):

```python
goldens = synthesizer.generate_goldens_from_scratch(
    subject="Customer support for an e-commerce platform",
    task="Answer customer queries about orders, returns, and products",
    num_goldens=50,
    include_expected_output=True
)
```

#### Method 4: generate_goldens_from_goldens()

Augment an existing dataset by generating variations:

```python
existing_goldens = [
    Golden(input="What is the return policy?",
           expected_output="30-day return policy"),
    Golden(input="How do I track my order?",
           expected_output="Use the tracking link in your confirmation email"),
]

augmented = synthesizer.generate_goldens_from_goldens(
    goldens=existing_goldens,
    num_goldens=20,
    include_expected_output=True
)
```

#### The Generation Pipeline

The synthesizer follows four main steps:

**Step 1: Input Generation** -- Golden inputs are generated using an LLM, optionally grounded in provided contexts.

**Step 2: Filtration** -- Each synthetic input receives a quality score (0-1) based on self-containment (understandable without additional context) and clarity (unambiguous intent). Inputs below the threshold are regenerated up to `max_quality_retries` times.

```python
from deepeval.synthesizer.config import FiltrationConfig

filtration_config = FiltrationConfig(
    critic_model="gpt-4o",
    synthetic_input_quality_threshold=0.5,
    max_quality_retries=3,
)
```

**Step 3: Evolution** -- Filtered inputs are rewritten with increasing complexity. Each evolution is sampled from a configured distribution. This technique originates from the Evol-Instruct paper (arXiv:2304.12244).

```python
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Evolution

evolution_config = EvolutionConfig(
    num_evolutions=4,
    evolutions={
        Evolution.REASONING: 1/4,       # Requires multi-step reasoning
        Evolution.MULTICONTEXT: 1/4,     # Requires combining multiple contexts
        Evolution.CONCRETIZING: 1/4,     # Makes queries more specific
        Evolution.CONSTRAINED: 1/4,      # Adds constraints to the query
    }
)
```

**Seven evolution types**:

| Evolution | Description | RAG-Safe* |
|-----------|-------------|-----------|
| `REASONING` | Requires multi-step logical reasoning | No |
| `MULTICONTEXT` | Requires synthesizing info from multiple contexts | Yes |
| `CONCRETIZING` | Makes the query more specific and detailed | Yes |
| `CONSTRAINED` | Adds constraints or conditions | Yes |
| `COMPARATIVE` | Requires comparing entities | Yes |
| `HYPOTHETICAL` | Introduces hypothetical scenarios | No |
| `IN_BREADTH` | Broadens the scope of the query | No |

*RAG-Safe means the answer is guaranteed to be derivable from the provided context.

**Step 4: Styling** -- Inputs and expected outputs are rewritten into desired formats:

```python
from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
    input_format="Questions in English that ask for SQL data",
    expected_output_format="SQL query based on the given input",
    task="Answering text-to-SQL queries",
    scenario="Non-technical users querying a database with plain English",
)
```

#### Saving Synthetic Data

```python
# As pandas DataFrame
df = synthesizer.to_pandas()
print(df.columns)
# Columns: input, actual_output, expected_output, context, retrieval_context,
#           n_chunks_per_context, context_length, context_quality,
#           synthetic_input_quality, evolutions, source_file

# Save locally
synthesizer.save_as(file_type='json', directory="./synthetic_data")
synthesizer.save_as(file_type='csv', directory="./synthetic_data")

# Push to Confident AI
from deepeval.dataset import EvaluationDataset
dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
dataset.push(alias="My Synthetic Dataset v1")
```

### RAGAS TestsetGenerator

RAGAS takes a different approach to synthetic data generation, using a knowledge graph to create diverse test sets with controlled distributions.

#### KnowledgeGraph-Based Generation

RAGAS builds a knowledge graph from your documents, then generates questions that test different reasoning patterns over that graph:

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# Load documents
loader = DirectoryLoader("./knowledge_base/", glob="**/*.md")
documents = loader.load()

# Create generator
generator_llm = ChatOpenAI(model="gpt-4o")
critic_llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

# Generate with controlled distribution
testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=100,
    distributions={
        simple: 0.3,          # 30% simple factual questions
        reasoning: 0.3,       # 30% requiring multi-step reasoning
        multi_context: 0.2,   # 20% requiring multiple document synthesis
        conditional: 0.2,     # 20% with conditional/hypothetical elements
    }
)

# Convert to pandas DataFrame
df = testset.to_pandas()
print(df.columns)  # question, contexts, ground_truth, evolution_type, metadata
```

#### Evolution Types in RAGAS

| Type | Description | Example |
|------|-------------|---------|
| `simple` | Direct factual questions | "What is the return policy?" |
| `reasoning` | Multi-step logical reasoning | "If a customer bought item X and Y, what discount applies?" |
| `multi_context` | Requires multiple documents | "Compare the features of Plan A and Plan B" |
| `conditional` | Hypothetical or conditional | "What would happen if the warranty expired?" |

#### Controlling Distribution

The `distributions` parameter controls what percentage of questions fall into each category. The values must sum to 1.0:

```python
# Heavy on reasoning (for testing complex QA)
distributions = {
    simple: 0.1,
    reasoning: 0.5,
    multi_context: 0.3,
    conditional: 0.1,
}

# Balanced distribution (default-like)
distributions = {
    simple: 0.25,
    reasoning: 0.25,
    multi_context: 0.25,
    conditional: 0.25,
}
```

### Best Practices for Synthetic Data

#### 1. Validate with Human Review

Synthetic data is only useful if it reflects real user behavior. Always have humans review a sample:

```python
# Generate a small batch for review
review_goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['knowledge_base.pdf'],
    include_expected_output=True,
)

# Export for human review
df = synthesizer.to_pandas()
df[['input', 'expected_output', 'context']].to_csv('for_human_review.csv')

# After review, filter based on human annotations
# Human reviewers mark each row as "valid" or "invalid"
reviewed = pd.read_csv('reviewed_by_humans.csv')
valid_goldens = [g for g, r in zip(review_goldens, reviewed['valid']) if r]
```

#### 2. Ensure Diversity

Synthetic generators can produce repetitive patterns. Actively ensure diversity across:

- **Topics**: Cover all areas of your knowledge base
- **Difficulty levels**: Mix simple, medium, and complex questions
- **Question types**: Factual, comparative, hypothetical, procedural
- **Phrasing**: Formal, informal, ambiguous, precise
- **Edge cases**: Questions with no answer in the KB, multi-language queries

#### 3. Augment Over Time with Production Data

Start with synthetic data, then progressively replace it with production data:

```python
# Phase 1: 100% synthetic (launch)
# Phase 2: 70% synthetic + 30% production (after 1 month)
# Phase 3: 30% synthetic + 70% production (after 3 months)
# Phase 4: 10% synthetic (edge cases) + 90% production (steady state)
```

---

## CI/CD Integration

### Why CI/CD for RAG Evaluation?

RAG systems change frequently: prompt templates are updated, chunk sizes are tuned, embedding models are swapped, retrieval strategies evolve. Without CI/CD integration, these changes can silently degrade performance. CI/CD catches regressions before deployment.

### DeepEval in CI/CD

#### deepeval test run

The primary CLI command for running evaluations in CI/CD:

```bash
deepeval test run test_llm_app.py
```

This is preferred over raw `pytest` because it adds enhanced functionality: testing reports, parallel execution, caching, and integration with Confident AI.

#### Writing Evaluation Tests

**Single-turn test**:

```python
# test_llm_app.py
from your_agent import your_llm_app
import pytest
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
)

# Load dataset
dataset = EvaluationDataset()
dataset.pull(alias="Production Eval Dataset")

# OR from file:
# dataset.add_goldens_from_csv_file(
#     file_path="test_data.csv",
#     input_col_name="query"
# )


@pytest.mark.parametrize("golden", dataset.goldens)
def test_rag_pipeline(golden: Golden):
    """Test the RAG pipeline against each golden test case."""
    response, retrieved_contexts = your_llm_app(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response,
        expected_output=golden.expected_output,
        retrieval_context=retrieved_contexts,
    )

    assert_test(
        test_case=test_case,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.8),
            ContextualPrecisionMetric(threshold=0.6),
        ]
    )
```

**Component-level test** (using tracing):

```python
@pytest.mark.parametrize("golden", dataset.goldens)
def test_rag_components(golden: Golden):
    """Test individual components of the RAG pipeline."""
    assert_test(
        golden=golden,
        observed_callback=your_llm_app,  # Must be @observe-decorated
    )
```

**Hyperparameter logging**:

```python
import deepeval

@deepeval.log_hyperparameters(model="gpt-4o", prompt_template="v2")
def hyperparameters():
    return {
        "model": "gpt-4o",
        "chunk_size": 512,
        "top_k": 5,
        "temperature": 0.1,
        "embedding_model": "text-embedding-3-large",
        "system_prompt": "You are a helpful customer support agent...",
    }
```

#### GitHub Actions Workflow

```yaml
name: RAG Evaluation Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    paths:
      # Only run when RAG-related code changes
      - 'src/rag/**'
      - 'src/prompts/**'
      - 'src/retrieval/**'
      - 'tests/eval/**'

jobs:
  # Fast deterministic tests (always run)
  deterministic-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install deepeval

      - name: Run Deterministic Eval Tests
        run: deepeval test run tests/eval/test_deterministic.py

  # LLM-based tests (on PRs to main only)
  llm-eval-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install deepeval

      - name: Run LLM-Based Eval Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
        run: deepeval test run tests/eval/test_llm_evals.py
```

#### Threshold Management

Define thresholds centrally and update them as your system improves:

```python
# eval_config.py
THRESHOLDS = {
    "answer_relevancy": 0.75,
    "faithfulness": 0.85,
    "contextual_precision": 0.70,
    "contextual_recall": 0.65,
    "bias": 0.3,           # Maximum acceptable (lower is better)
    "toxicity": 0.1,       # Maximum acceptable (lower is better)
}

# In test files:
from eval_config import THRESHOLDS

metrics = [
    AnswerRelevancyMetric(threshold=THRESHOLDS["answer_relevancy"]),
    FaithfulnessMetric(threshold=THRESHOLDS["faithfulness"]),
]
```

#### Regression Detection

Regression detection works through `assert_test()` -- tests fail when metrics fall below their thresholds, blocking deployment. For more sophisticated regression detection, track metrics over time:

```python
# Compare current run against baseline
import json

def check_regression(current_results, baseline_path="baseline_metrics.json"):
    with open(baseline_path) as f:
        baseline = json.load(f)

    regressions = []
    for metric_name, current_score in current_results.items():
        baseline_score = baseline.get(metric_name, 0)
        # Flag if score dropped by more than 5%
        if current_score < baseline_score - 0.05:
            regressions.append({
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "drop": baseline_score - current_score,
            })

    if regressions:
        for r in regressions:
            print(f"REGRESSION: {r['metric']} dropped from {r['baseline']:.3f} to {r['current']:.3f} (-{r['drop']:.3f})")
        raise AssertionError(f"{len(regressions)} metric(s) regressed!")
```

### RAGAS in CI/CD

RAGAS integrates into CI/CD through its `evaluate()` function with assertion checks:

```python
# test_rag_ragas.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

def test_rag_with_ragas():
    """RAGAS-based evaluation in CI."""
    # Prepare evaluation data
    eval_data = {
        "question": ["What is RAG?", "How does chunking work?"],
        "answer": [your_rag("What is RAG?"), your_rag("How does chunking work?")],
        "contexts": [
            [retrieve("What is RAG?")],
            [retrieve("How does chunking work?")]
        ],
        "ground_truth": [
            "RAG is Retrieval-Augmented Generation...",
            "Chunking splits documents into smaller segments..."
        ]
    }

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ])

    # Assert thresholds
    assert result["faithfulness"] >= 0.8, f"Faithfulness too low: {result['faithfulness']}"
    assert result["answer_relevancy"] >= 0.7, f"Relevancy too low: {result['answer_relevancy']}"
    assert result["context_precision"] >= 0.6, f"Precision too low: {result['context_precision']}"
    assert result["context_recall"] >= 0.6, f"Recall too low: {result['context_recall']}"
```

### CI/CD Best Practices

1. **Run on every PR that touches RAG components**: Use path filters in your CI configuration to trigger only when relevant files change.

2. **Separate fast and slow test suites**: Deterministic tests (JSON validation, word count, format checks) run in seconds. LLM-based tests can take minutes and cost money. Run deterministic tests on every commit, LLM tests on PRs.

3. **Cache evaluation results**: If the same input produces the same output and context, cache the evaluation result to avoid redundant LLM judge calls.

4. **Set up alerts for metric regressions**: Integrate with Slack, PagerDuty, or your alerting system to notify when metrics drop below thresholds.

5. **Use environment-specific thresholds**: Staging thresholds can be stricter than production minimums, catching issues before they reach users.

---

## Production Monitoring

### Online Evaluation vs Offline Evaluation

| Dimension | Offline Evaluation | Online Evaluation |
|-----------|-------------------|-------------------|
| **When** | Before deployment (CI/CD) | After deployment (production) |
| **Data** | Curated test sets | Real user interactions |
| **Cost** | Bounded, predictable | Ongoing, per-interaction |
| **Coverage** | Limited to test scenarios | Covers all real usage patterns |
| **Latency impact** | None (separate pipeline) | Must be minimal |
| **Purpose** | Catch regressions | Monitor drift, discover new patterns |

### Logging Production Inputs/Outputs

Capture every interaction for later evaluation:

```python
import json
import datetime

def log_interaction(query, response, contexts, metadata=None):
    """Log a production interaction for later evaluation."""
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": query,
        "actual_output": response,
        "retrieval_context": contexts,
        "metadata": metadata or {},
    }
    # Write to your logging system (e.g., S3, BigQuery, Elasticsearch)
    write_to_log_store(json.dumps(record))
```

### Sampling Strategies

Evaluating every production interaction is expensive. Use smart sampling:

```python
import random
import hashlib

def should_evaluate(query: str, sample_rate: float = 0.05) -> bool:
    """Deterministic sampling based on query hash."""
    # Using hash ensures the same query is always sampled consistently
    query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
    return (query_hash % 1000) < (sample_rate * 1000)

# Stratified sampling: higher rate for unusual queries
def sampling_rate_for_query(query: str) -> float:
    if is_adversarial(query):
        return 1.0    # Evaluate all suspicious queries
    elif is_new_topic(query):
        return 0.5    # Higher rate for new topics
    elif is_high_value(query):
        return 0.2    # Higher rate for important queries
    else:
        return 0.05   # 5% baseline
```

### DeepEval's @observe for Production Tracing

The same `@observe` decorator used for evaluation also works for production tracing:

```python
from deepeval.tracing import observe, update_current_trace

@observe(type="agent")
def production_rag(query: str):
    @observe(type="retriever")
    def retrieve(q: str):
        results = vector_store.search(q)
        update_current_trace(retrieval_context=results)
        return results

    @observe(type="llm")
    def generate(q: str, ctx: list):
        response = llm.complete(q, ctx)
        update_current_trace(input=q, output=response)
        return response

    context = retrieve(query)
    return generate(query, context)
```

When connected to Confident AI (via `deepeval login`), traces are automatically sent for visualization and monitoring.

**Environment variables for production**:
```bash
CONFIDENT_TRACE_VERBOSE=0    # Suppress console output
CONFIDENT_TRACE_FLUSH=0      # Disable immediate flushing (batch for performance)
```

### Confident AI Platform for Monitoring

Confident AI provides a dashboard for:
- **Trace visualization**: See the full execution tree for any interaction
- **Metric trends**: Track metric scores over time
- **Dataset curation**: Annotate production interactions to build evaluation datasets
- **Performance monitoring**: Track latency, cost, and error rates
- **Regression alerts**: Get notified when metrics drop

### Building Feedback Loops

The most powerful monitoring pattern is a feedback loop where production data feeds back into your evaluation pipeline:

```python
# Step 1: Log production interactions
production_logs = get_recent_logs(days=7)

# Step 2: Sample for evaluation
sampled = [log for log in production_logs if should_evaluate(log["input"])]

# Step 3: Evaluate sampled interactions
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

test_cases = [
    LLMTestCase(
        input=log["input"],
        actual_output=log["actual_output"],
        retrieval_context=log["retrieval_context"],
    )
    for log in sampled
]

results = evaluate(
    test_cases=test_cases,
    metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()]
)

# Step 4: Identify failures and add to evaluation dataset
failures = [r for r in results if not all(m.success for m in r.metrics)]
for failure in failures:
    # Add to curated evaluation dataset with human review
    add_to_review_queue(failure)

# Step 5: Re-evaluate periodically to track trends
```

### A/B Testing with Evaluation Metrics

When experimenting with RAG changes, use evaluation metrics to compare variants:

```python
def ab_test_evaluation(variant_a_results, variant_b_results):
    """Compare two RAG variants using evaluation metrics."""
    metrics = [FaithfulnessMetric(), AnswerRelevancyMetric()]

    scores_a = {m.__name__: [] for m in metrics}
    scores_b = {m.__name__: [] for m in metrics}

    for tc in variant_a_results:
        for metric in metrics:
            metric.measure(tc)
            scores_a[metric.__name__].append(metric.score)

    for tc in variant_b_results:
        for metric in metrics:
            metric.measure(tc)
            scores_b[metric.__name__].append(metric.score)

    # Statistical comparison
    from scipy import stats
    for metric_name in scores_a:
        a_scores = scores_a[metric_name]
        b_scores = scores_b[metric_name]
        t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
        print(f"{metric_name}:")
        print(f"  Variant A: {sum(a_scores)/len(a_scores):.3f}")
        print(f"  Variant B: {sum(b_scores)/len(b_scores):.3f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            winner = "A" if sum(a_scores) > sum(b_scores) else "B"
            print(f"  -> Variant {winner} is significantly better")
        else:
            print(f"  -> No significant difference")
```

---

## Evaluation-Driven Development

### The Eval-Driven Development Cycle

Evaluation-driven development (EDD) is the RAG analog of test-driven development. Instead of writing tests first, you define evaluation metrics and thresholds first, then iteratively improve your system until it meets those criteria.

```
1. Define metrics and thresholds
       |
2. Create/update evaluation dataset
       |
3. Run evaluation -> Check scores
       |
4. Identify weakest metric
       |
5. Diagnose root cause
       |
6. Make targeted change
       |
7. Re-evaluate -> Did it improve?
       |           |
      Yes         No -> Revert, try different approach
       |
8. Repeat from step 4
```

### Using Metrics to Guide Hyperparameter Tuning

Each metric failure points to specific system components and hyperparameters to adjust:

#### Low Contextual Relevancy -> Adjust Chunk Size, Top-K

If the retrieved context contains a lot of irrelevant information:

```python
# Diagnosis: Low ContextualRelevancy (e.g., 0.45)
# Meaning: Retrieved chunks contain too much irrelevant material

# Adjustment 1: Reduce chunk size (more precise chunks)
# Before: chunk_size=1024, chunk_overlap=100
# After:  chunk_size=256, chunk_overlap=50

# Adjustment 2: Reduce top-K (fewer but more relevant chunks)
# Before: top_k=10
# After:  top_k=3

# Adjustment 3: Add a relevance score threshold
# Only include chunks with similarity > 0.7
```

#### Low Contextual Precision -> Improve Reranker

If relevant chunks exist but are buried among irrelevant ones:

```python
# Diagnosis: Low ContextualPrecision (e.g., 0.50)
# Meaning: Relevant chunks are present but not ranked at the top

# Solution 1: Add a cross-encoder reranker
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, chunks, top_k=5):
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]

# Solution 2: Use Cohere Rerank or similar API
# Solution 3: Hybrid search (combine semantic + keyword)
```

#### Low Contextual Recall -> Better Embedding Model

If relevant information exists in the KB but is not retrieved:

```python
# Diagnosis: Low ContextualRecall (e.g., 0.40)
# Meaning: The retriever is missing relevant documents

# Solution 1: Upgrade embedding model
# Before: text-embedding-ada-002
# After:  text-embedding-3-large

# Solution 2: Hybrid search (semantic + BM25)
# Solution 3: Query expansion/rewriting
def expand_query(query):
    expanded = llm.complete(
        f"Generate 3 alternative phrasings for this search query: {query}"
    )
    return [query] + parse_alternatives(expanded)

# Solution 4: Increase top-K (retrieve more, then rerank)
```

#### Low Faithfulness -> Lower Temperature, Better Prompt

If the generated response contains information not in the context:

```python
# Diagnosis: Low Faithfulness (e.g., 0.55)
# Meaning: The LLM is hallucinating beyond the provided context

# Solution 1: Lower temperature
# Before: temperature=0.7
# After:  temperature=0.1

# Solution 2: Strengthen the grounding instruction in the prompt
system_prompt = """
You are a helpful assistant. Answer the user's question based ONLY on
the provided context. If the context does not contain enough information
to answer the question, say "I don't have enough information to answer this."

IMPORTANT: Do NOT use any prior knowledge. ONLY use the context below.

Context:
{context}
"""

# Solution 3: Add explicit citation requirements
system_prompt += """
For each claim you make, cite the specific context passage that supports it.
Format: [Source: passage text]
"""
```

#### Low Answer Relevancy -> Improve Prompt Template

If the response is correct but does not directly answer the question:

```python
# Diagnosis: Low AnswerRelevancy (e.g., 0.60)
# Meaning: Response is tangential or overly verbose

# Solution 1: Add format instructions
system_prompt = """
Answer the user's question directly and concisely.
Start with the direct answer, then provide supporting details.
Do not include information that is not directly relevant to the question.
"""

# Solution 2: Add few-shot examples
system_prompt += """
Example:
Q: What is the return policy?
A: Our return policy allows returns within 30 days of purchase for a full refund.
Items must be in original condition. Contact support@example.com to initiate.
"""
```

### Systematic Experimentation

Track hyperparameters alongside metrics to build a knowledge base of what works:

```python
import json
from datetime import datetime

def run_experiment(name, config, dataset, metrics):
    """Run an evaluation experiment and log results."""
    results = evaluate(dataset=dataset, metrics=metrics)

    experiment = {
        "name": name,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "scores": {
            m.__name__: m.score for m in metrics
        },
        "pass_rate": sum(1 for m in metrics if m.success) / len(metrics),
    }

    # Append to experiment log
    with open("experiments.jsonl", "a") as f:
        f.write(json.dumps(experiment) + "\n")

    return experiment

# Run experiments
experiments = [
    ("baseline", {"chunk_size": 512, "top_k": 5, "model": "gpt-4o", "temp": 0.3}),
    ("smaller_chunks", {"chunk_size": 256, "top_k": 5, "model": "gpt-4o", "temp": 0.3}),
    ("more_retrieval", {"chunk_size": 512, "top_k": 10, "model": "gpt-4o", "temp": 0.3}),
    ("lower_temp", {"chunk_size": 512, "top_k": 5, "model": "gpt-4o", "temp": 0.1}),
    ("with_reranker", {"chunk_size": 512, "top_k": 10, "model": "gpt-4o", "temp": 0.3, "reranker": True}),
]

for name, config in experiments:
    configure_rag_pipeline(config)
    result = run_experiment(name, config, eval_dataset, eval_metrics)
    print(f"{name}: {result['scores']}")
```

### When to Optimize vs When to Ship

Not every metric needs to be perfect. Apply the 80/20 rule:

| Metric | Minimum Viable | Good | Excellent |
|--------|---------------|------|-----------|
| Faithfulness | 0.80 | 0.90 | 0.95+ |
| Answer Relevancy | 0.70 | 0.80 | 0.90+ |
| Contextual Precision | 0.60 | 0.75 | 0.85+ |
| Contextual Recall | 0.60 | 0.75 | 0.85+ |
| Bias | < 0.30 | < 0.15 | < 0.05 |
| Toxicity | < 0.10 | < 0.05 | < 0.01 |

Ship when:
- All critical metrics (faithfulness, safety) meet "Minimum Viable" thresholds
- The overall user experience is acceptable
- You have monitoring in place to catch regressions
- Further improvements have diminishing returns relative to effort

---

## Comparing Evaluation Frameworks

### DeepEval vs RAGAS vs TruLens vs Arize Phoenix vs LangSmith

| Dimension | DeepEval | RAGAS | TruLens | Arize Phoenix | LangSmith |
|-----------|----------|-------|---------|---------------|-----------|
| **Open Source** | Yes | Yes | Yes | Yes | Partial (SDK open, platform closed) |
| **RAG Metrics** | Comprehensive (8+) | Comprehensive (6+) | Good (5+) | Good (4+) | Basic |
| **Agentic Metrics** | Excellent (6 dedicated) | Good (3+) | Limited | Limited | Basic |
| **Multi-Turn** | Excellent (11 metrics) | Good (MultiTurnSample) | Limited | Limited | Good |
| **MCP Support** | Yes (dedicated metrics) | No | No | No | No |
| **Custom Metrics** | BaseMetric + G-Eval | MetricWithLLM | Custom functions | Custom evaluators | Custom evaluators |
| **Synthetic Data** | Synthesizer (4 methods) | TestsetGenerator | No | No | No |
| **Tracing** | @observe decorator | Via LangSmith/other | TruLlama/TruChain | Built-in tracing | Built-in tracing |
| **CI/CD** | deepeval test run | Manual integration | Manual | Manual | Manual |
| **Red Teaming** | DeepTeam (dedicated) | No | No | No | No |
| **Dashboard** | Confident AI | No (use notebooks) | TruLens dashboard | Phoenix UI | LangSmith UI |
| **Pricing** | Free + paid tiers | Free | Free + paid | Free + paid | Free + paid |
| **Install** | `pip install deepeval` | `pip install ragas` | `pip install trulens` | `pip install arize-phoenix` | `pip install langsmith` |

### Detailed Comparison: Metrics Coverage

| Metric Category | DeepEval | RAGAS | TruLens | Phoenix |
|----------------|----------|-------|---------|---------|
| Faithfulness | FaithfulnessMetric | faithfulness | Groundedness | Relevance evals |
| Answer Relevancy | AnswerRelevancyMetric | answer_relevancy | QS Relevance | QA Relevance |
| Context Precision | ContextualPrecisionMetric | context_precision | Context Relevance | N/A |
| Context Recall | ContextualRecallMetric | context_recall | N/A | N/A |
| Context Relevancy | ContextualRelevancyMetric | context_relevancy | Context Relevance | N/A |
| Hallucination | HallucinationMetric | N/A (use faithfulness) | Groundedness | Hallucination |
| Bias | BiasMetric | N/A | N/A | N/A |
| Toxicity | ToxicityMetric | N/A | N/A | Toxicity |
| Task Completion | TaskCompletionMetric | AgentGoalAccuracy | N/A | N/A |
| Tool Correctness | ToolCorrectnessMetric | ToolCallAccuracy | N/A | N/A |
| G-Eval (Custom) | GEval | N/A | Custom feedback | LLM-as-judge |

### Decision Tree: When to Use Which

```
START: What is your primary use case?
  |
  +-> Standard RAG evaluation
  |     +-> Need comprehensive metrics + CI/CD? -> DeepEval
  |     +-> Lightweight, research-focused? -> RAGAS
  |     +-> Already using LlamaIndex/LangChain? -> TruLens / LangSmith
  |
  +-> Agentic RAG evaluation
  |     +-> DeepEval (most comprehensive agentic metrics)
  |
  +-> MCP evaluation
  |     +-> DeepEval (only framework with MCP metrics)
  |
  +-> Multi-turn chatbot
  |     +-> DeepEval (11 multi-turn metrics)
  |
  +-> Production monitoring
  |     +-> Need traces + dashboards? -> Arize Phoenix or LangSmith
  |     +-> Need eval metrics + traces? -> DeepEval (with Confident AI)
  |
  +-> Red teaming / safety
  |     +-> DeepTeam + DeepEval (most comprehensive)
  |
  +-> Synthetic data generation
        +-> From documents? -> DeepEval Synthesizer or RAGAS TestsetGenerator
        +-> From scratch? -> DeepEval Synthesizer
```

### Using Multiple Frameworks Together

There is no rule against using multiple frameworks. A practical combination:

```python
# Use DeepEval for CI/CD, agentic metrics, and safety
from deepeval.metrics import (
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    BiasMetric,
    FaithfulnessMetric,
)

# Use RAGAS for synthetic test generation
from ragas.testset.generator import TestsetGenerator

# Use LangSmith/Phoenix for production tracing
from langsmith import traceable

@traceable  # LangSmith tracing for production
def rag_pipeline(query):
    ...

# Evaluation with DeepEval
from deepeval import evaluate
evaluate(test_cases=test_cases, metrics=[FaithfulnessMetric(), BiasMetric()])

# Generate test data with RAGAS
generator = TestsetGenerator.from_langchain(...)
testset = generator.generate_with_langchain_docs(documents, test_size=100)
```

---

## Common Pitfalls and Anti-Patterns

### 1. Evaluating with Too Few Test Cases

**The problem**: Running evaluation on 5-10 test cases and drawing conclusions from the results. With so few samples, random variation can dominate, making results unreliable.

**The fix**: Use at least 50 test cases for meaningful signal. For statistical significance when comparing two approaches, aim for 100+ cases. Use confidence intervals, not point estimates:

```python
import numpy as np
from scipy import stats

def confidence_interval(scores, confidence=0.95):
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return mean, ci

scores = [m.score for m in metric_results]
mean, (lower, upper) = confidence_interval(scores)
print(f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
```

### 2. Using Weak Models as Judges

**The problem**: Using GPT-3.5 or a small open-source model as the evaluation judge. Weaker models are unreliable judges -- they miss nuances, produce inconsistent scores, and can be easily fooled.

**The fix**: Use the strongest available model as your judge. GPT-4o is the standard. If cost is a concern, use a strong model for a small sample to validate that a cheaper model agrees:

```python
# Validate cheaper judge against expensive judge
expensive_scores = run_eval_with_model("gpt-4o", test_cases)
cheap_scores = run_eval_with_model("gpt-4o-mini", test_cases)

correlation = np.corrcoef(expensive_scores, cheap_scores)[0, 1]
print(f"Correlation between judges: {correlation:.3f}")
# If correlation > 0.9, the cheaper model is an acceptable judge
```

### 3. Contaminating Input with Prompt Template

**The problem**: Including the system prompt and few-shot examples in the `input` field of your test case. This gives the LLM judge extra context that real users would not have, inflating metrics.

**The fix**: The `input` field should contain only what the user typed. Keep system prompts separate:

```python
# WRONG
test_case = LLMTestCase(
    input="System: You are a helpful assistant.\nUser: What is RAG?",
    actual_output="..."
)

# CORRECT
test_case = LLMTestCase(
    input="What is RAG?",  # Just the user's question
    actual_output="..."
)
```

### 4. Confusing HallucinationMetric with FaithfulnessMetric

**The problem**: Using HallucinationMetric when you mean FaithfulnessMetric, or vice versa. They measure different things.

**The fix**: Understand the distinction:
- **FaithfulnessMetric**: Does the response contain only information supported by the retrieval context? (Higher is better)
- **HallucinationMetric**: Does the response contradict or fabricate information not in the provided context/ground truth? (Lower is better)

For RAG evaluation, FaithfulnessMetric is almost always what you want. HallucinationMetric is typically used when you have a ground truth context (not retrieval context).

### 5. Not Evaluating Retriever and Generator Separately

**The problem**: Only evaluating the final output, which tells you the system is underperforming but not *why*. Is the retriever failing to find relevant context? Or is the generator hallucinating despite having good context?

**The fix**: Use component-level metrics:

```python
# Retriever metrics (evaluate retrieval quality)
ContextualPrecisionMetric()   # Is the retrieved context precise?
ContextualRecallMetric()      # Does the context cover what's needed?
ContextualRelevancyMetric()   # Is the context relevant?

# Generator metrics (evaluate generation quality given context)
FaithfulnessMetric()          # Does the response stick to context?
AnswerRelevancyMetric()       # Does the response answer the question?
```

If retriever metrics are good but generator metrics are bad, focus on the generation prompt and LLM. If retriever metrics are bad, focus on embedding model, chunking, and search strategy.

### 6. Over-Relying on a Single Metric

**The problem**: Optimizing exclusively for one metric (e.g., Answer Relevancy) while ignoring others. This leads to Goodhart's Law: the metric becomes the target and stops being a good measure.

**The fix**: Always use a balanced set of metrics. A response that scores 1.0 on Answer Relevancy but 0.3 on Faithfulness is worse than one that scores 0.8 on both. Consider a composite score:

```python
def composite_score(metrics_results):
    weights = {
        "Faithfulness": 0.30,
        "Answer Relevancy": 0.25,
        "Contextual Precision": 0.20,
        "Contextual Recall": 0.15,
        "Bias": 0.10,  # Note: inverted -- lower bias is better
    }
    total = 0
    for name, weight in weights.items():
        score = metrics_results[name]
        if name in ["Bias", "Toxicity"]:
            score = 1 - score  # Invert so higher is better
        total += score * weight
    return total
```

### 7. Not Tracking Metrics Over Time

**The problem**: Running evaluation once, getting good scores, and never evaluating again. RAG systems degrade over time as knowledge bases change, user behavior shifts, and model updates are deployed.

**The fix**: Run evaluation on a schedule (daily or weekly) and track trends:

```python
# Weekly evaluation cron job
import datetime

def weekly_eval():
    results = run_evaluation(production_sample)
    metrics = {name: score for name, score in results}
    metrics["timestamp"] = datetime.datetime.utcnow().isoformat()

    # Append to time series
    append_to_metrics_db(metrics)

    # Check for degradation
    last_week = get_metrics_from_db(days_ago=7)
    for name, current in metrics.items():
        if name == "timestamp":
            continue
        previous = last_week.get(name, current)
        if abs(current - previous) > 0.1:
            send_alert(f"Metric {name} changed significantly: {previous:.3f} -> {current:.3f}")
```

### 8. Using Production Data Without Sampling

**The problem**: Evaluating every single production interaction, which is prohibitively expensive (LLM judge calls cost money) and slow.

**The fix**: Use stratified sampling (covered in the Production Monitoring section above). Aim for 1-5% of total traffic, with higher rates for edge cases, new topics, and suspicious inputs.

### 9. Not Validating Your Evaluation Metrics (Meta-Evaluation)

**The problem**: Trusting that your evaluation metrics accurately capture quality without verifying. What if your FaithfulnessMetric gives high scores to hallucinated responses?

**The fix**: Periodically validate metrics against human judgment:

```python
# Meta-evaluation: compare metric scores to human ratings
human_ratings = load_human_ratings("human_eval_2024.csv")
# Each row: (input, actual_output, human_score_0_to_1)

metric = FaithfulnessMetric()
metric_scores = []
human_scores = []

for row in human_ratings:
    test_case = LLMTestCase(
        input=row["input"],
        actual_output=row["actual_output"],
        retrieval_context=row["context"]
    )
    metric.measure(test_case)
    metric_scores.append(metric.score)
    human_scores.append(row["human_score"])

# Calculate correlation
from scipy.stats import spearmanr
correlation, p_value = spearmanr(metric_scores, human_scores)
print(f"Spearman correlation with human judgment: {correlation:.3f} (p={p_value:.4f})")
# Good: > 0.7, Acceptable: > 0.5, Poor: < 0.5
```

### 10. Ignoring Cost of Evaluation

**The problem**: Not tracking how much evaluation itself costs. LLM-as-a-judge calls can add up quickly, especially with multiple metrics across large test sets.

**The fix**: Budget for evaluation cost and optimize:

```python
# Estimate evaluation cost
def estimate_eval_cost(num_test_cases, num_metrics, avg_tokens_per_judge_call=2000):
    # GPT-4o pricing (approximate)
    input_cost_per_1k = 0.0025
    output_cost_per_1k = 0.01

    total_calls = num_test_cases * num_metrics
    total_input_tokens = total_calls * avg_tokens_per_judge_call
    total_output_tokens = total_calls * 500  # Typically shorter

    cost = (total_input_tokens / 1000 * input_cost_per_1k +
            total_output_tokens / 1000 * output_cost_per_1k)
    return cost

cost = estimate_eval_cost(num_test_cases=200, num_metrics=5)
print(f"Estimated evaluation cost: ${cost:.2f}")
# Typical: $1-5 for 200 test cases with 5 metrics
```

---

## Streaming Output Evaluation

### The Challenge: Evaluating Token-by-Token Output

Modern LLM applications increasingly use streaming responses, where tokens arrive
one-by-one and are rendered to the user in real time. This creates a fundamental
tension with evaluation: **evaluation metrics need the complete response**, but
streaming delivers it incrementally.

```
STREAMING TIMELINE:

  Time -->
  t0    t1    t2    t3    t4    ...   tN
  |     |     |     |     |           |
  "The" "cap" "ital" "of"  "Fra" ... "[END]"

  User sees tokens immediately as they arrive.
  But evaluation needs the FULL response: "The capital of France is Paris."

  Challenge: When and how do you evaluate?
```

### Approach 1: Buffer the Complete Response, Then Evaluate Post-Stream

The simplest and most reliable approach. Collect all tokens into a buffer, and once
the stream completes, evaluate the full response just like a non-streaming response.

```python
import asyncio
from openai import OpenAI
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

client = OpenAI()

async def stream_and_evaluate(query: str, contexts: list[str]) -> dict:
    """
    Stream the response to the user while buffering for evaluation.
    """
    # Step 1: Stream the response, buffering tokens
    buffer = []
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Answer based on this context: {contexts}"},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            buffer.append(token)
            print(token, end="", flush=True)  # Stream to user

    complete_response = "".join(buffer)
    print()  # Newline after stream completes

    # Step 2: Evaluate the complete response
    test_case = LLMTestCase(
        input=query,
        actual_output=complete_response,
        retrieval_context=contexts,
    )

    metric = FaithfulnessMetric(threshold=0.7)
    metric.measure(test_case)

    return {
        "response": complete_response,
        "faithfulness_score": metric.score,
        "faithfulness_reason": metric.reason,
    }
```

**Advantages**: Simple, reliable, uses existing evaluation infrastructure unchanged.

**Disadvantages**: Evaluation happens only after the user has already seen the full response.
No real-time quality signal during streaming.

### Approach 2: Evaluate Intermediate Chunks (Partial Faithfulness)

For long-form streaming responses, you can evaluate chunks as they accumulate. This
provides earlier signal but at higher cost and with caveats about partial-text evaluation.

```python
async def stream_with_periodic_eval(query, contexts, eval_interval=100):
    """
    Evaluate the response periodically as tokens accumulate.
    Useful for detecting early hallucination in long-form responses.
    """
    buffer = []
    partial_scores = []
    metric = FaithfulnessMetric(threshold=0.7)

    for chunk in stream_response(query, contexts):
        buffer.append(chunk)

        # Every eval_interval tokens, run a partial evaluation
        if len(buffer) % eval_interval == 0:
            partial_response = "".join(buffer)
            test_case = LLMTestCase(
                input=query,
                actual_output=partial_response,
                retrieval_context=contexts,
            )
            metric.measure(test_case)
            partial_scores.append({
                "tokens_so_far": len(buffer),
                "score": metric.score,
            })

            # Early termination if quality drops too low
            if metric.score < 0.3:
                return {"status": "aborted", "reason": "Faithfulness too low"}

    return {
        "response": "".join(buffer),
        "partial_scores": partial_scores,
        "final_score": partial_scores[-1]["score"] if partial_scores else None,
    }
```

**Advantages**: Provides early signal; can abort a bad response mid-stream.

**Disadvantages**: Expensive (multiple LLM judge calls per response); partial text may
score differently than the complete response.

### Approach 3: Post-Hoc Evaluation on Logged Complete Responses

The most practical approach for production systems. Log every complete response
asynchronously, then evaluate in batch offline.

```python
import logging
from datetime import datetime

logger = logging.getLogger("rag_eval")

def log_streaming_response(query, contexts, response_tokens):
    """Log the complete streamed response for later evaluation."""
    complete_response = "".join(response_tokens)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "contexts": contexts,
        "response": complete_response,
        "token_count": len(response_tokens),
    }
    logger.info("streaming_response", extra=record)
    # Write to your data store (S3, BigQuery, etc.)
    write_to_eval_queue(record)

# Later, in a batch evaluation job:
def batch_evaluate_logged_responses(log_records, sample_rate=0.05):
    """Evaluate a sample of logged streaming responses."""
    sampled = [r for r in log_records if hash(r["query"]) % 100 < sample_rate * 100]
    test_cases = [
        LLMTestCase(
            input=r["query"],
            actual_output=r["response"],
            retrieval_context=r["contexts"],
        )
        for r in sampled
    ]
    results = evaluate(test_cases=test_cases, metrics=[FaithfulnessMetric()])
    return results
```

### Current Framework Support

Neither DeepEval nor RAGAS directly supports evaluating streaming responses. Both
frameworks expect a complete response string as input:

- **DeepEval**: `LLMTestCase.actual_output` must be a complete string
- **RAGAS**: `SingleTurnSample.response` must be a complete string

To evaluate streaming outputs, you must always collect the full response first.

### Best Practice Summary

```
STREAMING EVALUATION DECISION:

  Real-time quality gate needed?
  |
  +-- NO  --> Log complete responses, evaluate asynchronously (Approach 3)
  |           This is the recommended default for production systems.
  |
  +-- YES --> Buffer and evaluate post-stream (Approach 1)
              Use for: development, testing, high-stakes deployments

  Long-form responses with early abort needed?
  |
  +-- YES --> Periodic partial evaluation (Approach 2)
              Use sparingly: expensive and scores may be unreliable on partial text
```

---

## The Future of RAG Evaluation

### Emerging Approaches: Reward Models as Judges

Instead of using general-purpose LLMs as judges, fine-tuned reward models are emerging as cheaper and more consistent evaluators:

- **Trained on human preference data**: Reward models learn what "good" looks like from human annotations
- **Faster and cheaper**: Smaller models, no need for expensive API calls
- **More consistent**: Deterministic behavior eliminates run-to-run variation
- **Domain-specific**: Can be fine-tuned for specific evaluation criteria

The trade-off is that reward models require training data (human judgments) to build, whereas LLM-as-a-judge works out of the box.

### Multi-Modal RAG Evaluation

As RAG systems expand beyond text to include images, audio, video, tables, and structured
data, evaluation must follow. Multi-modal RAG pipelines retrieve and generate across
modalities -- a product search returning images alongside text descriptions, a medical
system referencing X-ray images, or a document understanding pipeline extracting data
from charts and tables.

#### What Is Multi-Modal RAG?

Multi-modal RAG extends the retrieval-augmented generation paradigm across data types:

```
MULTI-MODAL RAG PIPELINE:

  User Query: "Show me red running shoes under $100"
       |
       v
  +-- Text Retriever --> product descriptions, reviews
  |
  +-- Image Retriever --> product photos, lifestyle images
  |
  +-- Table Retriever --> pricing tables, comparison charts
       |
       v
  Multi-Modal Generator (e.g., GPT-4o, Gemini)
       |
       v
  Response: text + images + structured data
```

The evaluation challenge multiplies: you need to assess not only text quality but also
image retrieval relevance, cross-modal coherence, and whether the visual and textual
elements tell a consistent story.

#### DeepEval Multimodal Metrics

DeepEval provides dedicated metrics for evaluating multimodal LLM outputs through
the `MLLMTestCase` (Multimodal LLM Test Case). These metrics use vision-capable
models like GPT-4o as judges.

**Available multimodal metrics:**

| Metric | What It Measures |
|--------|-----------------|
| `TextToImageMetric` | Whether a generated image matches the text description |
| `ImageEditingMetric` | Quality of image edits based on instructions |
| `ImageCoherenceMetric` | Whether an image is coherent and non-corrupted |
| `ImageHelpfulnessMetric` | Whether the image is helpful in context of the query |
| `ImageReferenceMetric` | How closely a generated image matches a reference image |

**Creating a multimodal test case:**

```python
from deepeval.test_case import MLLMTestCase, MLLMImage

test_case = MLLMTestCase(
    input=["Describe this product image"],
    actual_output=["This is a pair of red Nike running shoes with a white sole."],
    retrieval_context=["Nike Air Max 90 in red colorway, retail price $89.99"],
    # Images can be URLs or local file paths
    input_image=[MLLMImage(url="https://example.com/product_query.jpg")],
    actual_output_image=[MLLMImage(url="https://example.com/generated_shoe.jpg")],
    expected_output_image=[MLLMImage(url="https://example.com/reference_shoe.jpg")],
)
```

**Evaluating image helpfulness:**

```python
from deepeval.metrics import ImageHelpfulnessMetric
from deepeval import evaluate

metric = ImageHelpfulnessMetric(
    threshold=0.5,
    model="gpt-4o",  # Must be a vision-capable model
)

evaluate(test_cases=[test_case], metrics=[metric])
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
```

**Evaluating image coherence:**

```python
from deepeval.metrics import ImageCoherenceMetric

coherence_metric = ImageCoherenceMetric(
    threshold=0.5,
    model="gpt-4o",
)

coherence_metric.measure(test_case)
# Checks: Is the image well-formed? Does it contain artifacts?
# Is it a reasonable image (not noise, not corrupted)?
```

**Evaluating against a reference image:**

```python
from deepeval.metrics import ImageReferenceMetric

reference_metric = ImageReferenceMetric(
    threshold=0.5,
    model="gpt-4o",
)

reference_metric.measure(test_case)
# Compares generated image against expected_output_image
# Useful for image generation/editing tasks
```

#### RAGAS Multimodal Metrics

RAGAS has introduced multimodal evaluation capabilities for RAG systems that process
both text and images. These extend the core faithfulness and relevance concepts to
cross-modal settings.

**Key RAGAS multimodal metrics:**

| Metric | What It Measures |
|--------|-----------------|
| `MultiModalFaithfulness` | Whether text+image response is faithful to multi-modal context |
| `MultiModalRelevance` | Whether the multi-modal response is relevant to the query |

```python
from ragas.metrics import MultiModalFaithfulness, MultiModalRelevance
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate

# Create a sample with both text and image contexts
sample = SingleTurnSample(
    user_input="What does the quarterly revenue chart show?",
    response="The chart shows Q3 revenue increased 15% year-over-year to $2.3B.",
    retrieved_contexts=[
        "Q3 2024 financial results: Revenue reached $2.3 billion, a 15% increase.",
        # Image contexts can be referenced as descriptions or base64-encoded
        "[Chart: Bar graph showing quarterly revenue from Q1-Q4, Q3 bar at $2.3B]",
    ],
    reference="Q3 revenue was $2.3 billion, representing 15% YoY growth.",
)

dataset = EvaluationDataset(samples=[sample])

results = evaluate(
    dataset=dataset,
    metrics=[MultiModalFaithfulness(), MultiModalRelevance()],
)
print(results)
```

#### Practical Challenges in Multi-Modal Evaluation

```
MULTI-MODAL EVALUATION CHALLENGES:

  1. IMAGE RETRIEVAL QUALITY
     How do you measure if the right images were retrieved?
     - Text-to-image similarity (CLIP score)
     - Image-to-image similarity (perceptual hashing, SSIM)
     - Human judgment (still the gold standard for visual relevance)

  2. CROSS-MODAL FAITHFULNESS
     Does the text description match what the image shows?
     - A text saying "red shoes" but an image of blue shoes = cross-modal hallucination
     - Requires vision-capable judge models (GPT-4o, Gemini, Claude)

  3. TABLE AND CHART UNDERSTANDING
     Can the system correctly extract data from visual tables?
     - Numeric precision matters (extracting "$2.3B" vs "$23B" is critical)
     - Structural understanding (rows, columns, headers)

  4. EVALUATION MODEL LIMITATIONS
     Vision-capable LLMs as judges have their own visual understanding limits
     - Fine-grained visual details may be missed
     - Spatial reasoning can be unreliable
     - OCR accuracy varies across fonts and layouts
```

#### When Multi-Modal Evaluation Matters

```
USE MULTI-MODAL EVALUATION WHEN:

  +-- Product catalogs: retrieving and describing product images
  +-- Medical imaging: referencing X-rays, MRIs, pathology slides
  +-- Document understanding: extracting info from PDFs with charts/tables
  +-- E-commerce: visual search, outfit recommendation, product comparison
  +-- Education: explaining diagrams, maps, scientific figures
  +-- Real estate: matching property descriptions to listing photos
  +-- Insurance: assessing damage from claim photos

USE TEXT-ONLY EVALUATION WHEN:
  +-- Your RAG system only handles text documents
  +-- Images are decorative, not informational
  +-- You do not have vision-capable judge models available
```

### Evaluation Benchmarks

Standardized benchmarks help compare different RAG systems objectively:

| Benchmark | Focus | Description |
|-----------|-------|-------------|
| **MTEB** | Embeddings | Massive Text Embedding Benchmark; evaluates embedding quality across 56 datasets |
| **BEIR** | Retrieval | Heterogeneous benchmark for information retrieval across 18 datasets |
| **KILT** | Knowledge-Intensive | Knowledge Intensive Language Tasks; evaluates fact verification, QA, dialogue |
| **ARES** | RAG | Automated RAG Evaluation System; focuses on context relevance, faithfulness, answer relevance |
| **RGB** | RAG | RAG Benchmark; tests robustness to noise, negative rejection, information integration, counterfactual |

### The Convergence of Evaluation and Fine-Tuning

A significant trend is the convergence of evaluation and training signals:

1. **RLHF/DPO**: The same preference data used for evaluation (which response is better?) can train models via reinforcement learning from human feedback or direct preference optimization
2. **Constitutional AI**: Evaluation criteria (principles) are used both for evaluation and for training self-correcting models
3. **Evaluation-guided generation**: At inference time, evaluation metrics can guide decoding (rejecting low-scoring candidate responses)
4. **Synthetic data from evaluation**: Evaluation failures become training data -- interactions that score poorly on faithfulness become negative examples for fine-tuning

This convergence means that building a strong evaluation pipeline has compounding returns: it not only measures quality but actively improves it.

### The Trend Toward Standardization

The RAG evaluation space is rapidly maturing:
- **MCP** standardizes tool interaction, enabling standardized tool-use evaluation
- **OpenTelemetry for LLMs** standardizes tracing, enabling cross-framework evaluation
- **Metric definitions** are converging across frameworks (faithfulness, relevancy, precision, recall)
- **Evaluation protocols** are being formalized in academic literature

The future likely holds a standard evaluation protocol that all RAG systems can be compared against, similar to how BLEU/ROUGE standardized NLG evaluation (but with much more sophistication).

---

## Key Takeaways

1. **Custom metrics are essential** for domain-specific evaluation. Use DeepEval's `BaseMetric` for programmatic control, G-Eval for flexible LLM-judged criteria, and RAGAS's `MetricWithLLM`/`MetricWithEmbeddings` for research-oriented custom metrics.

2. **Synthetic data solves the cold-start problem**. Use DeepEval's Synthesizer or RAGAS's TestsetGenerator to bootstrap evaluation datasets, but always validate with human review and augment with production data over time.

3. **CI/CD integration is non-negotiable** for production RAG systems. Use `deepeval test run` in GitHub Actions to catch regressions before deployment. Separate fast (deterministic) and slow (LLM-based) test suites.

4. **Production monitoring completes the evaluation loop**. Offline evaluation catches regressions; online monitoring catches distribution shift, new failure patterns, and edge cases your test set did not cover.

5. **Evaluation-driven development** uses metric scores to diagnose specific issues and guide targeted improvements. Each metric maps to specific hyperparameters and system components.

6. **No single framework does everything**. DeepEval leads in agentic/MCP evaluation and CI/CD integration. RAGAS excels at research-oriented evaluation and synthetic data. LangSmith/Phoenix lead in production tracing. Use multiple frameworks where appropriate.

7. **Avoid the top 10 pitfalls**: insufficient test cases, weak judge models, contaminated inputs, confused metrics, siloed evaluation, single-metric focus, one-time evaluation, unsampled production data, unvalidated metrics, and ignored evaluation costs.

8. **The field is converging** toward standardized metrics, multi-modal evaluation, reward-model judges, and unified evaluation-training pipelines. Building strong evaluation infrastructure today pays compounding dividends as the field matures.

---

## Further Reading

- DeepEval Custom Metrics: https://deepeval.com/docs/metrics-custom
- DeepEval G-Eval: https://deepeval.com/docs/metrics-llm-evals
- DeepEval Synthesizer: https://deepeval.com/docs/synthesizer-introduction
- DeepEval CI/CD: https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd
- RAGAS Metrics: https://docs.ragas.io/en/stable/concepts/metrics/
- RAGAS TestsetGenerator: https://docs.ragas.io/en/stable/getstarted/testset_generation/
- G-Eval Paper: "NLG Evaluation using GPT-4 with Better Human Alignment"
- Evol-Instruct Paper: arXiv:2304.12244
- MTEB Benchmark: https://huggingface.co/spaces/mteb/leaderboard
- BEIR Benchmark: https://github.com/beir-cellar/beir

---

*Previous: [09 - Agentic RAG Evaluation](09_agentic_rag_evaluation.md)*

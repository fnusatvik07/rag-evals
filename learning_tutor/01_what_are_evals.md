# Chapter 1: What Are Evaluations?

## A Comprehensive Guide to AI/ML Evaluation Systems

---

## 1.1 Definition and Core Concepts

**Evaluations** (commonly called "evals") are systematic processes for measuring the quality,
correctness, safety, and utility of AI/ML system outputs. They answer a deceptively simple
question: *"Is my system working well enough?"*

More formally:

```
Evaluation = f(System Output, Criteria) -> Score/Judgment

Where:
  - System Output: the response, prediction, or artifact produced by the AI system
  - Criteria: the standard against which quality is measured
  - Score/Judgment: a quantitative metric or qualitative assessment
```

In the context of **Retrieval-Augmented Generation (RAG)** systems specifically, evaluations
assess multiple interconnected components:

```
+-------------------+      +-------------------+      +-------------------+
|   Query           | ---> |   Retrieval       | ---> |   Generation      |
|   Understanding   |      |   Quality         |      |   Quality         |
+-------------------+      +-------------------+      +-------------------+
        |                          |                          |
        v                          v                          v
  - Intent captured?         - Relevant docs?          - Accurate answer?
  - Entities extracted?      - Sufficient context?     - Well-grounded?
  - Ambiguity handled?      - No noise?               - No hallucination?
```

### What Evaluations Are NOT

| Evaluations ARE                              | Evaluations ARE NOT                          |
|----------------------------------------------|----------------------------------------------|
| Systematic measurement of quality            | Ad-hoc manual testing                        |
| Repeatable and automated processes           | One-time checks before launch                |
| Multi-dimensional quality assessment         | A single accuracy number                     |
| Continuous monitoring infrastructure         | Something you do once and forget             |
| Evidence-based decision making tools         | Gut feeling about "it seems to work"         |
| A feedback loop for improvement              | A gate that blocks deployment                |

---

## 1.2 Why Evaluations Matter

### 1.2.1 Safety

AI systems deployed without rigorous evaluation can cause real harm:

- **Medical AI**: A system that hallucinates drug interactions could kill patients.
- **Legal AI**: Fabricated case citations waste court resources and can lead to sanctions.
- **Financial AI**: Incorrect financial advice can cause significant monetary loss.
- **Autonomous systems**: Wrong decisions in self-driving or robotics endanger lives.

**The safety equation:**

```
Risk = P(failure) x Impact(failure)

Without evals: P(failure) is UNKNOWN --> Risk is UNBOUNDED
With evals:    P(failure) is MEASURED --> Risk is MANAGEABLE
```

### 1.2.2 Reliability

Users need consistent, predictable behavior:

```
Without Evals:                    With Evals:
+---------------------------+     +---------------------------+
| Query: "What is Python?"  |     | Query: "What is Python?"  |
| Run 1: Great answer       |     | Run 1: Great answer  [OK] |
| Run 2: Mediocre answer    |     | Run 2: Great answer  [OK] |
| Run 3: Hallucination      |     | Run 3: Good answer   [OK] |
| Run 4: Great answer       |     | Run 4: Great answer  [OK] |
| Run 5: Off-topic          |     | Run 5: Great answer  [OK] |
|                           |     |                           |
| Reliability: UNKNOWN      |     | Reliability: 98.5%        |
+---------------------------+     +---------------------------+
```

### 1.2.3 Cost

Evaluations directly impact your bottom line:

| Cost Category          | Without Evals                          | With Evals                           |
|------------------------|----------------------------------------|--------------------------------------|
| API costs              | Unknown waste on bad prompts           | Optimized prompt = fewer tokens      |
| Human review           | 100% manual review needed              | Only flagged cases need review       |
| Customer support       | High ticket volume from errors         | Reduced error-related tickets        |
| Incident response      | Reactive firefighting                  | Proactive issue detection            |
| Model selection        | Guess which model to use               | Data-driven model selection          |
| Reputational damage    | Unquantified brand risk                | Measured and mitigated               |

**ROI of evaluation investment:**

```
Cost of building eval suite:           ~$10,000-50,000 (one-time)
Cost of maintaining eval suite:        ~$2,000-5,000/month
Cost of a single public hallucination: $50,000-5,000,000+ (legal, PR, lost customers)

ROI = (Prevented Losses - Eval Costs) / Eval Costs
    = ($500,000 - $80,000) / $80,000
    = 525% (conservative estimate)
```

### 1.2.4 Trust

Trust is the currency of AI adoption:

```
Trust Lifecycle:

  Initial Trust        Validated Trust        Deep Trust
  (marketing)   --->   (evals prove it) --->  (track record)
       |                     |                      |
  "It should work"    "We measured it"       "It consistently works"
       |                     |                      |
  FRAGILE               RESILIENT              ANTIFRAGILE
```

### 1.2.5 Iteration Speed

Without evals, every change is a leap of faith. With evals, you have a safety net:

```
Development WITHOUT Evals:
  Change prompt --> Deploy --> Wait for complaints --> Debug --> Repeat
  Cycle time: Days to weeks

Development WITH Evals:
  Change prompt --> Run evals --> See impact --> Adjust --> Deploy with confidence
  Cycle time: Minutes to hours
```

---

## 1.3 The Evaluation Lifecycle

The evaluation lifecycle mirrors — and integrates with — the AI development lifecycle:

```
+------------------------------------------------------------------+
|                    THE EVALUATION LIFECYCLE                        |
|                                                                    |
|   +----------+     +----------+     +--------+     +---------+    |
|   |          |     |          |     |        |     |         |    |
|   | DEVELOP  |---->| EVALUATE |---->| DEPLOY |---->| MONITOR |    |
|   |          |     |          |     |        |     |         |    |
|   +----------+     +----------+     +--------+     +---------+    |
|        ^                                                |         |
|        |              +----------+                      |         |
|        |              |          |                      |         |
|        +--------------| IMPROVE  |<---------------------+         |
|                       |          |                                 |
|                       +----------+                                |
+------------------------------------------------------------------+
```

### Phase 1: Develop

During development, evaluations help you:
- Choose the right model (GPT-4 vs Claude vs Llama vs Mistral)
- Optimize prompts (system prompt, few-shot examples, chain-of-thought)
- Tune retrieval parameters (chunk size, top-k, similarity threshold)
- Validate pipeline architecture (single-step vs multi-step, agents vs chains)

```python
# Example: Development-phase evaluation loop
for prompt_variant in prompt_candidates:
    for model in ["gpt-4o", "claude-sonnet", "llama-3.1-70b"]:
        results = run_eval_suite(
            model=model,
            prompt=prompt_variant,
            test_cases=dev_dataset,  # 50-100 representative cases
            metrics=["accuracy", "relevance", "faithfulness"]
        )
        log_results(results)

best_config = select_best(all_results, primary_metric="faithfulness")
```

### Phase 2: Evaluate

Formal evaluation before deployment:
- Run full test suite (hundreds to thousands of test cases)
- Compare against baselines and previous versions
- Check for regressions across all dimensions
- Validate edge cases, adversarial inputs, and safety scenarios
- Generate evaluation reports for stakeholders

```python
# Example: Pre-deployment evaluation gate
eval_results = run_full_eval_suite(
    model=production_config,
    test_cases=golden_dataset,     # 500-2000 curated cases
    metrics=ALL_METRICS,
    compare_to=current_production  # regression detection
)

# Deployment gate
if eval_results.passes_all_thresholds():
    approve_for_deployment()
elif eval_results.has_regressions():
    block_deployment(reason=eval_results.regressions)
else:
    flag_for_human_review(eval_results)
```

### Phase 3: Deploy

Evaluation-informed deployment strategies:
- Canary deployments: route 5% of traffic to new version, compare metrics
- A/B testing: split traffic and measure user-facing metrics
- Shadow mode: run new version in parallel, compare outputs without serving

### Phase 4: Monitor

Production monitoring evaluations:
- Real-time quality scoring on sampled traffic
- Drift detection (input distribution, output quality)
- Latency and cost tracking
- User feedback collection and analysis
- Automated alerting when metrics degrade

```python
# Example: Production monitoring
@monitor(sample_rate=0.05)  # Evaluate 5% of production traffic
def handle_query(query):
    response = rag_pipeline(query)
    
    # Async evaluation (non-blocking)
    evaluate_async(
        query=query,
        response=response,
        metrics=["faithfulness", "relevance", "toxicity"],
        alert_if=lambda scores: scores["faithfulness"] < 0.7
    )
    
    return response
```

### Phase 5: Improve

Using evaluation results to improve:
- Identify systematic failure patterns
- Curate new training/fine-tuning data from failures
- Update prompts based on error analysis
- Add new test cases from production failures
- Refine retrieval strategy based on eval insights

---

## 1.4 Types of Evaluation: Offline vs Online

### Offline Evaluation (Pre-Deployment)

```
+-----------------------------------------------+
|            OFFLINE EVALUATION                   |
|                                                 |
|  Input: Curated test dataset                    |
|  Environment: Development / CI/CD               |
|  Frequency: Per code change or scheduled        |
|  Latency tolerance: Minutes to hours            |
|  Cost model: Fixed (dataset size x eval cost)   |
+-----------------------------------------------+
```

| Aspect               | Details                                                        |
|-----------------------|----------------------------------------------------------------|
| **When**              | Before deployment, during development, in CI/CD pipelines      |
| **Data**              | Curated test sets, golden datasets, adversarial examples       |
| **Metrics**           | Task-specific: accuracy, faithfulness, relevance, etc.         |
| **Advantages**        | Controlled, repeatable, comprehensive, catches regressions     |
| **Disadvantages**     | May not represent real traffic, static, can become stale       |
| **Typical tools**     | RAGAS, DeepEval, LangSmith, Braintrust, custom frameworks     |

### Online Evaluation (Production Monitoring)

```
+-----------------------------------------------+
|            ONLINE EVALUATION                    |
|                                                 |
|  Input: Real production traffic (sampled)       |
|  Environment: Production                        |
|  Frequency: Continuous                          |
|  Latency tolerance: Seconds (async)             |
|  Cost model: Variable (traffic x sample rate)   |
+-----------------------------------------------+
```

| Aspect               | Details                                                        |
|-----------------------|----------------------------------------------------------------|
| **When**              | Continuously in production                                     |
| **Data**              | Real user queries and system responses                         |
| **Metrics**           | Quality + operational: latency, cost, user satisfaction        |
| **Advantages**        | Reflects real usage, catches novel failures, measures UX       |
| **Disadvantages**     | Noisy, expensive at scale, evaluation latency constraints      |
| **Typical tools**     | LangSmith, Langfuse, Arize, WhyLabs, Datadog LLM Monitoring   |

### Comparison Matrix

```
                        OFFLINE                     ONLINE
                   +----------------+          +----------------+
  Representativeness|     Medium     |          |     High       |
                   +----------------+          +----------------+
  Repeatability    |     High       |          |     Low        |
                   +----------------+          +----------------+
  Cost Control     |     High       |          |     Medium     |
                   +----------------+          +----------------+
  Speed of         |     Slow       |          |     Fast       |
  Feedback         | (batch)        |          | (real-time)    |
                   +----------------+          +----------------+
  Coverage         |     Defined    |          |     Emergent   |
                   | by dataset     |          | from traffic   |
                   +----------------+          +----------------+
```

---

## 1.5 Evaluation Components

Every evaluation system consists of five core components:

### 1.5.1 Test Cases

A test case is the atomic unit of evaluation:

```python
# Structure of a test case
test_case = {
    "id": "tc_001",
    "input": {
        "query": "What is the capital of France?",
        "context": ["France is a country in Western Europe. Paris is its capital."]
    },
    "expected_output": "Paris",          # Ground truth (optional for some evals)
    "metadata": {
        "category": "factual_qa",
        "difficulty": "easy",
        "source": "geography_dataset_v2",
        "tags": ["geography", "europe", "capitals"]
    }
}
```

**Types of test cases:**

| Type            | Description                                   | Example                                      |
|-----------------|-----------------------------------------------|----------------------------------------------|
| Golden          | Curated with verified correct answers          | Q: "Capital of France?" A: "Paris"           |
| Adversarial     | Designed to trick or stress-test the system    | Q: "Capital of France before 508 AD?"        |
| Edge case       | Unusual inputs at system boundaries            | Q: "" (empty), Q: very long query (10k chars)|
| Regression      | Previously failed cases now expected to pass   | Specific bugs that were fixed                |
| Synthetic       | Generated programmatically or by LLMs          | LLM-generated Q&A pairs from documents       |

### 1.5.2 Metrics

Metrics quantify different dimensions of quality:

```
                    METRIC TAXONOMY
                    
    +------------------+------------------+
    |   DETERMINISTIC  |   MODEL-BASED    |
    +------------------+------------------+
    | Exact match      | Faithfulness     |
    | BLEU / ROUGE     | Relevance        |
    | F1 (token)       | Coherence        |
    | Cosine similarity| Helpfulness      |
    | Regex match      | Safety           |
    | JSON validity    | Completeness     |
    | Latency (ms)     | Hallucination    |
    | Cost ($)         | Toxicity         |
    +------------------+------------------+
```

### 1.5.3 Thresholds

Thresholds define "good enough":

```python
# Example threshold configuration
thresholds = {
    "faithfulness": {
        "minimum": 0.85,       # Hard gate: below this = FAIL
        "target": 0.95,        # Aspiration: where we want to be
        "critical": 0.70       # Alert: immediate investigation needed
    },
    "relevance": {
        "minimum": 0.80,
        "target": 0.90,
        "critical": 0.60
    },
    "latency_p95_ms": {
        "minimum": 5000,       # Must respond in under 5 seconds
        "target": 2000,
        "critical": 10000
    },
    "cost_per_query_usd": {
        "minimum": 0.10,
        "target": 0.03,
        "critical": 0.50
    }
}
```

### 1.5.4 Datasets

Evaluation datasets are collections of test cases organized for systematic evaluation:

```
DATASET STRUCTURE:

golden_dataset/
  |-- v1.0/
  |     |-- metadata.json          # Version info, creation date, stats
  |     |-- factual_qa.jsonl       # Category: factual questions
  |     |-- summarization.jsonl    # Category: summarization tasks
  |     |-- adversarial.jsonl      # Category: adversarial inputs
  |     |-- safety.jsonl           # Category: safety-critical cases
  |     |-- README.md              # Dataset documentation
  |
  |-- v1.1/
        |-- ...                    # Updated version with new cases
```

**Dataset size recommendations:**

| Stage           | Recommended Size | Rationale                                    |
|-----------------|------------------|----------------------------------------------|
| Prototyping     | 20-50 cases      | Quick signal, fast iteration                 |
| Development     | 100-300 cases    | Reasonable coverage, catches major issues    |
| Pre-deployment  | 500-2000 cases   | Statistical significance, edge case coverage |
| Production      | 1000-5000+ cases | Comprehensive, domain-specific coverage      |

### 1.5.5 Judges

Judges are the entities that produce evaluation scores:

```
JUDGE TYPES:

1. DETERMINISTIC JUDGE          2. LLM JUDGE              3. HUMAN JUDGE
   (Code/Rules)                    (AI Model)                (Expert)
   
   input --> algorithm --> score   input --> LLM --> score   input --> person --> score
   
   Pros: Fast, cheap,             Pros: Handles nuance,    Pros: Gold standard,
         consistent,                    scales well,              understands context,
         explainable                    flexible                  catches subtlety
   
   Cons: Rigid, limited,          Cons: Costly, biased,    Cons: Expensive, slow,
         misses nuance                  inconsistent              inconsistent
```

---

## 1.6 Evaluation Terminology Glossary

### Core Terms

| Term                        | Definition                                                                                           |
|-----------------------------|------------------------------------------------------------------------------------------------------|
| **Ground Truth**            | The verified correct answer for a given input. The "right answer" against which outputs are compared.|
| **Golden Dataset**          | A curated, high-quality evaluation dataset with verified ground truth labels.                        |
| **Reference-Based Eval**    | Evaluation that compares system output against a known correct reference/ground truth.               |
| **Reference-Free Eval**     | Evaluation that assesses output quality without any reference answer (based on intrinsic quality).   |
| **LLM-as-a-Judge**          | Using a language model to evaluate the outputs of another language model.                            |
| **Human Evaluation**        | Using human annotators to assess system output quality.                                              |
| **Inter-Annotator Agreement (IAA)** | The degree to which multiple human annotators agree on their assessments.                    |
| **Rubric**                  | A detailed scoring guide that defines criteria for each score level.                                 |
| **Metric**                  | A quantitative measure of a specific quality dimension.                                              |
| **Benchmark**               | A standardized evaluation dataset and protocol for comparing systems.                                |
| **Baseline**                | A reference system or score against which improvements are measured.                                 |
| **Regression**              | A degradation in performance compared to a previous version.                                         |
| **Ablation Study**          | Systematic removal of components to measure their individual contribution.                           |

### RAG-Specific Terms

| Term                        | Definition                                                                                           |
|-----------------------------|------------------------------------------------------------------------------------------------------|
| **Faithfulness**            | Whether the generated answer is supported by the retrieved context (no hallucination).               |
| **Context Relevance**       | Whether the retrieved documents are relevant to the user's query.                                    |
| **Answer Relevance**        | Whether the generated answer actually addresses the user's question.                                 |
| **Context Recall**          | Proportion of the ground truth answer that can be attributed to retrieved context.                   |
| **Context Precision**       | Proportion of retrieved context that is actually relevant to answering the query.                    |
| **Groundedness**            | Degree to which generated claims are supported by source documents.                                  |
| **Hallucination**           | Generated content that is not supported by the provided context or factual reality.                  |
| **Chunk**                   | A segment of a source document used for retrieval.                                                   |
| **Retrieval Recall@K**      | Proportion of relevant documents found in the top K retrieved results.                               |

### Statistical Terms

| Term                        | Definition                                                                                           |
|-----------------------------|------------------------------------------------------------------------------------------------------|
| **Cohen's Kappa (kappa)**   | Measures agreement between two raters, accounting for chance. kappa = (p_o - p_e) / (1 - p_e)       |
| **Fleiss' Kappa**           | Extension of Cohen's Kappa to three or more raters.                                                  |
| **Confidence Interval**     | Range within which the true metric value likely falls (e.g., 95% CI).                                |
| **Statistical Significance**| Whether an observed difference is unlikely to be due to chance (p < 0.05 typically).                 |
| **Effect Size**             | Magnitude of the difference between two systems (e.g., Cohen's d).                                  |
| **Bootstrap Estimation**    | Resampling technique to estimate metric variance and confidence intervals.                           |

---

## 1.7 The Evaluation Pyramid

Like software testing, AI evaluation follows a layered pyramid structure. Each layer provides
different coverage, speed, and cost tradeoffs:

```
                          /\
                         /  \
                        / HE \        Human Evaluation
                       / EVAL \       - Expert review
                      /--------\      - User studies
                     /  SYSTEM  \     - A/B tests
                    /   TESTS    \    
                   /--------------\   System-Level Evals
                  /  INTEGRATION   \  - End-to-end pipeline tests
                 /    TESTS         \ - Multi-component interaction
                /--------------------\
               /     UNIT TESTS       \  Component-Level Evals
              /                        \ - Individual metric checks
             /__________________________\- Single function validation

  BOTTOM: Many tests, fast, cheap, automated
  TOP:    Few tests, slow, expensive, high-signal
```

### Layer 1: Unit Tests (Foundation)

```python
# Unit test examples for RAG components

def test_chunker_splits_correctly():
    """Test that the document chunker produces expected chunks."""
    doc = "Sentence one. Sentence two. Sentence three."
    chunks = chunker.split(doc, max_size=20)
    assert len(chunks) == 3
    assert all(len(c) <= 20 for c in chunks)

def test_embedding_dimensions():
    """Test that embeddings have correct dimensions."""
    embedding = embed("test query")
    assert len(embedding) == 1536  # OpenAI ada-002

def test_retriever_returns_k_results():
    """Test that retriever returns exactly k documents."""
    results = retriever.search("test query", k=5)
    assert len(results) == 5

def test_prompt_template_formats():
    """Test that prompt template handles all variables."""
    prompt = template.format(query="Q", context="C", history="H")
    assert "Q" in prompt and "C" in prompt and "H" in prompt
```

**Characteristics:** Fast (<1s each), cheap (no API calls), deterministic, run on every commit.

### Layer 2: Integration Tests

```python
# Integration test: retrieval + generation pipeline
def test_rag_pipeline_end_to_end():
    """Test that the full RAG pipeline produces relevant answers."""
    query = "What is photosynthesis?"
    response = rag_pipeline(query)
    
    assert response.answer is not None
    assert len(response.answer) > 50
    assert len(response.sources) > 0
    assert response.latency_ms < 5000

def test_retrieval_feeds_generation():
    """Test that retrieved context is actually used in generation."""
    query = "What is the Krebs cycle?"
    retrieval_result = retriever.search(query, k=3)
    generation_result = generator.generate(query, retrieval_result.documents)
    
    # Check that generation references retrieved content
    assert any(
        doc_keyword in generation_result.answer.lower()
        for doc in retrieval_result.documents
        for doc_keyword in extract_keywords(doc)
    )
```

**Characteristics:** Moderate speed (seconds), some API costs, tests component interactions.

### Layer 3: System Tests

```python
# System-level evaluation across a dataset
def test_system_faithfulness():
    """Evaluate faithfulness across golden dataset."""
    results = evaluate(
        pipeline=rag_pipeline,
        dataset=golden_dataset,
        metrics=[faithfulness, answer_relevance, context_precision]
    )
    
    assert results["faithfulness"].mean() >= 0.85
    assert results["answer_relevance"].mean() >= 0.80
    assert results["context_precision"].mean() >= 0.75
```

**Characteristics:** Slow (minutes to hours), expensive (many API calls), comprehensive coverage.

### Layer 4: Human Evaluation (Apex)

```
Human Evaluation Protocol:
1. Sample 100-200 responses from system tests
2. Distribute to 3+ expert annotators
3. Each annotator rates on rubric (1-5 scale):
   - Accuracy: Is the information correct?
   - Completeness: Does it fully answer the question?
   - Clarity: Is it well-written and understandable?
   - Safety: Is it free from harmful content?
4. Compute inter-annotator agreement (target: kappa > 0.7)
5. Aggregate scores and analyze disagreements
```

**Characteristics:** Very slow (days to weeks), very expensive, highest signal quality.

---

## 1.8 The Cost of NOT Evaluating

### Real-World Evaluation Failures

#### Case 1: ChatGPT Hallucinations in Legal Proceedings (2023)

**What happened:** A New York lawyer used ChatGPT to research case law. ChatGPT generated
six completely fabricated case citations with made-up judges, courts, and rulings. The lawyer
submitted these in a legal brief without verification.

**Consequence:**
- Lawyer sanctioned by federal judge
- $5,000 fine
- National media coverage
- Damage to the legal profession's trust in AI

**What evaluation would have caught:**
- Faithfulness evaluation: Are citations grounded in real sources?
- Factual verification: Do these cases exist in legal databases?
- Hallucination detection: Is the model generating unverifiable claims?

#### Case 2: Medical AI Giving Dangerous Advice

**What happened:** Multiple AI chatbots deployed in healthcare contexts have been documented
giving dangerous medical advice, including incorrect drug dosages, missed contraindications,
and failure to recommend emergency care for serious symptoms.

**Consequence:**
- Potential patient harm
- Regulatory scrutiny of AI in healthcare
- Erosion of trust in health AI tools

**What evaluation would have caught:**
- Safety evaluation with medical expert review
- Adversarial testing with known dangerous scenarios
- Ground truth comparison against medical guidelines
- Red-teaming for edge cases and rare conditions

#### Case 3: Customer Support Bot Offering Unauthorized Discounts

**What happened:** Air Canada's chatbot gave a customer incorrect information about
bereavement fare policies, promising a discount that did not exist. The customer relied
on this information and later sued.

**Consequence:**
- Air Canada was ordered to honor the chatbot's promise
- Financial loss from the discount
- Legal precedent that companies are liable for their chatbot's statements
- Negative publicity

**What evaluation would have caught:**
- Policy compliance evaluation: Does the output match company policy?
- Factual grounding: Is every claim traceable to an authoritative source?
- Boundary testing: What happens when the bot is asked about edge-case policies?

### The Financial Impact Framework

```
+-------------------------------------------------------------------+
|              COST OF EVALUATION FAILURES                            |
|                                                                     |
|   Direct Costs:                                                     |
|   +-- Legal liability:           $50K - $50M per incident          |
|   +-- Customer compensation:     $10K - $1M per incident           |
|   +-- Regulatory fines:          $100K - $100M+                    |
|   +-- Incident response:         $20K - $200K per incident         |
|                                                                     |
|   Indirect Costs:                                                   |
|   +-- Brand damage:              Hard to quantify (millions)        |
|   +-- Customer churn:            5-20% increase after public fail   |
|   +-- Employee morale:           Engineering team burnout           |
|   +-- Opportunity cost:          Time spent on damage control       |
|   +-- Regulatory burden:         Increased compliance requirements  |
|                                                                     |
|   Total potential cost per major failure: $100K - $100M+            |
|   Cost of comprehensive eval suite:      $50K - $500K/year         |
|                                                                     |
|   --> Evaluation is 100-1000x cheaper than failure                  |
+-------------------------------------------------------------------+
```

---

## 1.9 Evaluation Across Different Domains

### Customer Support

```
Key Metrics:
  - Resolution accuracy: Did it solve the problem?
  - Policy compliance: Did it follow company rules?
  - Tone appropriateness: Was it empathetic and professional?
  - Escalation accuracy: Did it correctly escalate when needed?
  - Response time: Was it fast enough?

Special Challenges:
  - Multi-turn conversation evaluation
  - Emotional intelligence assessment
  - Cultural sensitivity across markets
  - Integration with ticketing systems
```

### Legal

```
Key Metrics:
  - Citation accuracy: Are all referenced cases/statutes real and correct?
  - Jurisdictional relevance: Is the law applicable to the jurisdiction?
  - Completeness: Are all relevant precedents and statutes mentioned?
  - Recency: Is the law current (not overturned/amended)?
  - Disclaimer compliance: Are appropriate caveats included?

Special Challenges:
  - Jurisdiction-specific evaluation sets needed
  - Rapidly changing law (new cases, legislation)
  - Very high stakes (liberty, property, rights)
  - Professional responsibility obligations
```

### Medical

```
Key Metrics:
  - Clinical accuracy: Is the medical information correct?
  - Safety: Could following this advice cause harm?
  - Evidence level: Is advice based on peer-reviewed evidence?
  - Contraindication awareness: Are interactions/warnings included?
  - Scope awareness: Does it know when to refer to a doctor?

Special Challenges:
  - FDA/regulatory compliance
  - Patient safety is non-negotiable (zero tolerance for harmful advice)
  - Requires domain expert evaluation (expensive)
  - Liability and malpractice considerations
```

### Code Generation

```
Key Metrics:
  - Functional correctness: Does the code run and produce correct output?
  - Test pass rate: Does it pass the test suite?
  - Security: Are there vulnerabilities (SQL injection, XSS, etc.)?
  - Performance: Is it efficient (time/space complexity)?
  - Style compliance: Does it follow coding standards?
  - Maintainability: Is it readable and well-structured?

Special Challenges:
  - Need to actually execute generated code (sandboxing)
  - Multiple correct solutions possible
  - Language-specific evaluation needed
  - Security evaluation requires specialized tools
```

### Search / Information Retrieval

```
Key Metrics:
  - Precision@K: Proportion of top-K results that are relevant
  - Recall@K: Proportion of all relevant docs found in top K
  - NDCG (Normalized Discounted Cumulative Gain): Ranking quality
  - MRR (Mean Reciprocal Rank): Position of first relevant result
  - Click-through rate (online): User engagement with results

Special Challenges:
  - Relevance is subjective and query-dependent
  - Need large-scale relevance judgments
  - Position bias in user feedback
  - Cold start for new content
```

---

## 1.10 Guardrails vs Evaluations

Guardrails and evaluations are often confused or conflated, but they serve fundamentally
different purposes in the AI system lifecycle. Understanding the distinction -- and how
the two complement each other -- is essential for building robust AI systems.

### What Are Guardrails?

Guardrails are **runtime input/output filters** that intercept requests and responses in
real time. They act as safety nets that prevent harmful, non-compliant, or low-quality
content from reaching the user (or the model) on a per-request basis.

**Examples of guardrails:**
- **PII filter**: Detects and redacts personal information (SSNs, emails, phone numbers) before the response is sent
- **Toxicity blocker**: Rejects or rewrites responses containing hate speech, profanity, or harmful content
- **Topic filter**: Prevents the model from discussing off-limits subjects (e.g., competitors, politics)
- **Prompt injection detector**: Blocks adversarial inputs designed to manipulate the model
- **Output format enforcer**: Ensures the response matches the expected schema (JSON, XML, etc.)
- **Hallucination guardrail**: Checks claims against a knowledge base in real time and flags unsupported statements

### What Are Evaluations?

Evaluations are **offline quality measurement processes** that assess system performance
across a dataset. They answer questions like "How faithful is this system on average?"
or "Did this prompt change improve answer relevancy?"

**Examples of evaluations:**
- **Faithfulness scoring**: Measuring what percentage of generated claims are supported by retrieved context across 500 test cases
- **Relevancy measurement**: Scoring how well responses address user queries across a golden dataset
- **Regression detection**: Comparing metric scores before and after a system change
- **Bias auditing**: Measuring whether the system exhibits systematic bias across demographic groups

### Key Differences

| Dimension | Guardrails | Evaluations |
|-----------|------------|-------------|
| **Timing** | Runtime (per-request) | Offline (batch, scheduled, or CI/CD) |
| **Purpose** | Prevention -- stop bad output from reaching users | Measurement -- understand system quality |
| **Scope** | Per-request (individual interaction) | Dataset-level (aggregate over many cases) |
| **Speed** | Must be fast (< 100ms ideally) | Can be slow (minutes to hours) |
| **Failure mode** | Block or modify a single response | Produce a score or report |
| **Cost model** | Per-request overhead (latency + compute) | Per-run cost (bounded by dataset size) |
| **Action taken** | Reject, rewrite, redact, or escalate | Inform decisions, block deployments, guide improvement |
| **Feedback loop** | Immediate (user never sees bad output) | Delayed (results analyzed by engineers) |
| **Coverage** | Every request (or sampled subset) | Curated test set + sampled production data |
| **Determinism** | Often deterministic (rules, regex, classifiers) | May include LLM judges (non-deterministic) |

### How They Complement Each Other

Guardrails and evaluations work best together. Guardrails catch issues in real time on
individual requests, while evaluations catch systematic issues across the entire system.

```
GUARDRAILS + EVALUATIONS: DEFENSE IN DEPTH

  User Request
       |
       v
  +--[INPUT GUARDRAILS]--+     <-- Real-time: block prompt injection, PII in query
       |
       v
  RAG Pipeline (retrieve + generate)
       |
       v
  +--[OUTPUT GUARDRAILS]--+    <-- Real-time: block toxic output, enforce format
       |
       v
  Response to User
       |
       +--[LOG]--+
                 |
                 v
  +--[OFFLINE EVALUATION]--+   <-- Batch: measure faithfulness, relevancy, bias
                 |                    across sampled or full production logs
                 v
  Quality Report --> Improvement Decisions
```

**What guardrails catch that evaluations miss:**
- A single dangerous response that slips through before the next eval run
- Real-time prompt injection attempts
- PII leakage in a specific response

**What evaluations catch that guardrails miss:**
- Gradual quality degradation over time (drift)
- Systematic bias that appears only at aggregate level
- Retriever failures that produce subtly irrelevant context
- The 85th percentile of faithfulness dropping from 0.92 to 0.78

### When to Use Guardrails vs Evals vs Both

```
USE GUARDRAILS WHEN:
  +-- You need per-request protection (safety, PII, compliance)
  +-- Failures on individual requests are unacceptable
  +-- You need real-time intervention (block, rewrite, escalate)
  +-- The check is fast and deterministic (regex, classifier, rules)

USE EVALUATIONS WHEN:
  +-- You need aggregate quality measurement
  +-- You are comparing system versions or configurations
  +-- You need statistical confidence in quality claims
  +-- You are diagnosing systematic failure patterns
  +-- The assessment requires expensive LLM judges

USE BOTH WHEN:
  +-- You are deploying to production (always use both)
  +-- Safety and quality both matter (nearly always)
  +-- You want defense in depth (catch what the other misses)
```

---

## 1.11 Common Anti-Patterns in Evaluation

### Anti-Pattern 1: "Vibes-Based Evaluation"

```
BAD:  "I tried a few queries and it seems to work fine."
GOOD: "We ran 500 test cases across 8 categories and achieved 92% faithfulness."
```

**Why it's bad:** Human memory is selective. You remember the impressive demos and forget the
failures. Without systematic measurement, you have no idea how your system actually performs.

### Anti-Pattern 2: "Eval Once, Ship Forever"

```
BAD:  Evaluate at launch, never again.
GOOD: Continuous evaluation in CI/CD + production monitoring.
```

**Why it's bad:** Models change, data drifts, user behavior evolves. An evaluation that
passed 6 months ago tells you nothing about today's performance.

### Anti-Pattern 3: "One Metric to Rule Them All"

```
BAD:  "Our accuracy is 95%!"
GOOD: "Faithfulness: 92%, Relevance: 88%, Safety: 99.5%, Latency p95: 2.1s"
```

**Why it's bad:** A single metric hides important failure modes. A system can be "accurate"
on average while being dangerously wrong on specific categories.

### Anti-Pattern 4: "Testing on Training Data"

```
BAD:  Evaluate on the same examples used to develop prompts.
GOOD: Separate dev/eval/test splits with held-out test data.
```

**Why it's bad:** You are measuring memorization, not generalization. Your system will appear
much better than it actually is on novel inputs.

### Anti-Pattern 5: "Ignoring the Long Tail"

```
BAD:  Only test common, easy queries.
GOOD: Include adversarial, edge-case, and rare scenarios.
```

**Why it's bad:** Systems fail most often on unusual inputs. The long tail of rare queries
is where hallucinations and errors concentrate.

### Anti-Pattern 6: "Evaluating the Wrong Thing"

```
BAD:  Measuring BLEU score for open-ended QA.
GOOD: Using faithfulness + relevance metrics designed for the task.
```

**Why it's bad:** Choosing metrics that do not align with actual quality dimensions gives
you false confidence. High BLEU does not mean the answer is correct.

### Anti-Pattern 7: "No Versioning"

```
BAD:  Overwriting eval datasets and results.
GOOD: Version everything: datasets, prompts, results, configs.
```

**Why it's bad:** Without versioning, you cannot track progress, reproduce results,
or diagnose regressions.

---

## 1.12 Evaluation Across the ML Lifecycle

### Experimentation Phase

```
Goal:       Explore what works
Eval scope: Narrow (focused experiments)
Dataset:    Small (20-50 cases)
Metrics:    1-3 primary metrics
Speed:      Must be fast (seconds to minutes)
Rigor:      Low (directional signal is sufficient)

Example workflow:
  1. Try 5 different prompt variants
  2. Run each on 30 test cases
  3. Pick the best based on primary metric
  4. Iterate quickly
```

### Staging Phase

```
Goal:       Validate before production
Eval scope: Comprehensive
Dataset:    Large (500-2000 cases)
Metrics:    Full metric suite (5-15 metrics)
Speed:      Can be slower (minutes to hours)
Rigor:      High (statistical significance required)

Example workflow:
  1. Run full eval suite on candidate configuration
  2. Compare to current production baseline
  3. Check for regressions across ALL dimensions
  4. Generate detailed report
  5. Gate deployment on passing thresholds
```

### Production Phase

```
Goal:       Monitor and detect issues
Eval scope: Sampled but continuous
Dataset:    Real traffic (sampled 1-10%)
Metrics:    Key quality + operational metrics
Speed:      Real-time or near-real-time
Rigor:      Medium (alerting on significant drops)

Example workflow:
  1. Sample 5% of production traffic
  2. Run lightweight quality checks
  3. Aggregate metrics over time windows
  4. Alert when metrics drop below thresholds
  5. Deep-dive on flagged cases
```

### Summary Table

| Dimension        | Experimentation     | Staging              | Production           |
|------------------|---------------------|----------------------|----------------------|
| Dataset size     | 20-50               | 500-2000             | Continuous (sampled) |
| Metric count     | 1-3                 | 5-15                 | 3-8 key metrics      |
| Eval frequency   | Per experiment      | Per release          | Continuous           |
| Speed needed     | Fast (< 5 min)     | Moderate (< 2 hrs)  | Real-time            |
| Cost tolerance   | Low                 | Medium               | Ongoing budget       |
| Rigor required   | Directional         | Statistical          | Alerting-level       |
| Automation       | Optional            | Required             | Required             |
| Human review     | Ad-hoc              | Formal review        | Exception-based      |

---

## 1.13 Building Your First Evaluation: A Checklist

```
[ ] 1. DEFINE what you are evaluating
    - What system/component?
    - What are the inputs and outputs?
    - What does "good" look like?

[ ] 2. CHOOSE your metrics
    - What dimensions of quality matter most?
    - Which metrics capture those dimensions?
    - What are the minimum acceptable thresholds?

[ ] 3. CURATE your dataset
    - Collect representative test cases
    - Include edge cases and adversarial examples
    - Get ground truth labels (if reference-based)
    - Document the dataset (source, version, size)

[ ] 4. IMPLEMENT evaluation logic
    - Write metric computation code
    - Set up evaluation harness/framework
    - Configure judge models (if using LLM-as-a-judge)

[ ] 5. RUN and ANALYZE
    - Execute evaluation
    - Analyze results across categories
    - Identify failure patterns
    - Compare to baselines

[ ] 6. ITERATE
    - Fix identified issues
    - Re-evaluate
    - Update datasets with new cases
    - Track improvement over time

[ ] 7. AUTOMATE
    - Integrate into CI/CD pipeline
    - Set up production monitoring
    - Configure alerting
    - Schedule regular full evaluations
```

---

## 1.14 Key Takeaways

1. **Evaluations are not optional.** They are fundamental infrastructure for any AI system.
2. **Multiple dimensions matter.** No single metric captures all aspects of quality.
3. **Evaluation is continuous.** It spans the entire lifecycle from development to production.
4. **Start simple, iterate.** Begin with basic metrics and grow your eval suite over time.
5. **The cost of not evaluating far exceeds the cost of evaluating.** Every major AI failure can be traced to insufficient evaluation.
6. **Domain matters.** Evaluation strategies must be tailored to your specific use case.
7. **Avoid anti-patterns.** Vibes-based testing, single metrics, and stale datasets are dangerous.
8. **Automate everything.** Manual evaluation does not scale. Build it into your pipeline.

---

## References and Further Reading

- [RAGAS Documentation](https://docs.ragas.io/) - RAG evaluation framework
- [DeepEval Documentation](https://docs.confident-ai.com/) - LLM evaluation framework
- [LangSmith](https://docs.smith.langchain.com/) - LLM observability and evaluation
- [Braintrust](https://www.braintrust.dev/) - AI evaluation platform
- [HELM (Stanford)](https://crfm.stanford.edu/helm/) - Holistic Evaluation of Language Models
- [MMLU Benchmark](https://arxiv.org/abs/2009.03300) - Massive Multitask Language Understanding
- Shankar et al., "Who Validates the Validators?" (2024) - Meta-evaluation research
- Zheng et al., "Judging LLM-as-a-Judge" (2023) - Analysis of LLM judge reliability

---

*Next chapter: [02 - LLM Evals vs Traditional ML Evals](02_llm_evals_vs_traditional.md)*

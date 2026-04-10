# 04 - RAG Evaluation Fundamentals

> **Goal of this document:** Give you a complete mental model for evaluating Retrieval-Augmented Generation systems. After reading this you should be able to (a) decompose any RAG pipeline into evaluable components, (b) choose the right metrics for each component, (c) design an evaluation dataset, and (d) set up a repeatable evaluation loop. No other reference should be necessary.

---

## Table of Contents

1. [RAG Architecture Recap](#1-rag-architecture-recap)
2. [The RAG Evaluation Framework](#2-the-rag-evaluation-framework)
3. [Retriever Evaluation Concepts](#3-retriever-evaluation-concepts)
4. [Generator Evaluation Concepts](#4-generator-evaluation-concepts)
5. [End-to-End RAG Evaluation](#5-end-to-end-rag-evaluation)
6. [Practical Considerations](#6-practical-considerations)

---

## 1. RAG Architecture Recap

### 1.1 The Canonical RAG Pipeline

Every RAG system, no matter how complex, implements some variation of this data-flow:

```
User Query
    |
    v
[ Query Processor ]  (optional: query rewriting, HyDE, expansion)
    |
    v
[ Embedding Model ]  converts query to dense vector
    |
    v
[ Vector Store / Retriever ]  similarity search over document embeddings
    |
    v
[ Retrieved Chunks ]  (top-K passages)
    |
    v
[ Reranker ]  (optional: cross-encoder re-scoring)
    |
    v
[ Context Assembly ]  selected chunks formatted into a prompt window
    |
    v
[ LLM Generator ]  produces final answer conditioned on context
    |
    v
Response to User
```

Understanding this pipeline is prerequisite to evaluating it, because each stage introduces a distinct failure mode that requires a distinct metric.

### 1.2 Component Breakdown

| Component | Role | Typical Technology | Primary Failure Mode |
|-----------|------|-------------------|---------------------|
| **Query Processor** | Transform user query for better retrieval | Rule-based, LLM rewrite, HyDE | Distorts intent, loses nuance |
| **Embedding Model** | Encode text into dense vectors | OpenAI `text-embedding-3-small`, Cohere `embed-v3`, open-source `bge-large` | Poor semantic capture, domain mismatch |
| **Vector Store** | Store and search embeddings | Qdrant, Pinecone, Weaviate, ChromaDB, pgvector | Index corruption, stale data, wrong distance metric |
| **Retriever** | Return top-K most relevant chunks | Dense (ANN), sparse (BM25), hybrid | Misses relevant docs, ranks poorly |
| **Reranker** | Re-score retrieved chunks with cross-encoder | Cohere Rerank, `bge-reranker-v2`, `ms-marco-MiniLM` | Demotes relevant docs, adds latency |
| **Context Assembler** | Format chunks into prompt | Template logic, truncation, deduplication | Exceeds context window, loses important chunks |
| **LLM Generator** | Produce final answer | GPT-4.1, Claude, Llama 3, Mistral | Hallucination, irrelevance, verbosity |

### 1.3 RAG Variants

Understanding variants matters because each variant demands a different evaluation strategy.

#### Naive RAG
The simplest form: embed query, retrieve top-K, stuff into prompt, generate.

```
Query → Embed → Retrieve(top-K) → Generate → Answer
```

**Evaluation focus:** Retrieval quality is the bottleneck. Garbage in, garbage out.

#### Advanced RAG (with Reranking)
Adds a reranking stage between retrieval and generation.

```
Query → Embed → Retrieve(top-N) → Rerank(top-K) → Generate → Answer
```

**Evaluation focus:** You must evaluate the retriever AND the reranker separately. The reranker can either fix bad retrieval or make it worse. Measure precision before and after reranking.

#### Advanced RAG (with Query Transformation)
Adds query rewriting, expansion, or HyDE (Hypothetical Document Embeddings) before retrieval.

```
Query → Transform → Embed → Retrieve → [Rerank] → Generate → Answer
```

**Evaluation focus:** You need to evaluate whether the transformed query leads to better retrieval than the original. Compare retrieval metrics with and without transformation.

#### Modular RAG
Components are mix-and-matched. May include routing (choosing which index to query), multi-step retrieval, or iterative refinement.

```
Query → Router → [Index A or Index B] → Retrieve → Merge → Rerank → Generate
```

**Evaluation focus:** Each module needs isolated evaluation. The router needs accuracy metrics. Each index needs retrieval metrics. The merge strategy needs deduplication quality metrics.

#### Agentic RAG
An LLM agent decides when and how to retrieve. It may call retrieval tools multiple times, use different search strategies, or decide it already has enough information.

```
Query → Agent → [Tool: Search A] → [Tool: Search B] → [Tool: Calculate] → Synthesize → Answer
```

**Evaluation focus:** Standard RAG metrics PLUS agentic metrics: tool selection correctness, task completion, multi-turn coherence, and reasoning quality. This is the hardest variant to evaluate because the retrieval is dynamic and non-deterministic.

### 1.4 Why Each Component Needs Separate Evaluation

Consider a scenario: your RAG system gives a wrong answer. Without component-level evaluation, you have no idea where to fix the problem.

| Scenario | Root Cause | Fix |
|----------|-----------|-----|
| Wrong answer, correct context was retrieved | Generator hallucinated or misread context | Improve prompt, switch LLM, add grounding instructions |
| Wrong answer, relevant docs exist but were not retrieved | Retriever failed — embedding mismatch or wrong top-K | Improve embeddings, increase K, add hybrid search |
| Wrong answer, relevant docs don't exist in corpus | Knowledge gap | Add documents, expand corpus |
| Wrong answer, relevant docs retrieved but ranked low | Reranker failure or K too small after reranking | Tune reranker, increase K |
| Partially correct answer | Some relevant docs retrieved, some missed | Increase recall, add query expansion |

**The key insight:** End-to-end accuracy tells you THAT something is wrong. Component metrics tell you WHERE it is wrong and HOW to fix it.

---

## 2. The RAG Evaluation Framework

### 2.1 The Two Dimensions

Every RAG evaluation fundamentally measures two things:

1. **Retriever Quality:** Did the retriever find the right information?
2. **Generator Quality:** Did the generator produce a good answer from whatever information it received?

These are orthogonal dimensions. A perfect retriever paired with a bad generator still produces bad answers. A great generator paired with a bad retriever produces plausible-sounding wrong answers (arguably the most dangerous outcome).

### 2.2 The Retriever-Generator 2x2 Matrix

This is the single most important mental model for RAG evaluation:

```
                        Generator Quality
                    Good              Bad
                ┌──────────────┬──────────────┐
           Good │  CORRECT     │ HALLUCINATED │
Retriever       │  ANSWER      │ / POOR       │
Quality         │              │ SYNTHESIS    │
                │ Goal state.  │ Generator    │
                │ System works │ ignores or   │
                │ as intended. │ distorts the │
                │              │ context.     │
                ├──────────────┼──────────────┤
           Bad  │  PLAUSIBLE   │  COMPLETE    │
                │  BUT         │  FAILURE     │
                │  UNSUPPORTED │              │
                │ Generator    │ Both failed. │
                │ fills gaps   │ Easiest to   │
                │ from its     │ detect,      │
                │ parametric   │ hardest to   │
                │ knowledge.   │ fix.         │
                └──────────────┴──────────────┘
```

Let us examine each quadrant in detail:

#### Quadrant 1: Good Retrieval + Good Generation = Correct Answer

- The retriever found relevant, complete context
- The generator faithfully synthesized the context into a correct answer
- This is the target state
- **Metrics that confirm this:** High context precision, high context recall, high faithfulness, high answer relevancy

#### Quadrant 2: Good Retrieval + Bad Generation = Hallucination / Poor Synthesis

The context is right there in the prompt, but the generator:
- Ignores key information in the context
- Contradicts statements in the context (intrinsic hallucination)
- Adds fabricated details not in the context (extrinsic hallucination)
- Produces a vague or incomplete summary when a precise answer is available

**This is the most frustrating failure mode.** The information was retrieved correctly, but the LLM fumbled.

**Example:**
```
Context: "The company reported Q3 revenue of $4.2B, up 15% YoY."
Question: "What was the Q3 revenue?"
Bad answer: "The company reported Q3 revenue of approximately $4.5B."
```
The context was perfect. The generator introduced a factual error.

**Metrics that detect this:** Low faithfulness, low answer correctness despite high context precision and recall.

#### Quadrant 3: Bad Retrieval + Good Generation = Plausible but Unsupported Answer

The retriever failed to find the relevant context, but the generator:
- Uses its parametric knowledge (training data) to produce a plausible answer
- The answer might even be correct, but it is NOT grounded in the retrieved context
- This defeats the entire purpose of RAG (grounding responses in specific documents)

**This is the most dangerous failure mode.** The answer looks correct, the user trusts it, but it is not supported by the source documents. If the source documents are the authoritative source of truth (e.g., company policies, legal documents), this is catastrophic.

**Example:**
```
Retrieved Context: [chunks about company vacation policy, nothing about parental leave]
Question: "What is the parental leave policy?"
"Good" answer: "Most companies offer 12 weeks of parental leave..."
```
The answer might be generally true, but it is not from your company's documents.

**Metrics that detect this:** Low context recall, low faithfulness (if you measure grounding strictly), high answer relevancy (it addresses the question, just not from context).

#### Quadrant 4: Bad Retrieval + Bad Generation = Complete Failure

- The retriever found irrelevant context
- The generator produced an irrelevant, incoherent, or obviously wrong answer
- **Silver lining:** This is the easiest failure to detect (and often easiest to report as a failure to the user)

**Metrics that detect this:** Low scores across the board.

### 2.3 Why You MUST Evaluate Both Independently

If you only measure end-to-end accuracy:
- You cannot distinguish Quadrant 2 from Quadrant 3
- You cannot tell if your retriever needs fixing or your generator needs fixing
- You will waste time and money optimizing the wrong component
- Improvements to one component may mask regressions in the other

**Rule of thumb:** Always report at least one retriever metric and one generator metric alongside any end-to-end metric.

### 2.4 The Three-Axis Evaluation Model

A more nuanced framework uses three axes:

| Axis | Question | Key Metrics |
|------|----------|-------------|
| **Context Relevance** | Is the retrieved context relevant to the query? | Context Precision, Context Relevancy, NDCG |
| **Faithfulness** | Is the answer grounded in the retrieved context? | Faithfulness, Groundedness, Hallucination rate |
| **Answer Quality** | Is the answer good (relevant, correct, complete)? | Answer Relevancy, Answer Correctness, Answer Semantic Similarity |

These three axes are independent. You need all three to fully evaluate a RAG system.

```
                    Context Relevance
                         /\
                        /  \
                       /    \
                      /  RAG \
                     / Quality \
                    /    Zone   \
                   /______________\
     Faithfulness ──────────────── Answer Quality
```

A RAG system is only as strong as its weakest axis.

---

## 3. Retriever Evaluation Concepts

### 3.1 What Makes Good Retrieval

Good retrieval is not just about finding relevant documents. It encompasses four dimensions:

| Dimension | Definition | Why It Matters |
|-----------|-----------|---------------|
| **Relevance** | Retrieved chunks contain information pertinent to the query | Irrelevant chunks waste context window and confuse the generator |
| **Completeness** | All information needed to answer the query is retrieved | Missing information leads to incomplete or wrong answers |
| **Ranking** | The most relevant chunks appear first (highest rank) | LLMs exhibit position bias; they attend more to content at the top of the context |
| **Diversity** | Retrieved chunks cover different aspects of the query | Redundant chunks waste the context window without adding information |

### 3.2 Context Window Considerations

The context window is a scarce resource. What you include in it directly determines what the generator can work with.

#### Too Much Context (Over-retrieval)

```
Problem: Retrieving 20 chunks when 3 would suffice
```

- **Noise dilution:** Relevant information is buried among irrelevant passages
- **Lost in the middle:** Research shows LLMs pay more attention to the beginning and end of the context, often missing information in the middle (Liu et al., "Lost in the Middle," 2023)
- **Increased cost:** More input tokens = higher API costs
- **Increased latency:** More tokens to process = slower responses
- **Reduced faithfulness:** More irrelevant context = more opportunities for the generator to latch onto wrong information

#### Too Little Context (Under-retrieval)

```
Problem: Retrieving 1 chunk when 5 are needed for a complete answer
```

- **Incomplete answers:** The generator does not have enough information
- **Parametric fallback:** The generator fills gaps from training data (Quadrant 3 failure)
- **Missed nuance:** Complex questions often require synthesizing multiple sources

#### The Sweet Spot

There is no universal answer, but guidelines exist:

| Query Type | Recommended Context Size | Reasoning |
|-----------|------------------------|-----------|
| Simple factoid | 1-3 chunks | Single fact, low ambiguity |
| Comparison question | 3-5 chunks | Need info about multiple entities |
| Complex analytical | 5-10 chunks | Need multiple perspectives, data points |
| Multi-hop reasoning | 3-7 chunks (carefully selected) | Need chain of facts, quality > quantity |

### 3.3 Chunk Size Impact on Evaluation Scores

Chunk size is one of the most impactful RAG hyperparameters, and it directly affects evaluation metrics:

| Chunk Size | Effect on Precision | Effect on Recall | Effect on Generation |
|-----------|-------------------|-----------------|---------------------|
| **Small (100-200 tokens)** | Higher precision (less noise per chunk) | Lower recall per chunk (may need more chunks) | Fragmented context, may lose coherence |
| **Medium (300-500 tokens)** | Balanced | Balanced | Usually optimal for most use cases |
| **Large (500-1000 tokens)** | Lower precision (more noise per chunk) | Higher recall per chunk | More coherent context, but more noise |
| **Very large (1000+ tokens)** | Lowest precision | Highest recall per chunk | Risk of "lost in the middle" effect |

**Critical insight for evaluation:** When comparing two RAG systems, you MUST control for chunk size. A system with 200-token chunks and top-5 retrieval is working with ~1000 tokens of context. A system with 500-token chunks and top-5 retrieval is working with ~2500 tokens. Comparing their context precision scores directly is misleading.

### 3.4 Top-K and Its Effect on Metrics

Top-K is the number of chunks returned by the retriever.

```
Precision tends to DECREASE as K increases (more chances for irrelevant docs)
Recall tends to INCREASE as K increases (more chances to find relevant docs)
```

This is the classic precision-recall tradeoff:

```
Score
  ^
  |  ****
  | *    ****
  |*         ****  ← Recall
  |              ****
  |                  ****
  |
  |****
  |    ****
  |        ****  ← Precision
  |            ****
  |                ****
  +-----------------------> K
  1  3  5  7  10  15  20
```

**Practical guidance:**

| Use Case | Recommended K | Reasoning |
|----------|--------------|-----------|
| High-precision required (legal, medical) | 3-5 | Every chunk must be relevant; noise is dangerous |
| General Q&A | 5-10 | Balance between precision and recall |
| Exploratory / research | 10-20 | Want to cast a wide net, generator can filter |
| With reranker | Retrieve 20-50, rerank to top 5 | Let the reranker do the precision work |

### 3.5 Retriever Metrics Overview

#### 3.5.1 Context Precision

**What it measures:** Are the relevant chunks ranked higher than irrelevant chunks?

**Intuition:** If you retrieve 5 chunks and 2 are relevant, Context Precision rewards you more if those 2 relevant chunks are at positions 1 and 2, rather than positions 4 and 5.

**Formula (Weighted Cumulative Precision):**

```
Context Precision@K = (1/Number of Relevant Docs) * Sum_{k=1}^{K} [Precision@k * rel(k)]

where:
  Precision@k = (Number of relevant docs in top k) / k
  rel(k) = 1 if the document at rank k is relevant, 0 otherwise
```

**Example calculation:**

```
Retrieved chunks: [Relevant, Irrelevant, Relevant, Irrelevant, Irrelevant]
Relevance vector:  [1, 0, 1, 0, 0]

Precision@1 = 1/1 = 1.0    rel(1) = 1  → contributes 1.0
Precision@2 = 1/2 = 0.5    rel(2) = 0  → contributes 0
Precision@3 = 2/3 = 0.667  rel(3) = 1  → contributes 0.667
Precision@4 = 2/4 = 0.5    rel(4) = 0  → contributes 0
Precision@5 = 2/5 = 0.4    rel(5) = 0  → contributes 0

Context Precision = (1/2) * (1.0 + 0 + 0.667 + 0 + 0) = 0.833
```

**Score range:** 0.0 to 1.0
**Requires:** retrieval_context, expected_output (or relevance judgments)
**When to use:** When ranking quality matters (almost always)

#### 3.5.2 Context Recall

**What it measures:** Did the retriever find ALL the information needed to answer the question?

**Intuition:** If the ground truth answer contains 5 key claims, and your retrieved context supports 4 of them, your context recall is 0.8.

**Formula:**

```
Context Recall = |Ground Truth Statements Attributable to Context| / |Total Ground Truth Statements|
```

**Example:**

```
Expected answer: "Python was created by Guido van Rossum in 1991. It emphasizes readability."

Statements:
  S1: "Python was created by Guido van Rossum" → Found in context? YES
  S2: "Python was created in 1991" → Found in context? YES
  S3: "Python emphasizes readability" → Found in context? NO

Context Recall = 2/3 = 0.667
```

**Score range:** 0.0 to 1.0
**Requires:** retrieval_context, expected_output (ground truth)
**When to use:** When completeness is important (medical, legal, compliance)

#### 3.5.3 Context Relevancy

**What it measures:** What fraction of the retrieved context is actually relevant to the query?

**Intuition:** If you retrieve 1000 tokens and only 200 are relevant, your context relevancy is low. You are wasting context window space.

**Formula:**

```
Context Relevancy = |Relevant Statements in Context| / |Total Statements in Context|
```

**Score range:** 0.0 to 1.0
**Requires:** input (query), retrieval_context
**When to use:** When you want to minimize noise in the context window

#### 3.5.4 Context Entity Recall

**What it measures:** What fraction of entities in the ground truth answer are present in the retrieved context?

**Intuition:** If the correct answer mentions "Tesla," "Elon Musk," and "2023," and only "Tesla" and "2023" appear in the retrieved context, entity recall is 2/3.

**Formula:**

```
Context Entity Recall = |Entities in Ground Truth ∩ Entities in Context| / |Entities in Ground Truth|
```

**Score range:** 0.0 to 1.0
**Requires:** retrieval_context, expected_output
**When to use:** Factoid questions, entity-heavy domains (finance, healthcare)

#### 3.5.5 Noise Sensitivity

**What it measures:** How much does irrelevant context degrade the generator's output?

**Intuition:** Add noise chunks to the context and measure how much the answer quality drops. A robust system should tolerate some noise.

**Score range:** 0.0 to 1.0 (higher = more robust to noise)
**Requires:** Controlled experiment with and without noise
**When to use:** Stress testing, comparing retrieval strategies

### 3.6 Traditional IR Metrics and Their Relevance to RAG

These metrics predate RAG by decades but remain highly relevant.

#### 3.6.1 Hit Rate (Recall@K)

**What it measures:** Is at least one relevant document in the top-K results?

```
Hit Rate@K = 1 if any relevant doc is in top-K, else 0
```

**Average across queries** to get the system's hit rate.

**When to use:** As a minimum bar. If your hit rate@5 is below 0.8, you have a serious retrieval problem.

#### 3.6.2 Mean Reciprocal Rank (MRR)

**What it measures:** How high is the FIRST relevant document ranked?

```
MRR = (1/|Q|) * Sum_{i=1}^{|Q|} (1 / rank_i)

where rank_i is the rank of the first relevant document for query i.
```

**Example:**

```
Query 1: First relevant doc at rank 2 → 1/2 = 0.5
Query 2: First relevant doc at rank 1 → 1/1 = 1.0
Query 3: First relevant doc at rank 5 → 1/5 = 0.2

MRR = (0.5 + 1.0 + 0.2) / 3 = 0.567
```

**When to use:** When you care most about the first relevant result (factoid QA).

#### 3.6.3 Normalized Discounted Cumulative Gain (NDCG)

**What it measures:** How well is the entire ranked list ordered, with graded relevance?

Unlike precision and MRR which use binary relevance (relevant or not), NDCG supports graded relevance (e.g., highly relevant = 3, somewhat relevant = 2, marginally relevant = 1, irrelevant = 0).

```
DCG@K = Sum_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)

IDCG@K = DCG@K for the ideal ranking (sort docs by relevance descending)

NDCG@K = DCG@K / IDCG@K
```

**Example:**

```
Retrieved ranking:  [rel=3, rel=0, rel=2, rel=1]
Ideal ranking:      [rel=3, rel=2, rel=1, rel=0]

DCG@4  = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4) + (2^1-1)/log2(5)
       = 7/1 + 0/1.585 + 3/2 + 1/2.322
       = 7.0 + 0.0 + 1.5 + 0.431 = 8.931

IDCG@4 = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + (2^0-1)/log2(5)
        = 7/1 + 3/1.585 + 1/2 + 0/2.322
        = 7.0 + 1.893 + 0.5 + 0.0 = 9.393

NDCG@4 = 8.931 / 9.393 = 0.951
```

**Score range:** 0.0 to 1.0
**When to use:** When you have graded relevance labels and care about the entire ranking, not just the top result.

#### 3.6.4 Comparison Table

| Metric | Graded Relevance? | Position-Sensitive? | Requires Ground Truth? | Best For |
|--------|-------------------|--------------------|-----------------------|----------|
| Hit Rate@K | No | No (just checks presence) | Yes | Minimum viability check |
| MRR | No | Yes (first relevant) | Yes | Factoid QA |
| NDCG@K | Yes | Yes (full ranking) | Yes (graded) | Complete ranking quality |
| Context Precision | No (via LLM judge) | Yes (weighted) | Yes | LLM-judged RAG eval |
| Context Recall | No (via LLM judge) | No | Yes | LLM-judged completeness |

### 3.7 How Embedding Model Choice Affects Retrieval Quality

The embedding model is the foundation of dense retrieval. A poor embedding model means your retriever cannot even represent the semantics of queries and documents.

| Factor | Impact | Example |
|--------|--------|---------|
| **Model size** | Larger models generally produce better embeddings | `text-embedding-3-large` (3072d) vs `text-embedding-3-small` (1536d) |
| **Training data** | Models trained on your domain perform better | General-purpose vs domain-fine-tuned |
| **Dimensionality** | Higher dimensions capture more nuance but cost more storage | 384d vs 768d vs 1536d vs 3072d |
| **Max sequence length** | Determines how much text can be embedded at once | 512 tokens vs 8192 tokens |
| **Multilingual support** | Critical for non-English or mixed-language corpora | `multilingual-e5-large` vs English-only models |
| **Asymmetric vs symmetric** | Query-doc retrieval needs asymmetric; doc-doc needs symmetric | E5 uses "query:" and "passage:" prefixes |

**Evaluation tip:** When comparing embedding models, keep all other components constant and compare retrieval metrics (hit rate, MRR, NDCG) on the same test set.

**Common embedding models ranked by general quality (as of early 2025):**

| Model | Provider | Dimensions | Approximate MTEB Score |
|-------|----------|-----------|----------------------|
| `text-embedding-3-large` | OpenAI | 3072 | ~64.6 |
| `voyage-3-large` | Voyage AI | 1024 | ~68.0 |
| `embed-v4.0` | Cohere | 1024 | ~67.0 |
| `bge-en-icl` | BAAI (open-source) | 4096 | ~66.0 |
| `e5-mistral-7b-instruct` | Microsoft (open-source) | 4096 | ~66.6 |
| `gte-large-en-v1.5` | Alibaba (open-source) | 1024 | ~65.4 |
| `all-MiniLM-L6-v2` | Sentence Transformers (open-source) | 384 | ~56.3 |

Scores are approximate and evolve rapidly. Check the MTEB leaderboard for current rankings.

---

## 4. Generator Evaluation Concepts

### 4.1 What Makes a Good Generation

A good RAG generation satisfies four criteria:

| Criterion | Definition | Failure Example |
|-----------|-----------|-----------------|
| **Relevant** | Addresses the user's actual question | Q: "What is the refund policy?" A: "We have a great customer service team." |
| **Faithful** | Every claim is supported by the retrieved context | Context says "$4.2B revenue" but answer says "$4.5B" |
| **Complete** | Covers all aspects of the question using available context | Q: "Compare plans A and B" A: Only describes Plan A |
| **Concise** | Does not include unnecessary information or verbosity | Restates the entire context instead of synthesizing |

### 4.2 The Faithfulness Problem

Faithfulness is the single most important generator metric in RAG. The entire point of RAG is to ground generation in specific context. An unfaithful generator defeats the purpose.

**Faithfulness** = the degree to which the generated output is supported by the provided context.

```
Faithfulness Score = |Claims in output supported by context| / |Total claims in output|
```

#### Why Faithfulness Fails

1. **The LLM's parametric knowledge overrides context.** The model "knows" something from training and uses that instead of the context, even when the context says something different.

2. **The LLM cannot resist adding helpful information.** Asked "what is X?", the LLM adds related facts Y and Z that are not in the context.

3. **The LLM misreads the context.** Numbers, dates, names, and technical terms are frequently misquoted.

4. **The LLM over-generalizes.** The context says "Company X did Y in Q3 2023" and the LLM says "Companies typically do Y."

5. **Prompt-induced hallucination.** Prompts that say "be helpful and comprehensive" encourage the LLM to go beyond the context.

### 4.3 Answer Relevancy

**What it measures:** Does the answer actually address the question that was asked?

A common failure mode is when the generator produces a perfectly faithful summary of the context, but the context was about a different topic than the question. The answer is faithful but irrelevant.

**Algorithm (DeepEval / RAGAS approach):**

1. Given the answer, generate N synthetic questions that the answer could be responding to
2. Compute the semantic similarity between each synthetic question and the original question
3. Answer relevancy = average similarity

**Intuition:** If the answer is relevant to the question, then questions generated from the answer should be similar to the original question.

```
Original question: "What is the return policy for electronics?"
Answer: "Electronics can be returned within 30 days with receipt."

Generated questions from answer:
  Q1: "What is the return window for electronics?" (sim = 0.92)
  Q2: "How long do I have to return electronic items?" (sim = 0.88)
  Q3: "What do I need to return electronics?" (sim = 0.85)

Answer Relevancy = mean(0.92, 0.88, 0.85) = 0.883
```

### 4.4 Hallucination Types in RAG

Understanding hallucination taxonomy is critical for building the right evaluation.

#### 4.4.1 Intrinsic Hallucination (Contradicts Context)

The output directly contradicts information in the retrieved context.

```
Context: "The meeting is scheduled for March 15, 2024."
Output:  "The meeting is scheduled for March 16, 2024."
```

**Detection difficulty:** Medium. Can be caught by claim-level faithfulness checking.
**Severity:** High. The user trusts the system but gets wrong information.

#### 4.4.2 Extrinsic Hallucination (Adds Unsupported Information)

The output includes claims that are not present in the context, though they may not directly contradict it.

```
Context: "The company was founded in 2010."
Output:  "The company was founded in 2010 in San Francisco."
```

"In San Francisco" is not in the context. It might be true (from the LLM's training data), but it is not grounded.

**Detection difficulty:** Medium-High. Requires checking that every claim in the output has a supporting claim in the context.
**Severity:** Medium-High. Information may be correct but is unverifiable from the provided sources.

#### 4.4.3 Fabrication (Invents Facts Entirely)

The output invents information that has no basis in the context or reality.

```
Context: "The product costs $29.99."
Output:  "The product costs $29.99 and comes with a lifetime warranty backed by the Smith Foundation."
```

The Smith Foundation does not exist. This is pure fabrication.

**Detection difficulty:** Lower (often sounds unlikely), but dangerous when the fabrication is plausible.
**Severity:** Critical. The user may act on fabricated information.

#### Hallucination Detection Summary

| Type | Definition | Example | Detection Method |
|------|-----------|---------|-----------------|
| Intrinsic | Contradicts context | Wrong number, date, name | Claim-vs-context comparison |
| Extrinsic | Not in context, may be true | Adding unsupported details | Claim-vs-context exhaustive check |
| Fabrication | Not in context, not true | Inventing entities, facts | Claim-vs-context + fact verification |

### 4.5 Generator Metrics Overview

#### 4.5.1 Faithfulness / Groundedness

**What it measures:** What fraction of the claims in the output are supported by the retrieved context?

**Algorithm:**
1. Extract all claims / statements from the generated output
2. For each claim, check if it is supported by the retrieval context
3. Score = number of supported claims / total claims

```
Output: "Python was created by Guido van Rossum. It was first released in 1991. 
         It is the most popular language in the world."

Claims extracted:
  C1: "Python was created by Guido van Rossum" → Context supports? YES
  C2: "It was first released in 1991" → Context supports? YES
  C3: "It is the most popular language in the world" → Context supports? NO

Faithfulness = 2/3 = 0.667
```

**Score range:** 0.0 to 1.0
**Requires:** actual_output, retrieval_context
**When to use:** ALWAYS. This is the most fundamental RAG generator metric.

#### 4.5.2 Answer Relevancy

**What it measures:** Does the answer address the question? (Covered in section 4.3 above.)

**Score range:** 0.0 to 1.0
**Requires:** input, actual_output
**When to use:** ALWAYS alongside faithfulness. Together they catch most generator failures.

#### 4.5.3 Answer Correctness

**What it measures:** Is the answer factually correct compared to a ground truth reference answer?

**Algorithm:** Typically combines:
- **Semantic similarity** between output and expected output
- **Factual overlap** via claim-level F1 score

```
Answer Correctness = w1 * Semantic_Similarity + w2 * Factual_F1

where typically w1 = 0.5, w2 = 0.5
```

**Factual F1 breakdown:**

```
Expected: "Python was created by Guido van Rossum in 1991."
Actual:   "Guido van Rossum created Python in the early 1990s. It uses indentation."

True Positives (TP): claims in both → "Guido created Python" (1)
False Positives (FP): claims only in actual → "early 1990s" (partial), "uses indentation" (1)  
False Negatives (FN): claims only in expected → "in 1991" (partial) (1)

Precision = TP / (TP + FP) = 1 / (1 + 1.5) = 0.4
Recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
F1 = 2 * (0.4 * 0.5) / (0.4 + 0.5) = 0.444
```

**Score range:** 0.0 to 1.0
**Requires:** actual_output, expected_output
**When to use:** When you have ground truth answers in your evaluation dataset.

#### 4.5.4 Answer Semantic Similarity

**What it measures:** How semantically similar is the output to the expected output?

**Algorithm:** Compute embedding of actual output and expected output, then calculate cosine similarity.

```
Similarity = cosine(embed(actual_output), embed(expected_output))
```

**Score range:** 0.0 to 1.0 (cosine similarity of normalized embeddings)
**Requires:** actual_output, expected_output
**When to use:** As a softer correctness metric that allows paraphrasing. Use alongside, not instead of, answer correctness.

---

## 5. End-to-End RAG Evaluation

### 5.1 Why Component Metrics Aren't Enough

Component metrics tell you about each piece in isolation. But RAG is a pipeline, and pipelines have emergent behaviors:

- A retriever that retrieves slightly noisy context might still work fine if the generator is good at filtering
- A generator that is slightly unfaithful might still produce correct answers if the context is very focused
- Two components that are individually "good enough" might combine poorly (e.g., the generator expects context in a certain format that the retriever does not provide)

**End-to-end metrics measure what ultimately matters: did the user get a good answer?**

### 5.2 End-to-End Metrics

| Metric | What It Measures | Requires Ground Truth? |
|--------|-----------------|----------------------|
| **Answer Correctness** | Is the final answer correct? | Yes |
| **Answer Semantic Similarity** | Is the final answer semantically close to the reference? | Yes |
| **User Satisfaction** (human eval) | Did the user find the answer helpful? | No (but needs humans) |
| **Task Completion Rate** | Did the RAG system successfully complete the user's task? | Depends on task definition |
| **Latency** | How long did the full pipeline take? | No |
| **Cost** | How much did the full pipeline cost? | No |

### 5.3 The Role of Ground Truth

Ground truth is the expected answer for a given query. It is the gold standard against which you measure.

#### When You Have Ground Truth (Reference-Based Evaluation)

You can compute:
- Answer Correctness (comparing output to reference)
- Answer Semantic Similarity
- Context Recall (checking if context supports the reference)
- Context Precision (checking if retrieved docs are relevant per the reference)
- F1, exact match, BLEU, ROUGE (traditional NLP metrics)

**Advantages:** Objective, reproducible, automatable
**Disadvantages:** Expensive to create, may become stale, may not capture all valid answers

#### When You Don't Have Ground Truth (Reference-Free Evaluation)

You can still compute:
- Faithfulness (does the output match the context?)
- Answer Relevancy (does the output address the input?)
- Context Relevancy (is the retrieved context relevant to the input?)
- Hallucination detection
- Toxicity, bias

**Advantages:** No annotation needed, can evaluate every production query
**Disadvantages:** Cannot tell if the answer is correct, only that it is faithful and relevant

#### Decision Guide

```
Do you have ground truth answers?
├── YES → Use reference-based metrics (Answer Correctness, Context Recall)
│         PLUS reference-free metrics (Faithfulness, Answer Relevancy)
│
└── NO → Use reference-free metrics only
         Consider creating ground truth for a sample of queries
         Consider having domain experts spot-check answers
```

### 5.4 Building a RAG Evaluation Dataset

A good evaluation dataset is your most valuable asset. Here is how to build one:

#### Step 1: Collect Representative Queries

| Source | Method | Pros | Cons |
|--------|--------|------|------|
| Production logs | Sample real user queries | Most realistic | May contain PII, needs cleaning |
| Domain experts | Ask SMEs to write questions | High quality, covers edge cases | Expensive, limited scale |
| Synthetic generation | Use LLMs to generate questions from documents | Scalable, cheap | May not reflect real usage patterns |
| Adversarial generation | Generate hard/edge-case queries | Tests robustness | May not reflect typical usage |

#### Step 2: Create Ground Truth Answers

For each query, create:
- **Reference answer:** The ideal answer (used for Answer Correctness)
- **Relevant passages:** The chunks that contain the answer (used for Context Recall, Context Precision)
- **Metadata:** Query type, difficulty, topic, expected behavior

```python
# Example evaluation dataset entry
{
    "query": "What is the maximum loan amount for first-time homebuyers?",
    "reference_answer": "First-time homebuyers can borrow up to $350,000 with a maximum LTV of 95%.",
    "relevant_passages": [
        "Section 4.2: First-time homebuyer loans are capped at $350,000...",
        "Section 4.3: The maximum loan-to-value ratio for first-time buyers is 95%..."
    ],
    "metadata": {
        "type": "factoid",
        "difficulty": "easy",
        "topic": "mortgage",
        "source_document": "lending_policy_v3.pdf"
    }
}
```

#### Step 3: Ensure Dataset Quality

| Quality Check | Description |
|--------------|-------------|
| **Coverage** | Queries cover all topics in your document corpus |
| **Difficulty distribution** | Mix of easy, medium, hard queries |
| **Query type distribution** | Factoid, comparison, analytical, multi-hop, unanswerable |
| **Include negative cases** | Queries that CANNOT be answered from the corpus (test for appropriate "I don't know" responses) |
| **Freshness** | Update when documents change |
| **Minimum size** | At least 50-100 queries for statistical significance (more is better) |

#### Step 4: Include Unanswerable Queries

This is often overlooked but critical. Your dataset should include queries where the answer is NOT in the corpus. The expected behavior is that the RAG system says "I don't have this information" or equivalent.

```python
{
    "query": "What is the CEO's favorite color?",
    "reference_answer": "This information is not available in the knowledge base.",
    "relevant_passages": [],  # empty — nothing should be relevant
    "metadata": {
        "type": "unanswerable",
        "expected_behavior": "decline_to_answer"
    }
}
```

### 5.5 The Evaluation Loop

RAG evaluation is not a one-time activity. It is a continuous process:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Evaluate  │───>│ Identify │───>│   Fix    │  │
│  │ Pipeline  │    │ Weakness │    │ Component│  │
│  └──────────┘    └──────────┘    └──────────┘  │
│       ^                               │         │
│       │                               │         │
│       └───────────────────────────────┘         │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │ Track metrics over time (regression?)    │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Concrete example of the loop:**

1. **Evaluate:** Run evaluation suite. Faithfulness = 0.82, Context Recall = 0.65, Answer Correctness = 0.71
2. **Identify weakness:** Context Recall is low. The retriever is missing relevant information.
3. **Fix:** Increase top-K from 3 to 5, add hybrid search (BM25 + dense)
4. **Re-evaluate:** Context Recall = 0.78 (improved), Faithfulness = 0.80 (slight decrease due to more noise), Answer Correctness = 0.76 (improved)
5. **Identify weakness:** Faithfulness dropped slightly. Too much noise in the additional context.
6. **Fix:** Add a reranker to filter the top-5 after retrieving top-20
7. **Re-evaluate:** Context Recall = 0.82, Faithfulness = 0.88, Answer Correctness = 0.81

This iterative process is how you systematically improve a RAG system.

---

## 6. Practical Considerations

### 6.1 How Many Test Cases Do You Need?

Statistical significance matters. With too few test cases, your metrics are unreliable.

**Rule of thumb for confidence intervals:**

| Desired Margin of Error | Required Sample Size (95% confidence) |
|------------------------|--------------------------------------|
| +/- 10% | ~100 test cases |
| +/- 5% | ~400 test cases |
| +/- 3% | ~1,000 test cases |
| +/- 1% | ~10,000 test cases |

These assume binary outcomes (correct/incorrect). For continuous metrics (0.0 to 1.0), the required sample size depends on variance.

**Practical guidance:**

| Stage | Recommended Size | Reasoning |
|-------|-----------------|-----------|
| Prototyping | 20-50 | Quick signal, directional only |
| Pre-production | 100-200 | Reasonable confidence |
| Production monitoring | 200-500 | Statistically significant for most use cases |
| Benchmark / paper | 500-1000+ | Publication-quality results |

**A/B testing note:** When comparing two systems, you need enough test cases to detect the expected difference. Use a power analysis:

```python
# Rough power analysis for comparing two systems
from scipy import stats
import numpy as np

def required_sample_size(effect_size, alpha=0.05, power=0.8):
    """
    effect_size: expected difference in means / pooled standard deviation
    Returns required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# Example: detect a 0.05 improvement in faithfulness (std ~ 0.15)
effect_size = 0.05 / 0.15  # ~0.33
n = required_sample_size(effect_size)
print(f"Need {n} test cases per group")  # ~73
```

### 6.2 Dealing with Non-Determinism

LLMs are non-deterministic. The same query can produce different answers (and different evaluation scores) on different runs.

**Sources of non-determinism:**
1. **Generator temperature:** Higher temperature = more randomness
2. **Evaluation model temperature:** The LLM judge itself is non-deterministic
3. **Retriever randomness:** Some vector stores have non-deterministic ANN search
4. **API-level batching:** Some APIs have non-determinism even at temperature=0

**Mitigation strategies:**

| Strategy | Implementation | Tradeoff |
|----------|---------------|----------|
| Set temperature=0 | `model.temperature = 0` | Reduces but does not eliminate non-determinism |
| Fixed seed | `model.seed = 42` (if supported) | Not all APIs support seeds; even with seeds, results may vary |
| Multiple runs | Run each test case 3-5 times, take mean | 3-5x cost increase |
| Larger test sets | More test cases smooth out variance | Requires more ground truth |
| Deterministic metrics | Use exact match, F1, ROUGE instead of LLM-judge | Less nuanced, may miss semantic equivalence |

**Recommendation:** For production evaluation, run each test case at least 3 times and report mean +/- standard deviation. For quick iteration, single runs are acceptable if you have 100+ test cases.

### 6.3 Cost Estimation for RAG Evaluation

LLM-as-judge evaluation is not free. Each metric evaluation involves one or more LLM calls.

**Cost formula per test case:**

```
Cost = Sum over all metrics of (
    input_tokens_per_metric * input_price_per_token +
    output_tokens_per_metric * output_price_per_token
)
```

**Approximate LLM calls per metric (using GPT-4.1 as judge):**

| Metric | LLM Calls per Test Case | Approximate Input Tokens | Approximate Cost (GPT-4.1) |
|--------|------------------------|--------------------------|---------------------------|
| Faithfulness | 2 (extract claims + verify) | 2,000-5,000 | $0.008-$0.020 |
| Answer Relevancy | 1-2 (generate questions + embed) | 1,000-3,000 | $0.004-$0.012 |
| Context Precision | 1-2 | 2,000-5,000 | $0.008-$0.020 |
| Context Recall | 1-2 | 2,000-5,000 | $0.008-$0.020 |
| Answer Correctness | 1-2 | 1,500-3,000 | $0.006-$0.012 |
| G-Eval (custom) | 1-2 | 1,000-3,000 | $0.004-$0.012 |

**Total cost estimate for a full evaluation run:**

```
5 metrics x $0.01 avg per test case = $0.05 per test case
100 test cases = $5
500 test cases = $25
1000 test cases = $50

With 3 runs for statistical robustness:
100 test cases x 3 = $15
500 test cases x 3 = $75
```

**Cost reduction strategies:**
- Use GPT-4.1-mini or GPT-4.1-nano for some metrics (Faithfulness benefits from larger models; answer relevancy can use smaller models)
- Cache results for unchanged test cases
- Evaluate a sample (not all production queries)
- Use deterministic metrics where possible (exact match, F1, ROUGE)
- Batch evaluations to benefit from batch API discounts (if offered)

### 6.4 Evaluation Frequency

| Trigger | Metrics to Run | Typical Cost |
|---------|---------------|-------------|
| **Every commit (CI)** | 20-50 smoke tests, core metrics only | $1-3 |
| **Nightly** | Full test suite, all metrics | $25-75 |
| **Per release** | Full test suite + manual review of edge cases | $50-100 + human time |
| **Post-incident** | Targeted evaluation on failure queries | Variable |
| **Monthly regression** | Full test suite + comparison to historical baseline | $50-100 |

### 6.5 When to Use Which Metrics: Decision Guide

Use this decision tree to select metrics for your RAG evaluation:

```
START: What do you want to evaluate?
│
├── Retriever only?
│   ├── Do you have ground truth relevant passages?
│   │   ├── YES → Context Recall + Context Precision + NDCG
│   │   └── NO  → Context Relevancy (LLM-judged)
│   │
│   └── Do you care about ranking?
│       ├── YES → Context Precision, MRR, NDCG
│       └── NO  → Hit Rate, Context Recall
│
├── Generator only?
│   ├── Is faithfulness critical? (legal, medical, finance)
│   │   ├── YES → Faithfulness (threshold >= 0.9) + Hallucination check
│   │   └── NO  → Faithfulness (threshold >= 0.7)
│   │
│   ├── Do you have ground truth answers?
│   │   ├── YES → Answer Correctness + Answer Semantic Similarity
│   │   └── NO  → Answer Relevancy + Faithfulness
│   │
│   └── Do you need to check for harmful content?
│       └── YES → Bias + Toxicity metrics
│
└── End-to-end?
    ├── With ground truth → Answer Correctness + Context Recall + Faithfulness
    ├── Without ground truth → Faithfulness + Answer Relevancy + Context Relevancy
    └── Always add → Latency + Cost metrics
```

### 6.6 Common Pitfalls in RAG Evaluation

| Pitfall | Description | Prevention |
|---------|-------------|-----------|
| **Evaluating only end-to-end** | Cannot diagnose which component failed | Always include component metrics |
| **Too few test cases** | Metrics are unreliable | Minimum 50-100 for meaningful signal |
| **Test set leakage** | Test queries overlap with training/fine-tuning data | Strict train/test separation |
| **Stale ground truth** | Documents changed but ground truth answers did not | Version-lock evaluation dataset to document versions |
| **Ignoring unanswerable queries** | System never says "I don't know" | Include 10-20% unanswerable queries |
| **Single-metric optimization** | Optimizing faithfulness at expense of relevancy | Track multiple metrics, look for tradeoffs |
| **Ignoring latency and cost** | Accurate but slow/expensive system | Include non-quality metrics |
| **Over-relying on LLM judges** | LLM judges have biases (verbosity, position) | Cross-validate with human evaluation |
| **Not controlling for randomness** | Metrics fluctuate across runs | Multiple runs, report mean +/- std |
| **Cherry-picking results** | Showing only good examples | Report aggregate metrics, not anecdotes |

### 6.7 Metric Correlation and Redundancy

Not all metrics are independent. Understanding correlations helps you choose a minimal but informative set.

| Metric Pair | Typical Correlation | Interpretation |
|------------|-------------------|----------------|
| Faithfulness & Answer Correctness | Moderate (0.4-0.6) | Faithful answers tend to be correct, but not always (context may be wrong) |
| Context Precision & Answer Correctness | Moderate (0.5-0.7) | Better retrieval usually leads to better answers |
| Context Recall & Faithfulness | Low-Moderate (0.3-0.5) | More complete context helps, but does not guarantee faithfulness |
| Answer Relevancy & Answer Correctness | Moderate (0.4-0.6) | Relevant answers tend to be correct, but can be relevantly wrong |
| Context Precision & Context Recall | Low (0.2-0.4) | Often inversely related (precision-recall tradeoff) |

**Minimum metric set for a comprehensive RAG evaluation:**

1. **Context Recall** (retriever completeness)
2. **Context Precision** (retriever ranking)
3. **Faithfulness** (generator grounding)
4. **Answer Relevancy** (generator quality)
5. **Answer Correctness** (end-to-end, if ground truth available)

These five metrics cover the three axes (context relevance, faithfulness, answer quality) with minimal redundancy.

### 6.8 Evaluation Anti-Patterns

**Anti-pattern 1: "We'll evaluate later"**
RAG evaluation should start on day one. Even a basic smoke test with 10 queries is better than nothing. Teams that delay evaluation inevitably ship broken RAG systems to production.

**Anti-pattern 2: "Our users will tell us if something is wrong"**
By the time users complain, trust is eroded. Proactive evaluation catches problems before users do. Also, users often silently disengage rather than reporting issues.

**Anti-pattern 3: "We only need faithfulness"**
Faithfulness alone does not tell you if the answer is relevant, correct, or complete. A system that always responds "Based on the context, I cannot determine the answer" has perfect faithfulness but zero utility.

**Anti-pattern 4: "We evaluated once, we're good"**
Documents change. User behavior changes. LLM behavior changes (API model updates). Evaluation must be continuous.

**Anti-pattern 5: "Our evaluation dataset is our demo queries"**
Demo queries are chosen to make the system look good. Your evaluation dataset must include hard cases, edge cases, adversarial cases, and unanswerable queries.

---

## Summary: The RAG Evaluation Cheat Sheet

| Aspect | Key Question | Primary Metrics | Required Inputs |
|--------|-------------|-----------------|-----------------|
| **Retrieval Quality** | Did we find the right context? | Context Precision, Context Recall, NDCG | retrieval_context, expected_output |
| **Generation Faithfulness** | Is the answer grounded in context? | Faithfulness, Hallucination rate | actual_output, retrieval_context |
| **Answer Quality** | Is the answer good? | Answer Relevancy, Answer Correctness | input, actual_output, expected_output |
| **End-to-End** | Does the whole system work? | Answer Correctness, Task Completion | input, actual_output, expected_output |
| **Operational** | Is it fast and cheap enough? | Latency (p50, p95, p99), Cost per query | Monitoring data |

**The golden rule of RAG evaluation: Never evaluate only the output. Always evaluate the retrieval AND the generation independently, then together.**

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **Chunk** | A segment of a document stored in the vector database |
| **Context window** | The maximum number of tokens the LLM can process at once |
| **Dense retrieval** | Retrieval using embedding vectors and similarity search |
| **Sparse retrieval** | Retrieval using keyword matching (e.g., BM25) |
| **Hybrid retrieval** | Combining dense and sparse retrieval |
| **HyDE** | Hypothetical Document Embeddings — generate a hypothetical answer, embed it, then retrieve |
| **Parametric knowledge** | Knowledge encoded in the LLM's weights during training |
| **Grounding** | Ensuring generated text is supported by provided context |
| **Recall@K** | Fraction of relevant documents retrieved in the top-K results |
| **Precision@K** | Fraction of top-K retrieved documents that are relevant |
| **LLM-as-judge** | Using an LLM to evaluate the output of another LLM |
| **Reference-based evaluation** | Evaluation that compares output to a ground truth answer |
| **Reference-free evaluation** | Evaluation without a ground truth answer |
| **MTEB** | Massive Text Embedding Benchmark — standardized benchmark for embedding models |
| **ANN** | Approximate Nearest Neighbor — efficient similarity search algorithm |
| **Cross-encoder** | A model that takes (query, document) pair as input and outputs a relevance score — used in reranking |

## Appendix B: Further Reading

| Resource | Topic | Link |
|----------|-------|------|
| "Lost in the Middle" (Liu et al., 2023) | LLM attention over long contexts | arxiv.org/abs/2307.03172 |
| MTEB Leaderboard | Embedding model comparison | huggingface.co/spaces/mteb/leaderboard |
| RAGAS paper (Es et al., 2023) | RAG evaluation framework | arxiv.org/abs/2309.15217 |
| "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) | Original RAG paper | arxiv.org/abs/2005.11401 |
| DeepEval Documentation | DeepEval framework | docs.confident-ai.com |
| ARES (Saad-Falcon et al., 2023) | Automated RAG evaluation | arxiv.org/abs/2311.09476 |

---

**Next:** [05 - DeepEval Complete Guide](05_deepeval_complete_guide.md) -- The definitive reference for the DeepEval evaluation framework.

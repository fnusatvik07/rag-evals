"""Metric factory functions for DeepEval and RAGAS."""

from __future__ import annotations


def get_deepeval_rag_metrics(
    model: str = "gpt-4o-mini",
    threshold: float = 0.7,
) -> dict:
    """Return a dict of standard DeepEval RAG metrics.

    Returns:
        Dict mapping metric name to metric instance.
    """
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )

    return {
        "faithfulness": FaithfulnessMetric(model=model, threshold=threshold),
        "answer_relevancy": AnswerRelevancyMetric(model=model, threshold=threshold),
        "ctx_precision": ContextualPrecisionMetric(model=model, threshold=threshold),
        "ctx_recall": ContextualRecallMetric(model=model, threshold=threshold),
        "ctx_relevancy": ContextualRelevancyMetric(model=model, threshold=threshold),
    }


def get_ragas_rag_metrics(model: str = "gpt-4o-mini") -> list:
    """Return a list of standard RAGAS RAG metrics.

    Returns:
        List of RAGAS metric instances.
    """
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    return [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]


def get_all_metrics(
    model: str = "gpt-4o-mini",
    threshold: float = 0.7,
) -> dict:
    """Return both DeepEval and RAGAS metrics.

    Returns:
        Dict with keys 'deepeval' (dict) and 'ragas' (list).
    """
    return {
        "deepeval": get_deepeval_rag_metrics(model=model, threshold=threshold),
        "ragas": get_ragas_rag_metrics(model=model),
    }

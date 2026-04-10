"""LLM generation for RAG pipelines."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for Acme Corp. "
    "Answer the customer's question based ONLY on the provided context documents. "
    "If the context does not contain enough information to answer the question, say so clearly. "
    "Do not make up information that is not supported by the context. "
    "Be concise but thorough in your response."
)


def generate(
    query: str,
    contexts: list[str],
    client=None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str:
    """Generate an answer to a query using retrieved contexts.

    Args:
        query: The user's question.
        contexts: List of context text strings.
        client: Optional pre-initialized OpenAI client.
        model: OpenAI model name.
        temperature: Sampling temperature.
        system_prompt: System prompt (uses DEFAULT_SYSTEM_PROMPT if None).
        max_tokens: Maximum tokens in the response.

    Returns:
        Generated answer string.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    context_str = "\n\n---\n\n".join(contexts)
    user_message = (
        f"Context Documents:\n---\n{context_str}\n---\n\n"
        f"Customer Question: {query}\n\n"
        f"Please provide a helpful answer based on the context above."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

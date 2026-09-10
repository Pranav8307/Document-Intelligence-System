import os
import time
from typing import List, Tuple
from openai import OpenAI
from utils.logger import get_logger
from utils.metrics import metrics_store

logger = get_logger(__name__)

def _get_config() -> tuple[bool, str, str, str | None]:
    use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("LLM_BASE_URL")
    return use_mock, provider, model, base_url


def _get_api_key(provider: str) -> str | None:
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _build_prompt(question: str, chunks: List[Tuple[str, float]]) -> str:
    """
    Token optimization strategy:
    - Only the top-k most relevant chunks are sent (not the full doc)
    - Each chunk is ≈125 tokens → top-3 = ~375 context tokens
    - System prompt is kept short and instructional
    - Nearby chunks are included so split lists remain readable
    - Answer is capped at max_tokens=600 to prevent mid-sentence truncation
    """
    context = "\n\n---\n\n".join([chunk for chunk, _ in chunks])
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Give a complete answer. Include all relevant items when the question asks for a list.
Do not stop mid-sentence or invent details.

Context:
{context}

Question: {question}

Answer:"""


def get_answer(question: str, chunks: List[Tuple[str, float]]) -> str:
    start = time.perf_counter()
    prompt = _build_prompt(question, chunks)
    use_mock, _, _, _ = _get_config()

    if use_mock:
        logger.info("Using mock LLM response")
        answer = _mock_llm(question, chunks)
        metrics_store.record_timing(
            "llm_answer_generation", (time.perf_counter() - start) * 1000
        )
        return answer

    answer = _call_openai(prompt)
    metrics_store.record_timing(
        "llm_answer_generation", (time.perf_counter() - start) * 1000
    )
    return answer


def _mock_llm(question: str, chunks: List[Tuple[str, float]]) -> str:
    top_chunk = chunks[0][0] if chunks else "No context available."
    return (
        f"[MOCK ANSWER] Based on the document, here is a relevant excerpt: "
        f'"{top_chunk[:200]}..." '
        f"(Set USE_MOCK_LLM=false and add OPENAI_API_KEY to use real LLM.)"
    )


def _call_openai(prompt: str) -> str:
    _, provider, model, base_url = _get_config()
    api_key = _get_api_key(provider)
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}'. "
            "Set GROQ_API_KEY for Groq or OPENAI_API_KEY for OpenAI."
        )

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The LLM returned an empty answer. Please try again.")
    return content.strip()

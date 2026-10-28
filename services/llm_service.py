import os
import time
from typing import List, Tuple
from openai import OpenAI
from utils.logger import get_logger
from utils.metrics import metrics_store

logger = get_logger(__name__)

def _get_config() -> tuple[bool, str, str, str | None]:
    use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("LLM_BASE_URL")
    return use_mock, provider, model, base_url


def _get_api_key(provider: str) -> str | None:
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _build_prompt(
    question: str, chunks: List[Tuple[str, float]], answer_length: str
) -> str:
    """
    The caller may provide the complete document for broad questions such as
    "list all projects". Keeping the context ordered by document position
    helps the model preserve headings, lists, and details split across chunks.
    """
    context = "\n\n---\n\n".join([chunk for chunk, _ in chunks])
    length_instruction = (
        "Give a focused answer in 3-6 bullets or short paragraphs."
        if answer_length == "concise"
        else "Give a thorough answer with complete sections and all relevant details."
    )
    return f"""You are a document-grounded resume assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

For broad questions about projects, work experience, education, skills, or achievements:
- Include EVERY matching item found in the context.
- Give each item its own heading or numbered section.
- Include all available details for each item, including the description, responsibilities,
  technologies, methods, metrics, and outcomes. Do not reduce an item to its title.
- Preserve technical names and numbers exactly as written in the document.
- If a detail is not provided, omit it rather than guessing.

{length_instruction} Do not stop mid-sentence or invent details.

Context:
{context}

Question: {question}

Answer:"""


def get_answer(
    question: str, chunks: List[Tuple[str, float]], answer_length: str = "detailed"
) -> str:
    start = time.perf_counter()
    prompt = _build_prompt(question, chunks, answer_length)
    use_mock, _, _, _ = _get_config()

    if use_mock:
        logger.info("Using mock LLM response")
        answer = _mock_llm(question, chunks)
        metrics_store.record_timing(
            "llm_answer_generation", (time.perf_counter() - start) * 1000
        )
        return answer

    answer = _call_openai(prompt, answer_length)
    metrics_store.record_timing(
        "llm_answer_generation", (time.perf_counter() - start) * 1000
    )
    return answer


def _mock_llm(question: str, chunks: List[Tuple[str, float]]) -> str:
    top_chunk = chunks[0][0] if chunks else "No context available."
    return (
        f"[MOCK ANSWER] Based on the document, here is a relevant excerpt: "
        f'"{top_chunk[:200]}..." '
        f"(Set USE_MOCK_LLM=false and add GROQ_API_KEY or OPENAI_API_KEY to use a real LLM.)"
    )


def _completion_options(provider: str, model: str, answer_length: str) -> dict:
    """Choose an output budget that leaves room for reasoning-model final text."""
    if provider == "groq":
        options = {
            "max_completion_tokens": 2048 if answer_length == "concise" else 4096
        }
        if model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            options.update(
                reasoning_effort="low" if answer_length == "concise" else "medium",
                include_reasoning=False,
            )
        return options

    return {"max_tokens": 800 if answer_length == "concise" else 4000}


def _call_openai(prompt: str, answer_length: str = "detailed") -> str:
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
        temperature=0.2,
        **_completion_options(provider, model, answer_length),
    )
    choices = getattr(response, "choices", None) or []
    content = choices[0].message.content if choices else None
    if not content or not content.strip():
        finish_reason = choices[0].finish_reason if choices else "no_choice"
        logger.warning(
            "LLM returned empty content | provider=%s model=%s answer_length=%s finish_reason=%s",
            provider,
            model,
            answer_length,
            finish_reason,
        )
        raise ValueError(
            "The LLM returned no visible answer. Please retry; the response may have "
            "used its output budget before producing an answer."
        )
    return content.strip()

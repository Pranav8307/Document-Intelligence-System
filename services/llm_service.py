import os
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

USE_MOCK = os.getenv("USE_MOCK_LLM", "true").lower() == "true"


def _build_prompt(question: str, chunks: List[Tuple[str, float]]) -> str:
    """
    Token optimization strategy:
    - Only the top-k most relevant chunks are sent (not the full doc)
    - Each chunk is ≈125 tokens → top-3 = ~375 context tokens
    - System prompt is kept short and instructional
    - Answer is capped at max_tokens=300 to control cost
    - In production: add token counter, trim chunks if total > budget
    """
    context = "\n\n---\n\n".join([chunk for chunk, _ in chunks])
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def get_answer(question: str, chunks: List[Tuple[str, float]]) -> str:
    prompt = _build_prompt(question, chunks)

    if USE_MOCK:
        logger.info("Using mock LLM response")
        return _mock_llm(question, chunks)

    return _call_openai(prompt)


def _mock_llm(question: str, chunks: List[Tuple[str, float]]) -> str:
    top_chunk = chunks[0][0] if chunks else "No context available."
    return (
        f"[MOCK ANSWER] Based on the document, here is a relevant excerpt: "
        f'"{top_chunk[:200]}..." '
        f"(Set USE_MOCK_LLM=false and add OPENAI_API_KEY to use real LLM.)"
    )


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
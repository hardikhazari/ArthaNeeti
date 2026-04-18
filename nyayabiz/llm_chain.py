# ─────────────────────────────────────────────────────────────────────────────
# LLM + PROMPT
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nyayabiz.config import LLM_ENDPOINT


# ── Module-level state (initialized by init_llm) ────────────────────────
_llm = None


def init_llm(endpoint: str = LLM_ENDPOINT):
    """Initialize the ChatDatabricks LLM.

    Must be called once before using the answer chain.
    """
    global _llm
    from databricks_langchain import ChatDatabricks

    _llm = ChatDatabricks(endpoint=endpoint, temperature=0.0)
    print(f"LLM initialized: {endpoint}")


def get_llm():
    """Return the initialized LLM instance."""
    return _llm


# ── Prompt templates ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are NyayaBiz, a specialised legal research assistant.

Rules you MUST follow:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. Cite every factual claim with the bracket marker [N] shown next to each clause.
   A claim without a citation is not allowed.
3. If the answer is not in the context, say exactly:
   "The provided sources do not contain information about this."
4. Quote exact legal wording when precision matters; otherwise paraphrase faithfully.
5. Do not offer legal opinions or strategy — only report what the clauses say.
6. End with a short "Sources:" line listing every [N] you cited.
"""

USER_TEMPLATE = """\
Context:
{context}

Question:
{question}

Answer (with bracketed citations):"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user",   USER_TEMPLATE),
])


def format_context(results: List[Dict]) -> str:
    if not results:
        return "(no relevant clauses found)"
    blocks = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r['source_file']} | {r['section_id']}"
        if r.get("page") is not None:
            header += f" | p.{r['page']}"
        if r["source"].startswith("xref:"):
            header += f" | via {r['source']}"
        blocks.append(f"{header}\n{r['raw_text']}")
    return "\n\n".join(blocks)

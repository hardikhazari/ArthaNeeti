# ─────────────────────────────────────────────────────────────────────────────
# HALLUCINATION CHECKER (LLM-as-Judge)
#
# How it works:
#   1. Takes the LLM answer + retrieved chunks from run_rag()
#   2. Sends them to the SAME Llama 3.3 70B as a "judge" with a strict prompt
#   3. The judge checks if every claim in the answer is supported by the chunks
#   4. Returns SUPPORTED / NOT_SUPPORTED with reasoning
#   5. If hallucinated → replaces answer with a safe fallback message
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nyayabiz.config import LLM_ENDPOINT
from nyayabiz.llm_chain import format_context
from nyayabiz.translation import translate_to_indic
from nyayabiz.pipeline import run_rag


# ── Module-level state (initialized by init_hallucination_checker) ───────
_hallucination_chain = None


HALLUCINATION_PROMPT = """\
You are a hallucination detection system for a legal research assistant.

Your job: compare the ANSWER against the SOURCE CHUNKS and determine if the answer is faithful to the sources.

Rules:
1.⁠ ⁠Every factual claim in the ANSWER must be directly supported by at least one SOURCE CHUNK.
2.⁠ ⁠If the answer paraphrases a source, that is acceptable as long as the meaning is preserved.
3.⁠ ⁠If the answer contains ANY claim that cannot be traced back to the source chunks, it is HALLUCINATED.
4.⁠ ⁠Citations like [1], [2] etc. in the answer should correspond to real content in the matching source chunk.
5.⁠ ⁠If the answer says "information is not available" or similar, that is NOT a hallucination.

Respond with EXACTLY this format (no extra text):

VERDICT: SUPPORTED
REASON: <one-line explanation>

OR

VERDICT: NOT_SUPPORTED
REASON: <one-line explanation of what was hallucinated>
"""

HALLUCINATION_USER = """\
SOURCE CHUNKS:
{context}

ANSWER TO VERIFY:
{answer}

Is every claim in the answer supported by the source chunks above?"""


hallucination_check_prompt = ChatPromptTemplate.from_messages([
    ("system", HALLUCINATION_PROMPT),
    ("user",   HALLUCINATION_USER),
])


def init_hallucination_checker(endpoint: str = LLM_ENDPOINT):
    """Initialize the hallucination judge LLM chain.

    Must be called once before verify_hallucination().
    """
    global _hallucination_chain
    from databricks_langchain import ChatDatabricks

    hallucination_judge = ChatDatabricks(endpoint=endpoint, temperature=0.0)
    _hallucination_chain = hallucination_check_prompt | hallucination_judge | StrOutputParser()
    print("✅ Hallucination checker ready. Use run_rag_verified() instead of run_rag().")


def verify_hallucination(english_answer: str, sources: list) -> dict:
    """
    Cross-check the LLM answer against retrieved chunks.

    Returns:
        {
            "is_supported": bool,
            "verdict": "SUPPORTED" | "NOT_SUPPORTED",
            "reason": str,
            "raw_judge_output": str  (full judge response for debugging)
        }
    """
    # Build the same context string the LLM saw
    context = str(format_context(sources))

    # Ask the judge
    judge_output = str(_hallucination_chain.invoke({
        "context": context,
        "answer":  english_answer,
    }))

    # Parse verdict
    verdict = "UNKNOWN"
    reason  = ""
    for line in judge_output.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    is_supported = ("SUPPORTED" in verdict and "NOT" not in verdict)

    return {
        "is_supported":     is_supported,
        "verdict":          verdict,
        "reason":           reason,
        "raw_judge_output": judge_output,
    }


# ── Safe fallback message (used when hallucination is detected) ──────────
FALLBACK_EN = (
    "I do not have enough context in the provided legal documents to answer "
    "this question accurately. Please try rephrasing your question or consult "
    "a qualified legal professional."
)


def run_rag_verified(
    query: str,
    initial_k: int = 15,
    final_k: int   = 5,
    resolve_xrefs: bool = True,
    multilingual: bool  = True,
) -> Dict:
    """
    Same as run_rag() but adds a hallucination check at the end.

    If the answer is NOT supported by the retrieved chunks:
      - english_answer is replaced with a safe fallback
      - answer (translated) is also replaced
      - out["hallucination_detected"] = True

    If supported:
      - original answer is kept
      - out["hallucination_detected"] = False
    """
    # Step 1: Run the normal RAG pipeline
    out = run_rag(
        query,
        initial_k=initial_k,
        final_k=final_k,
        resolve_xrefs=resolve_xrefs,
        multilingual=multilingual,
    )

    # Step 2: Verify hallucination
    print("🔍 Verifying answer against retrieved sources...")
    check = verify_hallucination(out["english_answer"], out["sources"])

    out["hallucination_check"] = check
    out["hallucination_detected"] = not check["is_supported"]

    if check["is_supported"]:
        print(f"✅ VERIFIED: {check['reason']}")
    else:
        print(f"⚠️  HALLUCINATION DETECTED: {check['reason']}")
        print("   → Replacing answer with safe fallback.")

        # Replace answer with fallback
        out["english_answer_original"] = out["english_answer"]
        out["answer_original"]         = out["answer"]
        out["english_answer"]          = FALLBACK_EN

        # Translate fallback to user's language if needed
        indic_code = out["detected_language"]
        if indic_code != "eng_Latn":
            out["answer"] = str(translate_to_indic(FALLBACK_EN, indic_code))
        else:
            out["answer"] = FALLBACK_EN

    return out

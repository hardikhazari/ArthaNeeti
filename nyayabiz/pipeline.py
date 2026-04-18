# ─────────────────────────────────────────────────────────────────────────────
# FULL RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict

from langchain_core.output_parsers import StrOutputParser

from nyayabiz.translation import detect_language, translate_to_english, translate_to_indic
from nyayabiz.retrieval import retrieve_legal
from nyayabiz.reranking import rerank_results
from nyayabiz.llm_chain import format_context, prompt, get_llm


def run_rag(
    query: str,
    initial_k: int = 15,
    final_k: int   = 5,
    resolve_xrefs: bool = True,
    multilingual: bool  = True,
) -> Dict:
    """
    End-to-end RAG:
      detect language → translate → retrieve → rerank → LLM → translate back
    """
    # 1. Language detection & optional translation
    indic_code   = detect_language(query) if multilingual else "eng_Latn"
    is_indic     = indic_code != "eng_Latn"
    english_query = translate_to_english(query, indic_code) if is_indic else query
    english_query = str(english_query)  # guard against TextAccessor from IndicTrans

    if is_indic:
        print(f"🔍 Detected: {indic_code}  →  English query: {english_query!r}")

    # 2. Retrieve
    raw_results  = retrieve_legal(english_query, k=initial_k, resolve_xrefs=resolve_xrefs)

    # 3. Rerank
    best_results = rerank_results(english_query, raw_results, top_n=final_k)

    # 4. Build context & run LLM — str() ensures no pandas objects reach the prompt
    context        = str(format_context(best_results))
    answer_chain   = prompt | get_llm() | StrOutputParser()
    english_answer = str(answer_chain.invoke({"context": context, "question": english_query}))

    # 5. Translate answer back if needed
    answer = translate_to_indic(english_answer, indic_code) if is_indic else english_answer
    answer = str(answer)  # guard final output too

    return {
        "original_query":    query,
        "detected_language": indic_code,
        "english_query":     english_query,
        "english_answer":    english_answer,
        "answer":            answer,
        "sources":           best_results,
    }

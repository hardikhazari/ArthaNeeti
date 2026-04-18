# ─────────────────────────────────────────────────────────────────────────────
# NyayaBiz — AI-Powered Legal Research Assistant
# ─────────────────────────────────────────────────────────────────────────────

from nyayabiz.config import (
    HF_TOKEN, VOLUME_PATH, DELTA_TABLE, VS_INDEX,
    LLM_ENDPOINT, RERANKER_EP,
    CHUNK_SIZE, CHUNK_OVERLAP,
    IN2EN_MODEL, EN2IN_MODEL,
    DEVICE,
)
from nyayabiz.pipeline import run_rag
from nyayabiz.hallucination import run_rag_verified
from nyayabiz.display import print_rag_result, print_verified_result

__all__ = [
    "run_rag",
    "run_rag_verified",
    "print_rag_result",
    "print_verified_result",
]

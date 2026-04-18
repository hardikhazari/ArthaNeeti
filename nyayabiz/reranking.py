# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ENCODER RERANKING (Local, no Databricks endpoint needed)
# Uses BAAI/bge-reranker-base — a 278M param cross-encoder trained on
# MS MARCO & NQ. Runs locally on GPU/CPU, no serving endpoint required.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List

from nyayabiz.config import DEVICE


# ── Module-level state (initialized by init_cross_encoder) ───────────────
_cross_encoder = None


def init_cross_encoder(model_name: str = "BAAI/bge-reranker-base"):
    """Load the cross-encoder reranker model.

    Must be called once before rerank_results().
    """
    global _cross_encoder
    from sentence_transformers import CrossEncoder

    print(f"Loading cross-encoder reranker on {DEVICE}...")
    _cross_encoder = CrossEncoder(
        model_name,
        max_length=512,
        device=DEVICE,
    )
    print("✅ Cross-encoder reranker ready.")


def rerank_results(query: str, results: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Cross-encoder reranking using BAAI/bge-reranker-base.

    How it works:
      1. Takes the query + each chunk as a (query, passage) pair
      2. Cross-encoder scores each pair jointly (not independently like bi-encoders)
      3. Sorts by cross-encoder score descending
      4. Returns top_n results

    This is more accurate than bi-encoder similarity because the cross-encoder
    sees query and passage tokens together with full attention.
    """
    if not results:
        return []

    # Build (query, passage) pairs for the cross-encoder
    pairs = [[query, r["text"]] for r in results]

    # Score all pairs in one batch (GPU-accelerated if available)
    scores = _cross_encoder.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    # Attach scores and sort
    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_n]

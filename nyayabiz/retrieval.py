# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL  (hybrid search + cross-reference stitching)
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import Dict, List, Optional

from nyayabiz.config import VS_INDEX


# ── Module-level state (initialized by init_vector_store) ────────────────
_vector_store = None


def init_vector_store(index_name: str = VS_INDEX):
    """Initialize the Databricks Vector Search store.

    Must be called once before retrieve_legal().
    """
    global _vector_store
    from databricks_langchain import DatabricksVectorSearch

    _vector_store = DatabricksVectorSearch(
        index_name=index_name,
        columns=["source_file", "section_id", "heading_chain", "xrefs", "page", "raw_text"],
    )
    print(f"Vector store initialized: {index_name}")


def _norm(s: str) -> str:
    return re.sub(r"[\s\.]+", "", s.lower()) if s else ""


def retrieve_legal(
    query: str,
    k: int = 15,
    filters: Optional[Dict] = None,
    resolve_xrefs: bool = True,
) -> List[Dict]:
    """Hybrid semantic+BM25 search, then stitch any referenced clauses."""
    hits = _vector_store.similarity_search_with_score(
        query=query, k=k, query_type="HYBRID", filter=filters,
    )

    def _row(doc, score, tag) -> Dict:
        # str() casts guard against pandas TextAccessor objects that
        # Databricks Vector Search occasionally returns for string columns.
        return {
            "score":         score,
            "source":        tag,
            "source_file":   str(doc.metadata.get("source_file") or ""),
            "section_id":    str(doc.metadata.get("section_id") or "root"),
            "heading_chain": str(doc.metadata.get("heading_chain") or ""),
            "xrefs":         str(doc.metadata.get("xrefs") or ""),
            "page":          doc.metadata.get("page"),
            "text":          str(doc.page_content),
            "raw_text":      str(doc.metadata.get("raw_text") or doc.page_content),
        }

    primary = [_row(d, s, "primary") for d, s in hits]
    if not resolve_xrefs:
        return primary

    seen    = {r["section_id"] for r in primary}
    pending = {
        ref.strip()
        for r in primary if r["xrefs"]
        for ref in r["xrefs"].split(",")
        if ref.strip()
    }

    referenced: List[Dict] = []
    for ref in pending:
        ref_norm = _norm(ref)
        if not ref_norm:
            continue
        for doc, score in _vector_store.similarity_search_with_score(query=ref, k=2, query_type="HYBRID"):
            sid    = doc.metadata.get("section_id", "")
            hchain = doc.metadata.get("heading_chain", "")
            if sid in seen:
                continue
            if ref_norm in _norm(sid) or ref_norm in _norm(hchain):
                seen.add(sid)
                referenced.append(_row(doc, score, f"xref:{ref}"))
                break

    return primary + referenced

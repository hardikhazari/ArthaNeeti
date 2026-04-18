# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict


def print_rag_result(out: Dict) -> None:
    print("\n─── NYAYABIZ RESPONSE " + "─" * 40)
    print(f"Language : {out['detected_language']}")
    if out["detected_language"] != "eng_Latn":
        print(f"EN Query : {out['english_query']}")
    print()
    print(out["answer"])
    print("\n─── SOURCES " + "─" * 50)
    for i, r in enumerate(out["sources"], 1):
        v = r.get("score", 0.0)
        rr = r.get("rerank_score", 0.0)
        page = f"  p.{r['page']}" if r.get("page") is not None else ""
        print(f"[{i}] {r['source_file']} | {r['section_id']}{page}")
        print(f"     Vector={v:.3f}  Rerank={rr:.3f}  ({r['source'].upper()})")


def print_verified_result(out: Dict) -> None:
    """Extended print that shows hallucination check status."""
    print_rag_result(out)

    check = out.get("hallucination_check", {})
    print("\n─── HALLUCINATION CHECK " + "─" * 37)
    if out.get("hallucination_detected"):
        print(f"⚠️  VERDICT  : {check.get('verdict', 'N/A')}")
        print(f"   REASON   : {check.get('reason', 'N/A')}")
        print(f"   ACTION   : Answer replaced with safe fallback.")
    else:
        print(f"✅ VERDICT  : {check.get('verdict', 'N/A')}")
        print(f"   REASON   : {check.get('reason', 'N/A')}")

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nyayabiz.config import CHUNK_SIZE, CHUNK_OVERLAP


# 4a. Cross-reference extraction
_XREF_PATTERNS = [
    r"§\s*\d+(?:\([a-z0-9]+\))*",
    r"\b(?:Article|Art\.)\s*\d+(?:\.\d+)*",
    r"\b(?:Section|Sec\.)\s*\d+(?:\.\d+)*",
    r"\b(?:Clause|Cl\.)\s*\d+",
    r"(?:¶|Paragraph\s*|Para\.\s*)\d+",
    r"\bChapter\s*\d+",
    r"\bRule\s*\d+(?:\.\d+)*",
]
_XREF_RE = re.compile("|".join(_XREF_PATTERNS), flags=re.IGNORECASE)


def _extract_xrefs(text: str) -> List[str]:
    seen, out = set(), []
    for m in _XREF_RE.finditer(text):
        ref = re.sub(r"\s+", " ", m.group(0)).strip()
        if ref.lower() not in seen:
            seen.add(ref.lower())
            out.append(ref)
    return out


# 4b. Heading detection
_HEADING_PATTERNS: List[Tuple[int, re.Pattern]] = [
    (1, re.compile(r"^\s*(?:PART|Part|TITLE|Title)\s+([IVXLCDM]+|\d+)[^\n]*",  re.MULTILINE)),
    (2, re.compile(r"^\s*(?:CHAPTER|Chapter)\s+([IVXLCDM]+|\d+)[^\n]*",        re.MULTILINE)),
    (3, re.compile(r"^\s*(?:ARTICLE|Article|Art\.)\s+(\d+(?:\.\d+))[^\n]",     re.MULTILINE)),
    (3, re.compile(r"^\s*(?:SECTION|Section|Sec\.)\s+(\d+(?:\.\d+))[^\n]",     re.MULTILINE)),
    (4, re.compile(r"^\s*§\s*(\d+(?:\([a-z0-9]+\)))[^\n]",                     re.MULTILINE)),
    (4, re.compile(r"^\s*(?:CLAUSE|Clause|Cl\.)\s+(\d+)[^\n]*",                re.MULTILINE)),
    (4, re.compile(r"^\s*(?:RULE|Rule)\s+(\d+(?:\.\d+))[^\n]",                 re.MULTILINE)),
]


def _detect_headings(text: str) -> List[Tuple[int, int, int, str]]:
    """Return [(start, end, level, label)] sorted by offset, deduplicated."""
    found = []
    for level, pat in _HEADING_PATTERNS:
        for m in pat.finditer(text):
            label = re.sub(r"\s+", " ", m.group(0)).strip()
            found.append((m.start(), m.end(), level, label))
    found.sort(key=lambda x: (x[0], -x[1]))

    deduped, prev_end = [], -1
    for start, end, level, label in found:
        if start < prev_end:
            continue
        deduped.append((start, end, level, label))
        prev_end = end
    return deduped


def _update_stack(stack: Dict[int, str], level: int, label: str) -> Dict[int, str]:
    new_stack = {k: v for k, v in stack.items() if k < level}
    new_stack[level] = label
    return new_stack


def _chain(stack: Dict[int, str]) -> List[str]:
    return [stack[k] for k in sorted(stack)]


# 4c. Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
)

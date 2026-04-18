# ─────────────────────────────────────────────────────────────────────────────
# PDF INGESTION → DELTA TABLE
# Run once to populate the vector index source table.
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
from typing import Dict, List, Tuple

import pandas as pd
from pypdf import PdfReader

from nyayabiz.config import VOLUME_PATH, DELTA_TABLE, CHUNK_SIZE
from nyayabiz.chunking import (
    _detect_headings,
    _extract_xrefs,
    _update_stack,
    _chain,
    text_splitter,
)


def ingest_pdfs(spark, volume_path: str = VOLUME_PATH, table_name: str = DELTA_TABLE) -> int:
    """Parse all PDFs in the volume, chunk them, and write to Delta.

    Args:
        spark: The active SparkSession (passed from the Databricks notebook context).
        volume_path: Path to the DBFS/Volume directory containing PDFs.
        table_name: Fully qualified Delta table name.

    Returns:
        Number of chunks written.
    """
    all_chunks = []
    print(f"Scanning {volume_path} …")

    for file in os.listdir(volume_path):
        if not file.endswith(".pdf"):
            continue
        full_path = os.path.join(volume_path, file)
        print(f"  Processing: {file}")

        try:
            reader = PdfReader(full_path)
            heading_stack: Dict[int, str] = {}

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                headings = _detect_headings(page_text)
                segments: List[Tuple[List[str], str]] = []

                if not headings:
                    segments.append((_chain(heading_stack), page_text.strip()))
                else:
                    first_start = headings[0][0]
                    if first_start > 0:
                        pre = page_text[:first_start].strip()
                        if pre:
                            segments.append((_chain(heading_stack), pre))

                    for i, (_, end, level, label) in enumerate(headings):
                        heading_stack = _update_stack(heading_stack, level, label)
                        body_end = headings[i + 1][0] if i + 1 < len(headings) else len(page_text)
                        body = page_text[end:body_end].strip()
                        if body:
                            segments.append((_chain(heading_stack), body))

                for hchain, body in segments:
                    section_id     = " > ".join(hchain) if hchain else "root"
                    heading_prefix = ("\n".join(hchain) + "\n\n") if hchain else ""
                    pieces = [body] if len(body) <= CHUNK_SIZE else text_splitter.split_text(body)

                    for piece in pieces:
                        all_chunks.append({
                            "id":            str(uuid.uuid4()),
                            "source_file":   file,
                            "text":          heading_prefix + piece,
                            "raw_text":      piece,
                            "section_id":    section_id,
                            "heading_chain": " > ".join(hchain),
                            "xrefs":         ", ".join(_extract_xrefs(piece)),
                            "page":          page_num,
                        })

            print(f"    ✓ {len(reader.pages)} pages")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    n = len(all_chunks)
    print(f"\nTotal chunks: {n}")

    pdf_df    = pd.DataFrame(all_chunks)
    spark_df  = spark.createDataFrame(pdf_df)

    spark_df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(table_name)

    spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"Saved to {table_name}. Go to the UI → Vector Index → 'Sync Now'.")
    return n

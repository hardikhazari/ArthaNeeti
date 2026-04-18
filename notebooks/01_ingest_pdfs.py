# Databricks notebook source
# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 — ONE-TIME PDF INGESTION → DELTA TABLE
#
# Run this notebook once (or whenever you add new PDFs to the volume).
# After completion, go to the Databricks UI → Vector Index → 'Sync Now'.
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

from nyayabiz.ingestion import ingest_pdfs

# COMMAND ----------

# `spark` is the SparkSession provided by Databricks notebook context.
# Pass it explicitly to the ingestion function.
num_chunks = ingest_pdfs(spark)
print(f"\n✅ Ingestion complete — {num_chunks} chunks written.")

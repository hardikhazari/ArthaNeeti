# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these values as needed)
# ─────────────────────────────────────────────────────────────────────────────

import torch

# Secrets — prefer Databricks Secrets in production, never hardcode tokens
HF_TOKEN        = "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"
VOLUME_PATH     = "/Volumes/workspace/default/data/final_data/"
DELTA_TABLE     = "workspace.default.kartik_vector"
VS_INDEX        = "workspace.default.final_vector"
LLM_ENDPOINT    = "databricks-meta-llama-3-3-70b-instruct"
RERANKER_EP     = "databricks-bge-reranker-large-en"

# Chunking
CHUNK_SIZE      = 3_000
CHUNK_OVERLAP   = 200

# Translation models
IN2EN_MODEL     = "ai4bharat/indictrans2-indic-en-dist-200M"
EN2IN_MODEL     = "ai4bharat/indictrans2-en-indic-dist-200M"

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Databricks notebook source
# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 0 — INSTALL DEPENDENCIES
# Run this once on a new cluster, then restart Python.
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

# Core LangChain + PDF + Data
%pip install langchain langchain-community langchain-text-splitters pypdf pandas

# Databricks integrations
%pip install databricks-vectorsearch databricks-langchain

# Translation models (IndicTrans2)
%pip install transformers==4.38.2 torch sentencepiece sacremoses langdetect

# Reranking
%pip install sentence-transformers

# IndicTransToolkit (from source)
%pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

# COMMAND ----------

# Audio (Whisper) — optional, only needed for voice queries
%pip install openai-whisper ffmpeg-python

# COMMAND ----------

# Gradio UI — optional, only needed for the web interface
%pip install gradio

# COMMAND ----------

# Restart Python to pick up all installed packages
dbutils.library.restartPython()

# Databricks notebook source
# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 — MAIN APPLICATION ENTRY POINT
#
# This notebook initializes all models and launches the app.
# Run cells in order. Each section can be run independently after the first.
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

# ── 1. Initialize Translation Models ────────────────────────────────────────
from nyayabiz.translation import load_translation_models
load_translation_models()

# COMMAND ----------

# ── 2. Initialize Vector Store ──────────────────────────────────────────────
from nyayabiz.retrieval import init_vector_store
init_vector_store()

# COMMAND ----------

# ── 3. Initialize Cross-Encoder Reranker ────────────────────────────────────
from nyayabiz.reranking import init_cross_encoder
init_cross_encoder()

# COMMAND ----------

# ── 4. Initialize LLM ──────────────────────────────────────────────────────
from nyayabiz.llm_chain import init_llm
init_llm()

# COMMAND ----------

# ── 5. Initialize Hallucination Checker ─────────────────────────────────────
from nyayabiz.hallucination import init_hallucination_checker
init_hallucination_checker()

# COMMAND ----------

# ── 6. Run a Test Query (text) ──────────────────────────────────────────────
from nyayabiz.hallucination import run_rag_verified
from nyayabiz.display import print_verified_result

query = "What are the penalties for violating food safety labelling requirements"
out = run_rag_verified(query, initial_k=15, final_k=5, resolve_xrefs=True, multilingual=True)
print_verified_result(out)

# COMMAND ----------

# ── 7. Launch Voice Query Widget (ipywidgets) ──────────────────────────────
# NOTE: You must define run_rag_from_audio() below (requires Whisper).
#
# import whisper
# whisper_model = whisper.load_model("base")
#
# def run_rag_from_audio(audio_path):
#     result = whisper_model.transcribe(audio_path)
#     question = result["text"]
#     language = result.get("language", "unknown")
#     out = run_rag_verified(question, multilingual=True)
#     out["question"] = question
#     out["whisper_language"] = language
#     return out
#
# from nyayabiz.ui.voice_widget import launch_voice_widget
# from nyayabiz.display import print_rag_result
# launch_voice_widget(run_rag_from_audio, print_rag_result)

# COMMAND ----------

# ── 8. Launch Gradio Web UI ─────────────────────────────────────────────────
from nyayabiz.ui.gradio_app import launch_gradio
launch_gradio(share=True, server_port=8000)

# Databricks notebook source
# MAGIC %md
# MAGIC # ArthaNeeti: Enterprise Legal Intelligence ⚖️
# MAGIC 
# MAGIC This notebook serves as the main entry point for the ArthaNeeti RAG system.
# MAGIC It utilizes a modular Python backend for production-grade scalability.

# COMMAND ----------
# 1. SETUP: Add src to path and import orchestrator
import sys
import os

# Assuming the repository is cloned to /Workspace/Repos/<user>/ArthaNeeti
# We add the root directory to sys.path to allow imports from /src
repo_root = os.path.abspath("..") 
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.pipeline.orchestrator import ArthaNeetiOrchestrator
from src.ingestion.load_data import run_full_ingestion

# COMMAND ----------
# 2. CONFIGURATION (Optional: Run Ingestion)
# Uncomment the line below to run the full Medallion pipeline (Bronze -> Silver -> Gold)
# run_full_ingestion("../configs/settings.yaml")

# COMMAND ----------
# 3. INITIALIZE ORCHESTRATOR
orchestrator = ArthaNeetiOrchestrator("../configs/settings.yaml")

# COMMAND ----------
# 4. DEFINE UI HANDLERS
def run_query_modular(question: str):
    """
    Called by the UI to execute the full pipeline.
    """
    try:
        result = orchestrator.run(question)
        # Prepare for JS consumption
        serialised_docs = [
            {
                "page_content": d.page_content,
                "metadata": d.metadata
            } for d in result["docs"]
        ]
        return {
            "answer": result["answer"],
            "docs": serialised_docs,
            "category": result["category"],
            "is_grounded": result.get("is_grounded", True)
        }
    except Exception as e:
        return {"error": str(e)}

# COMMAND ----------
# 5. REGISTER IPYTHON COMM
from IPython import get_ipython
ip = get_ipython()

def handle_query_comm(comm, open_msg):
    @comm.on_msg
    def _recv(msg):
        data = msg["content"]["data"]
        question = data.get("question", "")
        result = run_query_modular(question)
        comm.send(result)

try:
    ip.kernel.comm_manager.register_target("nyayabiz_query", handle_query_comm)
    print("✅ Comm handler registered successfully.")
except:
    print("⚠️ Comm handler already registered or unavailable.")

# COMMAND ----------
# 6. RENDER INTERACTIVE UI
# We reuse the HTML template logic here (omitted for brevity in this script, 
# but in the real notebook, we'd load it from a file or keep it in a variable)

from IPython.display import display, HTML

# Load HTML from a consolidated location or local variable
HTML_UI = """
<!-- Omitted full HTML for brevity, same as in ArthaNeeti.ipynb -->
<div style="background: #0d1b2a; color: #e8e4dc; padding: 20px; border-radius: 8px;">
    <h2>ArthaNeeti Modular Interface</h2>
    <p>The system is now running on the modular Python backend.</p>
</div>
"""
# In production, we'd use displayHTML() or similar
print("UI Launched. Use the dashboard below to interact with ArthaNeeti.")

# ArthaNeeti: Enterprise-Grade Legal RAG for Indian Jurisprudence

[![Tech Stack](https://img.shields.io/badge/Stack-Databricks%20|%20LangChain%20|%20Gemini-blue)](https://github.com/hardikhazari/ArthaNeeti)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Stage%20RAG-green)]()
[![Precision](https://img.shields.io/badge/Metric-High%20Groundedness-orange)]()

**ArthaNeeti** is a sophisticated legal intelligence platform designed to navigate the intricate regulatory landscape of India. By leveraging state-of-the-art Retrieval-Augmented Generation (RAG) and high-performance computation on Databricks, ArthaNeeti provides businesses with contextually grounded, hallucination-free legal consultation across thousands of central and state-level regulations.

---

## 🎯 The Problem: India's Regulatory Maze

Industrial and commercial operations in India are governed by a hyper-complex web of laws:
* **1,500+ Central Laws** and an even greater number of state-level amendments.
* **25,000+ Compliances** across various industrial sectors.
* **3,000+ Business-Specific Regulations** frequently updated by various ministries.

**Critical Pain Points:**
- **Legal Ambiguity:** High overlap between sectoral and general legislations.
- **Dynamic Environments:** Frequent changes in compliance deadlines (e.g., license renewals).
- **Consultation Costs:** Prohibitive legal fees for SMEs and growing startups.
- **LLM Hallucinations:** Generic models (ChatGPT, etc.) often "hallucinate" legal sections or cite repealed laws, creating significant liability risks.

---

## 💡 The Solution: Multistage Neural Retrieval

ArthaNeeti solves these challenges through a **highly advanced RAG pipeline** that goes beyond simple vector search. We implement a modular, scalable architecture designed for high-stakes legal reasoning.

### Heterogeneous Data Pipeline
The system ingests and indexes high-fidelity legal corpora, including:
- **FSNI Data**: Comprehensive Financial, Sectoral, and National Indicators.
- **FSSAI Compliance**: Granular food safety and standards regulations.
- **RERA**: Real Estate Regulatory Authority acts and development guidelines.
- **Agricultural Legislations**: Complex land and harvest regulatory frameworks.
- **Startup Policy Guidebooks**: Navigating the "Startup India" legal ecosystem.

---

## 🧠 Technical Architecture

ArthaNeeti implements a **five-layer intelligence stack** to ensure precision and reliability.

### 1. Hierarchical Parsing & Semantic Indexing
Traditional chunking breaks legal context. ArthaNeeti uses a **Structural Awareness Parser**:
- **Heading Stack Maintenance**: Chunks preserve their hierarchy (e.g., *Part II > Chapter IV > Section 12*).
- **Cross-Reference Extraction**: Automatically identifies internal references like "Subject to Section 4..." to trigger recursive retrieval.

### 2. Multi-Stage Hybrid Retrieval
We avoid retrieval misses by combining search strategies:
- **Dense Retrieval**: Semantic search for conceptual queries.
- **Sparse Retrieval (BM25)**: Exact matching for legal section IDs and specific terminology.
- **Recursive Stitching**: If a primary chunk references another section, the system "stitches" the referenced text into the context window dynamically.

### 3. Neural Re-ranking
Retrieved documents are re-evaluated using a **Cross-Encoder Reranker** (BGE-large). This ensures that only the most contextually relevant clauses—not just those with high keyword overlap—are passed to the LLM.

### 4. Grounded Generation via Gemini
We utilize **Google Gemini** for the final reasoning layer due to its:
- **Extended Context Window**: Ability to ingest massive legal appendices.
- **Sophisticated Reasoning**: Superior handling of complex "if-then" legal logic.
- **Contextual Grounding**: Implementation of strict system prompts to prevent external knowledge leakage.

---

## 📊 Rigorous Evaluation

Our system is evaluated against a "Gold Standard" dataset of domestic legal queries using industry-standard RAG metrics:

| Metric | Methodology | Objective |
| :--- | :--- | :--- |
| **Groundedness** | LLM-as-a-judge comparison | Ensures every claim is present in the source. |
| **Faithfulness** | NLI-based verification | Measures answer consistency with context. |
| **Precision@K** | Retrieval quality audit | Percentage of top-K results that are relevant. |
| **Answer Relevance** | Semantic similarity | Alignment between query intent and response. |

---

## ⚙️ Setup & Execution

### Prerequisites
- Python 3.10+
- Databricks Workspace (Runtime 15.1+)
- Gemini API Key / Databricks Model Serving Endpoint

### Installation
```bash
# Clone the repository
git clone https://github.com/hardikhazari/ArthaNeeti.git
cd ArthaNeeti

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install core dependencies
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the root directory:
```env
DATABRICKS_HOST="your-workspace-url"
DATABRICKS_TOKEN="your-access-token"
GEMINI_API_KEY="your-gemini-key"
VECTOR_SEARCH_ENDPOINT_NAME="legal-vector-search"
```

### Running the Pipeline
```python
from arthaneeti.pipeline import ArthaNeetiRAG

# Initialize the engine
rag = ArthaNeetiRAG(endpoint="databricks-llama-3-3-70b") # or Gemini

# Execute a legal query
response = rag.query("What are the FSSAI compliance requirements for cloud kitchens?")
print(response['answer'])
```

---

## 🚀 Future Roadmap
- [ ] **State-Level Knowledge Graphs**: Visualizing dependencies between central and state laws.
- [ ] **Multi-Agent Compliance Auditing**: Autonomous agents that scan business docs for violations.
- [ ] **Localized LLM Adapters**: Fine-tuning on specific Indian regional law corpora.

---
*Built with precision for the future of Indian Jurisprudence.*

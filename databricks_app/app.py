"""
NyayaBiz Legal Advisor — Databricks App (Dash)
===============================================

A professional legal research assistant powered by:
  - Databricks Vector Search (hybrid retrieval)
  - Meta Llama 3.3 70B (via Databricks serving endpoint)
  - SQL Warehouse (for structured queries)

Deployment:
  1. Place this file + app.yaml + requirements.txt in a workspace folder
  2. Create a new Databricks App → select "Custom"
  3. Deploy from the workspace folder
  4. Add SQL Warehouse resource (key: "sql_warehouse") in App Settings
  5. Redeploy
"""

import os
import logging

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nyayabiz")

VECTOR_SEARCH_INDEX = os.getenv("VECTOR_SEARCH_INDEX", "workspace.default.final_vector")
LLM_ENDPOINT        = os.getenv("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
APP_PORT             = int(os.getenv("DATABRICKS_APP_PORT", "8000"))

# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION — Bootstrap from Databricks App runtime
# ─────────────────────────────────────────────────────────────────────────────
# Databricks Apps use OAuth (not PAT tokens), so w.config.token is None.
# We extract a bearer token from the SDK's auth header factory and propagate
# it via env vars so the databricks-vectorsearch SDK can authenticate.

w = WorkspaceClient()
host = w.config.host

# Try PAT first, fall back to extracting OAuth bearer token from SDK headers
token = w.config.token
if not token:
    try:
        headers = w.config.authenticate()
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    except Exception as e:
        logger.warning(f"Could not extract OAuth token: {e}")

if not token:
    raise RuntimeError(
        "Could not obtain auth token. Ensure the Databricks App has a "
        "Service Principal configured and the 'sql_warehouse' resource "
        "is added in App Settings."
    )

os.environ["DATABRICKS_HOST"]  = host
os.environ["DATABRICKS_TOKEN"] = token
logger.info(f"Authenticated to: {host}")


# ─────────────────────────────────────────────────────────────────────────────
# RAG COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

# Vector store — create client with explicit auth to avoid SDK auto-detect issues
vs_client = VectorSearchClient(
    workspace_url=host,
    personal_access_token=token,
    disable_notice=True,
)

vector_store = DatabricksVectorSearch(
    index_name=VECTOR_SEARCH_INDEX,
    columns=["source_file", "section_id", "heading_chain", "xrefs", "page", "raw_text"],
)

# LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)

# Prompt template
TEMPLATE = """\
You are NyayaBiz Legal Advisor, an expert AI assistant specialising in
regulatory and legal analysis.

Your role:
- Provide accurate, concise legal guidance based on the provided context
- Cite specific sections, articles, or clauses when available
- If information is not in the context, clearly state:
  "This information is not available in the provided documents"
- Use professional legal terminology while remaining accessible
- Highlight key compliance requirements and obligations

Context from Legal Documents:
{context}

User Question:
{question}

Professional Legal Analysis:"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def query_legal_advisor(question: str, num_sources: int = 5, temperature: float = 0.1):
    """
    End-to-end RAG query:
      retrieve → format context → LLM → return (answer, sources)
    """
    if not question or not question.strip():
        return "⚠️ Please enter a legal question.", ""

    try:
        # Update temperature
        llm.temperature = temperature

        # Retrieve documents
        retriever = vector_store.as_retriever(
            search_kwargs={"k": num_sources, "query_type": "HYBRID"}
        )
        docs = retriever.invoke(question)

        # Build context string
        context = "\n\n".join(doc.page_content for doc in docs)

        # Run the RAG chain
        rag_chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(question)

        # Format sources for display
        sources_items = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source_file = meta.get("source_file", "Unknown")
            section     = meta.get("section_id", "N/A")
            page        = meta.get("page", "N/A")
            heading     = meta.get("heading_chain", "")
            text_preview = doc.page_content[:400] + "…" if len(doc.page_content) > 400 else doc.page_content

            heading_line = f" — {heading}" if heading and heading != "N/A" else ""

            sources_items.append(
                html.Div(className="source-card", children=[
                    html.Div(className="source-header", children=[
                        html.Span(f"[{i}]", className="source-badge"),
                        html.Span(f" {source_file}", className="source-file"),
                    ]),
                    html.Div(className="source-meta", children=[
                        html.Span(f"Section: {section}"),
                        html.Span(f"  |  Page: {page}{heading_line}"),
                    ]),
                    html.Pre(text_preview, className="source-text"),
                ])
            )

        return answer, sources_items

    except Exception as e:
        logger.exception("Query failed")
        return f"❌ Error: {str(e)}\n\nEnsure the vector index is synced and the SQL Warehouse resource is added.", ""


# ─────────────────────────────────────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="NyayaBiz Legal Advisor",
    update_title="⏳ Analysing…",
    suppress_callback_exceptions=True,
)

# For Databricks App deployment
server = app.server

# ── Example queries ──────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "What are the compliance requirements for food safety?",
    "What regulations apply to worker health and safety?",
    "What are the data privacy requirements under Indian law?",
    "What penalties exist for environmental regulation violations?",
    "What are the licensing requirements for medical facilities?",
]


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div(className="app-container", children=[

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div(className="header", children=[
        html.Div(className="header-seal", children="⚖️"),
        html.Div(className="header-text", children=[
            html.H1("NyayaBiz Legal Advisor"),
            html.P("AI-Powered Regulatory & Compliance Intelligence"),
        ]),
        html.Div(className="header-badges", children=[
            html.Span("● System Online", className="badge badge-green"),
            html.Span("Llama 3.3 · 70B", className="badge badge-gold"),
            html.Span("Databricks VSS · Hybrid", className="badge badge-gold"),
        ]),
    ]),

    # ── Main content ────────────────────────────────────────────────────────
    html.Div(className="main-content", children=[

        # ── Left panel: Input ───────────────────────────────────────────────
        html.Div(className="input-panel panel", children=[

            html.Label("LEGAL QUERY", className="panel-label"),
            dcc.Textarea(
                id="query-input",
                placeholder="State your legal question or compliance concern…",
                className="query-textarea",
            ),

            # Example queries
            html.Div(className="examples-section", children=[
                html.Label("PRECEDENT QUERIES", className="panel-label"),
                html.Div(className="examples-grid", children=[
                    html.Button(
                        q, id={"type": "example-btn", "index": i},
                        className="example-btn", n_clicks=0,
                    )
                    for i, q in enumerate(EXAMPLE_QUERIES)
                ]),
            ]),

            # Parameters
            html.Div(className="params-section", children=[
                html.Label("⚙  ANALYSIS PARAMETERS", className="panel-label"),

                html.Div(className="param-row", children=[
                    html.Label("Source Documents:", className="param-label"),
                    dcc.Slider(
                        id="num-sources-slider",
                        min=3, max=10, step=1, value=5,
                        marks={3: "3", 5: "5", 7: "7", 10: "10"},
                        className="param-slider",
                    ),
                ]),

                html.Div(className="param-row", children=[
                    html.Label("Temperature:", className="param-label"),
                    dcc.Slider(
                        id="temperature-slider",
                        min=0.0, max=1.0, step=0.05, value=0.1,
                        marks={0: "0.0", 0.5: "0.5", 1.0: "1.0"},
                        className="param-slider",
                    ),
                ]),
            ]),

            # Buttons
            html.Div(className="button-row", children=[
                html.Button("⚖  Analyse", id="submit-btn", className="btn-primary", n_clicks=0),
                html.Button("✕  Clear", id="clear-btn", className="btn-secondary", n_clicks=0),
            ]),
        ]),

        # ── Right panel: Output ─────────────────────────────────────────────
        html.Div(className="output-panel", children=[

            # Tabs
            dcc.Tabs(id="output-tabs", value="analysis", className="output-tabs", children=[
                dcc.Tab(label="💡 Legal Analysis", value="analysis", className="output-tab", selected_className="output-tab-selected"),
                dcc.Tab(label="📖 Source Documents", value="sources", className="output-tab", selected_className="output-tab-selected"),
            ]),

            html.Div(id="tab-content", className="tab-content panel", children=[
                html.P("Submit a query above to receive a professional legal analysis.",
                       className="placeholder-text"),
            ]),
        ]),
    ]),

    # ── Loading overlay ─────────────────────────────────────────────────────
    dcc.Loading(
        id="loading-indicator",
        type="circle",
        color="#c9a84c",
        children=html.Div(id="loading-target"),
    ),

    # ── Hidden stores ───────────────────────────────────────────────────────
    dcc.Store(id="answer-store", data=""),
    dcc.Store(id="sources-store", data=""),

    # ── Disclaimer ──────────────────────────────────────────────────────────
    html.Div(className="disclaimer", children=[
        html.Span("⚠"),
        html.Div(children=[
            html.Strong("Disclaimer: "),
            "This AI assistant provides informational guidance only and does not "
            "constitute formal legal advice. Always consult a qualified legal "
            "professional before taking action. · Powered by Databricks Vector "
            "Search + Meta Llama 3.3 70B.",
        ]),
    ]),
])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# Example query buttons → fill the textarea
@callback(
    Output("query-input", "value", allow_duplicate=True),
    [Input({"type": "example-btn", "index": i}, "n_clicks") for i in range(len(EXAMPLE_QUERIES))],
    prevent_initial_call=True,
)
def fill_example(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    prop_id = ctx.triggered[0]["prop_id"]
    try:
        import json
        btn_info = json.loads(prop_id.split(".")[0])
        idx = btn_info["index"]
        return EXAMPLE_QUERIES[idx]
    except Exception:
        return no_update


# Submit query → run RAG pipeline
@callback(
    Output("answer-store", "data"),
    Output("sources-store", "data"),
    Output("loading-target", "children"),
    Input("submit-btn", "n_clicks"),
    State("query-input", "value"),
    State("num-sources-slider", "value"),
    State("temperature-slider", "value"),
    prevent_initial_call=True,
)
def on_submit(n_clicks, question, num_sources, temperature):
    if not n_clicks or not question:
        return no_update, no_update, no_update

    answer, sources = query_legal_advisor(question, num_sources, temperature)

    # Convert sources (list of Dash components) to a serializable marker
    # We'll render them fresh in the tab callback
    return answer, "has_results", ""


# Clear button
@callback(
    Output("query-input", "value"),
    Output("answer-store", "data", allow_duplicate=True),
    Output("sources-store", "data", allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_clear(n_clicks):
    return "", "", ""


# Tab content rendering
@callback(
    Output("tab-content", "children"),
    Input("output-tabs", "value"),
    Input("answer-store", "data"),
    Input("sources-store", "data"),
    State("query-input", "value"),
    State("num-sources-slider", "value"),
    State("temperature-slider", "value"),
)
def render_tab(tab, answer, sources_flag, question, num_sources, temperature):
    if not answer:
        return html.P(
            "Submit a query above to receive a professional legal analysis.",
            className="placeholder-text",
        )

    if tab == "analysis":
        return html.Div(className="analysis-content", children=[
            html.H3("⚖️ Legal Analysis", className="analysis-title"),
            dcc.Markdown(answer, className="analysis-text"),
        ])

    elif tab == "sources":
        # Re-query to get source documents for display
        if question:
            _, sources = query_legal_advisor(question, num_sources, temperature)
            if isinstance(sources, list) and sources:
                return html.Div(className="sources-content", children=[
                    html.H3("📚 Source Documents", className="analysis-title"),
                    *sources,
                ])
        return html.P("No sources available.", className="placeholder-text")

    return no_update


# ─────────────────────────────────────────────────────────────────────────────
# CSS (embedded — no external files needed)
# ─────────────────────────────────────────────────────────────────────────────

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap" rel="stylesheet">
    <style>
        /* ── Root palette ── */
        :root {
            --navy:       #0d1b2a;
            --navy-mid:   #1a2e44;
            --navy-light: #243b55;
            --gold:       #c9a84c;
            --gold-light: #e8c96e;
            --cream:      #f5f0e8;
            --text:       #e8e4dc;
            --muted:      #8a8070;
            --border:     rgba(201,168,76,0.22);
            --red:        #c0392b;
        }

        /* ── Global ── */
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--navy);
            color: var(--text);
            min-height: 100vh;
        }

        .app-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 32px;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--navy); }
        ::-webkit-scrollbar-thumb { background: var(--gold); border-radius: 3px; }

        /* ════════════════════════════════
           HEADER
        ════════════════════════════════ */
        .header {
            background: linear-gradient(135deg, var(--navy-mid) 0%, var(--navy-light) 100%);
            border: 1px solid var(--border);
            border-top: 3px solid var(--gold);
            border-radius: 4px;
            padding: 28px 36px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.45);
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .header-seal {
            width: 60px; height: 60px;
            background: radial-gradient(circle at 38% 38%, #e8c96e, #c9a84c, #9a7a2c);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 26px;
            box-shadow: 0 0 0 3px var(--navy-mid), 0 0 0 5px var(--gold),
                        inset 0 2px 8px rgba(0,0,0,0.3);
            flex-shrink: 0;
        }

        .header h1 {
            font-family: 'Playfair Display', serif;
            font-size: 26px;
            font-weight: 700;
            color: var(--gold-light);
            letter-spacing: 0.02em;
            margin: 0 0 4px;
        }

        .header p {
            font-size: 12px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 0;
            font-weight: 300;
        }

        .header-badges {
            margin-left: auto;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 5px;
        }

        .badge {
            font-size: 10px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 3px 10px;
            border-radius: 2px;
            font-weight: 500;
        }

        .badge-green {
            background: rgba(46,204,113,0.1);
            color: #2ecc71;
            border: 1px solid rgba(46,204,113,0.25);
        }

        .badge-gold {
            background: rgba(201,168,76,0.1);
            color: var(--gold);
            border: 1px solid var(--border);
        }

        /* ════════════════════════════════
           LAYOUT
        ════════════════════════════════ */
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1.5fr;
            gap: 24px;
            align-items: start;
        }

        .panel {
            background: var(--navy-mid);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .panel-label {
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--gold);
            font-weight: 500;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 12px;
            display: block;
        }

        /* ════════════════════════════════
           INPUT PANEL
        ════════════════════════════════ */
        .query-textarea {
            width: 100%;
            min-height: 120px;
            background: var(--navy);
            border: 1px solid rgba(201,168,76,0.2);
            border-radius: 3px;
            color: var(--text);
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            line-height: 1.6;
            padding: 12px;
            resize: vertical;
            transition: border-color 0.2s;
            margin-bottom: 16px;
        }

        .query-textarea:focus {
            border-color: var(--gold);
            box-shadow: 0 0 0 2px rgba(201,168,76,0.12);
            outline: none;
        }

        .query-textarea::placeholder {
            color: var(--muted);
            font-style: italic;
        }

        .examples-section { margin-bottom: 20px; }

        .examples-grid {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .example-btn {
            background: var(--navy);
            border: 1px solid var(--border);
            border-radius: 3px;
            color: var(--muted);
            font-family: 'DM Sans', sans-serif;
            font-size: 12px;
            padding: 8px 12px;
            text-align: left;
            cursor: pointer;
            transition: all 0.15s;
        }

        .example-btn:hover {
            background: rgba(201,168,76,0.06);
            color: var(--cream);
            border-left: 2px solid var(--gold);
        }

        .params-section { margin-bottom: 20px; }

        .param-row {
            margin-bottom: 16px;
        }

        .param-label {
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 4px;
            display: block;
        }

        /* Dash slider overrides */
        .rc-slider-track { background: var(--gold) !important; }
        .rc-slider-handle { border-color: var(--gold) !important; }
        .rc-slider-rail { background: rgba(201,168,76,0.2) !important; }
        .rc-slider-dot-active { border-color: var(--gold) !important; }
        .rc-slider-mark-text { color: var(--muted) !important; font-size: 11px !important; }

        .button-row {
            display: flex;
            gap: 12px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #c9a84c, #a07a2a);
            border: none;
            border-radius: 3px;
            color: var(--navy);
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 13px 28px;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(201,168,76,0.3);
            transition: all 0.2s;
            flex: 1;
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(201,168,76,0.45);
        }

        .btn-secondary {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 3px;
            color: var(--muted);
            font-family: 'DM Sans', sans-serif;
            font-size: 12px;
            letter-spacing: 0.06em;
            padding: 12px 20px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-secondary:hover {
            border-color: var(--gold);
            color: var(--gold);
        }

        /* ════════════════════════════════
           OUTPUT PANEL
        ════════════════════════════════ */
        .output-tabs .tab {
            background: rgba(0,0,0,0.2) !important;
        }

        .output-tab {
            color: var(--muted) !important;
            font-size: 11px !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            font-weight: 500 !important;
            border-bottom: 2px solid transparent !important;
            background: transparent !important;
            padding: 12px 20px !important;
        }

        .output-tab-selected {
            color: var(--gold) !important;
            border-bottom-color: var(--gold) !important;
            background: rgba(201,168,76,0.05) !important;
        }

        .tab-content {
            min-height: 400px;
        }

        .placeholder-text {
            color: var(--muted);
            font-style: italic;
            font-size: 14px;
            text-align: center;
            padding: 60px 20px;
        }

        /* ── Analysis ── */
        .analysis-title {
            font-family: 'Playfair Display', serif;
            color: var(--gold-light);
            font-size: 18px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 16px;
        }

        .analysis-text {
            font-size: 14px;
            line-height: 1.75;
            color: var(--text);
        }

        .analysis-text p { margin-bottom: 12px; }
        .analysis-text strong { color: var(--gold-light); }

        /* ── Source cards ── */
        .source-card {
            background: var(--navy);
            border: 1px solid var(--border);
            border-radius: 3px;
            padding: 14px;
            margin-bottom: 12px;
        }

        .source-header {
            margin-bottom: 6px;
        }

        .source-badge {
            color: var(--gold);
            font-weight: 700;
            font-size: 13px;
        }

        .source-file {
            color: var(--cream);
            font-family: 'DM Mono', monospace;
            font-size: 13px;
        }

        .source-meta {
            font-size: 11px;
            color: var(--muted);
            margin-bottom: 8px;
            border-left: 3px solid var(--gold);
            padding-left: 10px;
            background: rgba(201,168,76,0.05);
            padding: 6px 10px;
            border-radius: 0 2px 2px 0;
        }

        .source-text {
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--border);
            border-radius: 3px;
            color: var(--cream);
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            padding: 10px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 200px;
            overflow-y: auto;
        }

        /* ════════════════════════════════
           LOADING
        ════════════════════════════════ */
        ._dash-loading { color: var(--gold) !important; }

        /* ════════════════════════════════
           DISCLAIMER
        ════════════════════════════════ */
        .disclaimer {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 14px 18px;
            background: rgba(192,57,43,0.07);
            border: 1px solid rgba(192,57,43,0.2);
            border-left: 3px solid var(--red);
            border-radius: 3px;
            font-size: 12px;
            color: var(--muted);
            line-height: 1.5;
            margin-top: 24px;
        }

        .disclaimer strong { color: #e07060; }

        /* ── Responsive ── */
        @media (max-width: 900px) {
            .main-content { grid-template-columns: 1fr; }
            .header { flex-direction: column; text-align: center; }
            .header-badges { align-items: center; margin-left: 0; }
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run the Dash server
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"🚀 Starting NyayaBiz Legal Advisor on port {APP_PORT}")
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)

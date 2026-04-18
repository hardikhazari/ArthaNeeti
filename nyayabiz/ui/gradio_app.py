# ─────────────────────────────────────────────────────────────────────────────
# GRADIO WEB UI  (NyayaBiz Legal Advisor)
# ─────────────────────────────────────────────────────────────────────────────

from nyayabiz.config import VS_INDEX, LLM_ENDPOINT


# ────────────────────────────────────────────────────────────────────────────
# QUERY FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def query_legal_advisor(question: str, num_sources: int = 5, temperature: float = 0.1):
    from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    if not question.strip():
        return "⚠️ Please enter a legal question.", ""

    try:
        vector_store = DatabricksVectorSearch(
            index_name=VS_INDEX,
            columns=["source_file", "section_id", "heading_chain", "xrefs", "page", "raw_text"]
        )

        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
        llm.temperature = temperature

        template = """You are NyayaBiz Legal Advisor, an expert AI assistant specialising in regulatory and legal analysis.

Your role:
- Provide accurate, concise legal guidance based on the provided context
- Cite specific sections, articles, or clauses when available
- If information is not in the context, clearly state: "This information is not available in the provided documents"
- Use professional legal terminology while remaining accessible
- Highlight key compliance requirements and obligations

Context from Legal Documents:
{context}

User Question:
{question}

Professional Legal Analysis:"""

        prompt = ChatPromptTemplate.from_template(template)

        retriever = vector_store.as_retriever(
            search_kwargs={'k': num_sources, 'query_type': 'HYBRID'}
        )
        docs = retriever.invoke(question)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        context = format_docs(docs)

        rag_chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(question)

        sources_md = f"### 📚 Source Documents &nbsp;·&nbsp; {len(docs)} results retrieved\n\n"
        sources_md += "---\n\n"
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source_file = meta.get('source_file', 'Unknown')
            section     = meta.get('section_id', 'N/A')
            page        = meta.get('page', 'N/A')
            heading     = meta.get('heading_chain', '')

            sources_md += f"**[{i}]** `{source_file}`\n\n"
            sources_md += f"> **Section:** {section} &nbsp;|&nbsp; **Page:** {page}"
            if heading and heading != 'N/A':
                sources_md += f" &nbsp;|&nbsp; **{heading}**"
            sources_md += "\n\n"
            text = doc.page_content[:450] + "…" if len(doc.page_content) > 450 else doc.page_content
            sources_md += f"```\n{text}\n```\n\n---\n\n"

        return f"### ⚖️ Legal Analysis\n\n{answer}", sources_md

    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nEnsure the vector index is ready and properly configured.", ""


# ────────────────────────────────────────────────────────────────────────────
# THEME & CSS
# ────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

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

/* ── Global reset ── */
body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--navy) !important;
    color: var(--text) !important;
}

/* Remove Gradio's default white card */
.gradio-container > .main,
.gradio-container .contain {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--gold); border-radius: 3px; }

/* ════════════════════════════════
   HEADER BANNER
════════════════════════════════ */
.nyaya-header {
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

.nyaya-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: var(--gold-light) !important;
    letter-spacing: 0.02em;
    margin: 0 0 4px !important;
}

.nyaya-header p {
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
}

.nyaya-seal {
    width: 60px; height: 60px;
    background: radial-gradient(circle at 38% 38%, #e8c96e, #c9a84c, #9a7a2c);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 0 3px var(--navy-mid), 0 0 0 5px var(--gold), inset 0 2px 8px rgba(0,0,0,0.3);
    flex-shrink: 0;
}

.nyaya-badges {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
}

.nyaya-badge {
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    font-weight: 500;
}

.badge-green { background: rgba(46,204,113,0.1); color: #2ecc71; border: 1px solid rgba(46,204,113,0.25); }
.badge-gold  { background: rgba(201,168,76,0.1); color: var(--gold); border: 1px solid var(--border); }

/* ════════════════════════════════
   PANEL CARDS
════════════════════════════════ */
.nyaya-panel {
    background: var(--navy-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    overflow: hidden !important;
}

/* ── Panel section labels ── */
.nyaya-label {
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
    font-weight: 500 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 12px !important;
}

/* ════════════════════════════════
   INPUTS
════════════════════════════════ */
textarea, input[type="text"] {
    background: var(--navy) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
    outline: none !important;
}

textarea::placeholder { color: var(--muted) !important; font-style: italic; }

/* ════════════════════════════════
   BUTTONS
════════════════════════════════ */
button.primary {
    background: linear-gradient(135deg, #c9a84c, #a07a2a) !important;
    border: none !important;
    border-radius: 3px !important;
    color: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 13px 28px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(201,168,76,0.3) !important;
    transition: all 0.2s !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201,168,76,0.45) !important;
}

button.secondary {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    padding: 12px 20px !important;
    transition: all 0.2s !important;
}

button.secondary:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}

/* ════════════════════════════════
   SLIDERS
════════════════════════════════ */
input[type="range"] { accent-color: var(--gold) !important; }

.gradio-slider .svelte-1gfkn6j { color: var(--text) !important; }

/* ════════════════════════════════
   TABS
════════════════════════════════ */
.tabs > .tab-nav {
    background: rgba(0,0,0,0.2) !important;
    border-bottom: 1px solid var(--border) !important;
}

.tabs .tab-nav button {
    color: var(--muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}

.tabs .tab-nav button.selected {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
    background: rgba(201,168,76,0.05) !important;
}

/* ════════════════════════════════
   MARKDOWN OUTPUT
════════════════════════════════ */
.prose, .markdown {
    color: var(--text) !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    background: transparent !important;
}

.prose h3, .markdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--gold-light) !important;
    font-size: 18px !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 10px !important;
    margin-bottom: 16px !important;
}

.prose code, .markdown code,
.prose pre, .markdown pre {
    background: var(--navy) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--cream) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

.prose blockquote, .markdown blockquote {
    border-left: 3px solid var(--gold) !important;
    background: rgba(201,168,76,0.05) !important;
    padding: 8px 14px !important;
    color: var(--muted) !important;
    margin: 8px 0 !important;
}

/* ════════════════════════════════
   ACCORDION (Advanced Settings)
════════════════════════════════ */
.accordion {
    background: var(--navy-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

.accordion .label-wrap span {
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
}

/* ════════════════════════════════
   EXAMPLES
════════════════════════════════ */
.examples table {
    background: var(--navy) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}

.examples td {
    color: var(--muted) !important;
    font-size: 12px !important;
    border-color: var(--border) !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}

.examples td:hover {
    background: rgba(201,168,76,0.06) !important;
    color: var(--cream) !important;
    border-left: 2px solid var(--gold) !important;
}

/* ════════════════════════════════
   DISCLAIMER BAR
════════════════════════════════ */
.nyaya-disclaimer {
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
    margin-top: 20px;
}

.nyaya-disclaimer strong { color: #e07060; }

/* ── Hide Gradio footer ── */
footer { visibility: hidden !important; }
"""


def launch_gradio(share: bool = True, server_port: int = 8000):
    """Build and launch the Gradio NyayaBiz Legal Advisor UI."""
    import gradio as gr

    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="nyaya-header">
            <div class="nyaya-seal">⚖️</div>
            <div>
                <h1>NyayaBiz Legal Advisor</h1>
                <p>AI-Powered Regulatory &amp; Compliance Intelligence</p>
            </div>
            <div class="nyaya-badges">
                <span class="nyaya-badge badge-green">● System Online</span>
                <span class="nyaya-badge badge-gold">Llama 3.3 · 70B</span>
                <span class="nyaya-badge badge-gold">Databricks VSS · Hybrid</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column: input ──────────────────────────────────────────
            with gr.Column(scale=2, elem_classes="nyaya-panel"):

                question_input = gr.Textbox(
                    label="Legal Query",
                    placeholder="State your legal question or compliance concern…",
                    lines=4,
                    show_label=True,
                    elem_classes="nyaya-label"
                )

                gr.Examples(
                    examples=[
                        ["What are the compliance requirements for food safety?"],
                        ["What regulations apply to worker health and safety?"],
                        ["What are the data privacy requirements under Indian law?"],
                        ["What penalties exist for environmental regulation violations?"],
                        ["What are the licensing requirements for medical facilities?"],
                    ],
                    inputs=question_input,
                    label="Precedent Queries",
                )

                with gr.Accordion("⚙  Analysis Parameters", open=True, elem_classes="accordion"):
                    num_sources = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="Source Documents to Retrieve",
                        info="More sources = broader context; fewer = tighter precision"
                    )
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                        label="Response Temperature",
                        info="0.0 = deterministic · 1.0 = exploratory"
                    )

                with gr.Row():
                    submit_btn = gr.Button("⚖  Analyse", variant="primary", size="lg")
                    clear_btn  = gr.Button("✕  Clear",   variant="secondary")

        # ── Output ──────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("💡  Legal Analysis"):
                        answer_output = gr.Markdown(
                            value="*Submit a query above to receive a professional legal analysis.*",
                            elem_classes="nyaya-panel"
                        )

                    with gr.Tab("📖  Source Documents"):
                        sources_output = gr.Markdown(
                            value="*Source documents will appear here after analysis.*",
                            elem_classes="nyaya-panel"
                        )

        # ── Disclaimer ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="nyaya-disclaimer">
            <span>⚠</span>
            <div>
                <strong>Disclaimer:</strong> This AI assistant provides informational guidance only and does not
                constitute formal legal advice. Always consult a qualified legal professional before taking
                action. &nbsp;·&nbsp; Powered by Databricks Vector Search + Meta Llama 3.3 70B.
            </div>
        </div>
        """)

        # ── Event handlers ──────────────────────────────────────────────────
        submit_btn.click(
            fn=query_legal_advisor,
            inputs=[question_input, num_sources, temperature],
            outputs=[answer_output, sources_output]
        )

        clear_btn.click(
            fn=lambda: ("*Submit a query above to receive a professional legal analysis.*",
                        "*Source documents will appear here after analysis.*", ""),
            outputs=[answer_output, sources_output, question_input]
        )

    # ────────────────────────────────────────────────────────────────────────
    print("🚀 Launching NyayaBiz Legal Advisor…")
    demo.launch(share=share, debug=True, server_name="0.0.0.0", server_port=server_port)

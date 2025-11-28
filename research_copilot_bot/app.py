import gradio as gr
import copy
import os
import yaml
import json
import uuid
import shutil
import threading
import time
from typing import List
from moya.agents.openai_agent import OpenAIAgent, OpenAIAgentConfig
from src.agents.gemini_agent import GeminiAgent, GeminiAgentConfig
from src.orchestrator import ResearchOrchestrator
import sqlite_utils

# --- Theme Definition ---
theme = gr.themes.Base(
    primary_hue="pink",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Proxima Nova"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Open Sans"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="#1a1a1a", # Black 90%
    body_text_color="#e5e5e5",
    background_fill_primary="#262626",
    background_fill_secondary="#1a1a1a",
    border_color_primary="#333333",
    block_background_fill="#262626",
    block_label_background_fill="#A4123F", # Primary Color
    block_title_text_color="#ffffff",
    input_background_fill="#333333",
    button_primary_background_fill="#A4123F",
    button_primary_text_color="#ffffff",
    slider_color="#A4123F",
    color_accent="#A4123F",
)

# gr.set_static_paths(paths=[os.path.abspath("../sessions")])

# --- Session Management ---

SESSIONS = {}

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.base_dir = os.path.join("sessions", self.session_id)
        self.pdfs_dir = os.path.join(self.base_dir, "pdfs")
        self.storage_dir = os.path.join(self.base_dir, "storage")
        self.trace_path = os.path.join(self.base_dir, "trace.jsonl")
        self.db_path = os.path.join(self.storage_dir, "research.db")

        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)

        self.orchestrator = None
        self.running = False
        self.stop_event = threading.Event()
        self.chat_agent = None

        # Cache for UI updates
        self.last_log_size = 0
        self.last_survey_mtime = 0
        self.last_synthesis_mtime = 0
        self.last_verification_mtime = 0
        self.last_paper_count = 0

    def get_config(self, api_key, topic, max_papers, temp, seed, model_backend):
        return {
            "seed": int(seed),
            "temperature": float(temp),
            "model_backend": model_backend,
            "model_name": "gpt-5-nano-2025-08-07" if model_backend == "openai" else "gemini-2.5-flash",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_store": "faiss",
            "db_path": self.db_path,
            "trace_path": self.trace_path,
            "pdf_chunk_size": 1200,
            "pdf_chunk_stride": 300,
            "input_papers": self.pdfs_dir,
            "seed_topic": topic if topic.strip() else None,
            "max_papers": int(max_papers),
            "api_key": api_key
        }


def get_session(session_id):
    if session_id is None or session_id not in SESSIONS:
        session = Session()
        SESSIONS[session.session_id] = session
        print(f"Creating new session (fallback for {session_id}): {session.session_id}")
        return session
    return SESSIONS[session_id]

def init_session_id():
    session = Session()
    SESSIONS[session.session_id] = session
    print(f"Initializing new session: {session.session_id}")
    return session.session_id

# --- Logic ---

def run_research(api_key, topic, max_papers, temp, seed, model_backend, uploaded_pdfs, doi_links, bibtex_file, session_id):
    session = get_session(session_id)

    # Set API Key
    if api_key:
        if model_backend == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif model_backend == "gemini":
            os.environ["GEMINI_API_KEY"] = api_key

    # Handle Uploaded PDFs
    if uploaded_pdfs:
        for file in uploaded_pdfs:
            shutil.copy(file.name, os.path.join(session.pdfs_dir, os.path.basename(file.name)))

    # Handle BibTeX (Just copy for now, orchestrator could be updated to parse it)
    if bibtex_file:
        shutil.copy(bibtex_file.name, os.path.join(session.pdfs_dir, "references.bib"))

    # Start Orchestrator in a separate thread
    if not session.running:
        session.running = True
        session.stop_event.clear()

        config = session.get_config(api_key, topic, max_papers, temp, seed, model_backend)
        session.orchestrator = ResearchOrchestrator(config_dict=config, stop_event=session.stop_event)

        def target():
            try:
                session.orchestrator.run()
            except Exception as e:
                print(f"Orchestrator failed: {e}")
                # Log error to trace
                with open(session.trace_path, "a") as f:
                    f.write(json.dumps({"ts": "ERROR", "type": "error", "msg": str(e)}) + "\n")
            finally:
                session.running = False

        thread = threading.Thread(target=target)
        thread.start()

    return session.session_id



def cancel_research(session_id):
    session = get_session(session_id)
    if session.running:
        session.stop_event.set()
        with open(session.trace_path, "a") as f:
            f.write(json.dumps({"type": "info", "msg": "Cancellation requested..."}) + "\n")
    return session.session_id

def chat_with_research(message, history, session_id):
    session = get_session(session_id)

    # Gather Context
    survey_path = os.path.join(session.storage_dir, "mini_survey.txt")
    synthesis_path = os.path.join(session.storage_dir, "synthesis.json")

    context = ""
    if os.path.exists(survey_path):
        with open(survey_path, "r") as f:
            context += f"--- Mini-Survey ---\n{f.read()}\n\n"

    if os.path.exists(synthesis_path):
        with open(synthesis_path, "r") as f:
            context += f"--- Synthesis ---\n{f.read()}\n\n"

    if not context:
        return "I don't have any research results yet. Please run a research session first."

    # Initialize Chat Agent if needed
    if not session.chat_agent:
        backend = "openai"
        api_key = None

        if session.orchestrator:
             backend = session.orchestrator.config.get("model_backend", "openai")
             api_key = session.orchestrator.config.get("api_key")

        if not api_key:
             if backend == "gemini":
                 api_key = os.environ.get("GEMINI_API_KEY")
             else:
                 api_key = os.environ.get("OPENAI_API_KEY")

        final_key = api_key or "mock-key"
        is_mock = (final_key == "mock-key")

        if backend == "gemini":
            config = GeminiAgentConfig(
                agent_name="chat",
                agent_type="GeminiAgent",
                description="Research Chat Agent",
                system_prompt="You are a helpful research assistant. Answer questions based on the provided research context.",
                model_name="gemini-1.5-flash",
                api_key=final_key
            )
            session.chat_agent = GeminiAgent(config)
        else:
            config = OpenAIAgentConfig(
                agent_name="chat",
                agent_type="OpenAIAgent",
                description="Research Chat Agent",
                system_prompt="You are a helpful research assistant. Answer questions based on the provided research context.",
                model_name="gpt-4o-mini",
                api_key=final_key
            )
            session.chat_agent = OpenAIAgent(config)

        session.chat_agent._is_mock = is_mock

    prompt = f"""
    Context:
    {context}

    User Question: {message}
    """

    if getattr(session.chat_agent, "_is_mock", False):
        return f"Mock Answer: Based on the research, '{message}' is an interesting point. (API Key missing)", session.session_id

    return session.chat_agent.handle_message(prompt), session.session_id

def read_logs(session_id):
    session = get_session(session_id)

    status = "**Status:** ⚪ Ready"
    if session.running:
        status = "**Status:** 🟢 Running Research..."
    elif session.orchestrator:
        status = "**Status:** ✅ Completed"
        # Check for error in logs
        if os.path.exists(session.trace_path):
             # Read last few lines to check for error
             try:
                 with open(session.trace_path, "r") as f:
                     content = f.read()
                     if '"type": "error"' in content:
                         status = "**Status:** 🔴 Error Occurred"
             except:
                 pass

    if not os.path.exists(session.trace_path):
        return "<p style='color: #888;'>No logs yet.</p>", status, session.session_id

    # Check if logs have changed
    current_size = os.path.getsize(session.trace_path)
    if current_size == session.last_log_size:
        return gr.update(), status, session.session_id

    session.last_log_size = current_size

    rows = []
    with open(session.trace_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                timestamp = entry.get("ts", "").split("T")[-1][:8]
                msg_type = entry.get("type", "INFO")

                # Create a summary message
                details = {k:v for k,v in entry.items() if k not in ["ts", "type"]}
                details_str = json.dumps(details, indent=2).replace('"', '&quot;')

                # Custom summary based on type
                if msg_type == "tool_call":
                    summary = f"Tool: <b>{entry.get('tool')}</b>"
                    if entry.get('paper_id'): summary += f" ({entry.get('paper_id')})"
                elif msg_type == "model_call":
                    summary = f"Agent: <b>{entry.get('agent')}</b>"
                elif msg_type == "arxiv_download":
                    summary = f"Downloaded {len(entry.get('files', []))} papers"
                elif msg_type == "process_paper_start":
                    summary = f"Processing <b>{entry.get('paper_id')}</b>"
                elif msg_type == "error":
                    summary = f"<span style='color:red'>Error: {entry.get('msg')}</span>"
                else:
                    # Fallback: show first key-value or empty
                    keys = list(details.keys())
                    if keys:
                        summary = f"{keys[0]}: {str(details[keys[0]])[:50]}"
                    else:
                        summary = ""

                # Color coding for type
                type_color = "#888"
                if "start" in msg_type: type_color = "#4CAF50" # Green
                if "end" in msg_type: type_color = "#2196F3" # Blue
                if "error" in msg_type: type_color = "#F44336" # Red
                if "tool" in msg_type: type_color = "#FF9800" # Orange
                if "model" in msg_type: type_color = "#9C27B0" # Purple

                row = f"""
                <tr title="{details_str}" style="border-bottom: 1px solid #333; cursor: help;">
                    <td style="padding: 4px 8px; color: #888; font-family: monospace;">{timestamp}</td>
                    <td style="padding: 4px 8px; color: {type_color}; font-weight: bold;">{msg_type}</td>
                    <td style="padding: 4px 8px;">{summary}</td>
                </tr>
                """
                rows.append(row)
            except:
                pass

    # Reverse rows to show newest first
    rows.reverse()

    html = f"""
    <div style="max-height: 400px; overflow-y: auto; background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
            <thead style="position: sticky; top: 0; background-color: #2d2d2d; z-index: 1;">
                <tr>
                    <th style="text-align: left; padding: 8px; color: #ccc;">Time</th>
                    <th style="text-align: left; padding: 8px; color: #ccc;">Type</th>
                    <th style="text-align: left; padding: 8px; color: #ccc;">Details (Hover for more)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """
    return html, status, session.session_id

def get_results(session_id):
    session = get_session(session_id)
    survey_path = os.path.join(session.storage_dir, "mini_survey.txt")
    synthesis_path = os.path.join(session.storage_dir, "synthesis.json")
    verification_path = os.path.join(session.storage_dir, "verification.json")

    # Check for updates
    survey_mtime = os.path.getmtime(survey_path) if os.path.exists(survey_path) else 0
    synthesis_mtime = os.path.getmtime(synthesis_path) if os.path.exists(synthesis_path) else 0
    verification_mtime = os.path.getmtime(verification_path) if os.path.exists(verification_path) else 0

    if (survey_mtime == session.last_survey_mtime and
        synthesis_mtime == session.last_synthesis_mtime and
        verification_mtime == session.last_verification_mtime):
        return gr.update(), gr.update(), gr.update(), session.session_id

    session.last_survey_mtime = survey_mtime
    session.last_synthesis_mtime = synthesis_mtime
    session.last_verification_mtime = verification_mtime

    survey = ""
    if os.path.exists(survey_path):
        with open(survey_path, "r") as f:
            survey = f.read()

    synthesis = {}
    if os.path.exists(synthesis_path):
        with open(synthesis_path, "r") as f:
            synthesis = json.load(f)

    verification = []
    if os.path.exists(verification_path):
        with open(verification_path, "r") as f:
            verification = json.load(f)

    return survey, synthesis, verification, session.session_id

def get_paper_list(session_id):
    session = get_session(session_id)
    if not os.path.exists(session.db_path):
        return gr.update(choices=[]), session.session_id

    try:
        db = sqlite_utils.Database(session.db_path)
        if "papers" not in db.table_names():
            return gr.update(choices=[]), session.session_id

        papers = []
        for row in db["papers"].rows:
            papers.append(row["id"])

        # Only update if count changed to avoid jitter/resetting selection
        if len(papers) == session.last_paper_count:
            return gr.update(), session.session_id

        session.last_paper_count = len(papers)
        return gr.update(choices=papers), session.session_id
    except:
        return gr.update(), session.session_id

def get_paper_details(session_id, paper_id):
    session = get_session(session_id)
    if not paper_id or not os.path.exists(session.db_path):
        return "", "", session.session_id

    try:
        db = sqlite_utils.Database(session.db_path)

        # Get Summary
        summary_md = "_No summary available._"
        if "summaries" in db.table_names():
            summary_row = db["summaries"].get(paper_id)
            if summary_row:
                s = json.loads(summary_row["summary"])
                # Format as Markdown
                summary_md = f"""
## {s.get('title', 'Untitled')}
*{s.get('authors', 'Unknown Authors')}*

### 🎯 Abstract
{s.get('abstract', 'N/A')}

### ❓ Problem
{s.get('problem', 'N/A')}

### 🔬 Method
{s.get('method', 'N/A')}

### 📊 Results
{s.get('results', 'N/A')}

### ⚠️ Limitations
{s.get('limitations', 'N/A')}

### 🏷️ Key Phrases
{', '.join([f"`{p}`" for p in s.get('key_phrases', [])])}
"""

        # Get PDF Path
        iframe_html = ""
        if "papers" in db.table_names():
            paper_row = db["papers"].get(paper_id)
            if paper_row and paper_row.get("path"):
                pdf_path = paper_row["path"]
                # Ensure absolute path for Gradio
                abs_path = os.path.abspath(pdf_path)
                print(f"Serving PDF from path: {abs_path}, {pdf_path}")
                # Use Gradio's /file/ route to serve the PDF
                iframe_html = f'<iframe src="/gradio_api/file={pdf_path}" width="100%" height="800px" style="border: none;"></iframe>'

        return summary_md, iframe_html, session.session_id
    except Exception as e:
        return f"Error: {str(e)}", f"Error loading PDF: {str(e)}", session.session_id

# --- UI Layout ---

with gr.Blocks(theme=theme, title="Research Copilot") as app:
    session_state = gr.State(init_session_id)


    # Allow serving files from the sessions directory
    app.allowed_paths = [os.path.abspath("sessions")]

    with gr.Row():
        gr.Markdown("#Research Copilot Bot\n### Powered by MOYA Framework")

    with gr.Row():
        with gr.Column(scale=1):
            # Control Panel
            with gr.Group():
                start_btn = gr.Button("🚀 Start Research", variant="primary")
                status_output = gr.Markdown("**Status:** ⚪ Ready")

            gr.Markdown("### ⚙️ Configuration")
            model_backend = gr.Dropdown(choices=["openai", "gemini"], value="openai", label="Model Provider")
            api_key = gr.Textbox(label="API Key (OpenAI or Gemini)", type="password", placeholder="sk-... or AIza...")
            topic = gr.Textbox(label="Seed Topic (Arxiv Search)", placeholder="e.g., LLM in Software Engineering")
            max_papers = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Papers to Download")

            with gr.Accordion("Advanced Settings", open=False):
                temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
                seed = gr.Number(value=12345, label="Random Seed")

            gr.Markdown("### 📂 Data Sources")
            uploaded_pdfs = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
            doi_links = gr.Textbox(label="DOI Links (One per line)", lines=3, placeholder="10.1145/3611643.3616242...")
            bibtex_file = gr.File(label="Upload BibTeX", file_types=[".bib"])

        with gr.Column(scale=2):
            gr.Markdown("### 📡 Live Trace")
            # Using HTML for custom table view
            logs_output = gr.HTML(label="System Logs")

            gr.Markdown("### 📝 Results")
            with gr.Tabs():
                with gr.TabItem("📄 Paper Explorer"):
                    gr.Markdown("_Select a processed paper to view its structured summary and the original PDF side-by-side._")
                    with gr.Row():
                        paper_dropdown = gr.Dropdown(label="Select Paper", choices=[], interactive=True, scale=3)
                        refresh_papers_btn = gr.Button("🔄 Refresh List", size="sm", scale=1)

                    with gr.Row():
                        with gr.Column(scale=1):
                            paper_summary_md = gr.Markdown()
                        with gr.Column(scale=1):
                            paper_pdf_viewer = gr.HTML(label="PDF Viewer")

                with gr.TabItem("Mini-Survey"):
                    gr.Markdown("_A generated survey article synthesizing the findings from the processed papers._")
                    survey_output = gr.Markdown()
                with gr.TabItem("Synthesis"):
                    gr.Markdown("_Key themes, gaps, and insights extracted from the research set._")
                    synthesis_output = gr.JSON()
                with gr.TabItem("Verification"):
                    gr.Markdown("_Fact-checking results verifying claims in the survey against source texts._")
                    verification_output = gr.JSON()
                with gr.TabItem("Chat with Research"):
                    gr.Markdown("_Ask questions about the collected research and generated survey._")
                    chat_interface = gr.ChatInterface(
                        fn=chat_with_research,
                        additional_inputs=[session_state],
                        additional_outputs=[session_state],
                        type="messages"
                    )

    # Event Handlers
    def toggle_process(api_key, topic, max_papers, temp, seed, model_backend, uploaded_pdfs, doi_links, bibtex_file, session_id):
        session = get_session(session_id)
        if session.running:
            # Cancel
            cancel_research(session_id)
            return session.session_id, gr.update(value="🛑 Cancelling...", variant="stop", interactive=False)
        else:
            # Start
            gr.Info("Starting Research Session... 🚀")
            new_session_id = run_research(api_key, topic, max_papers, temp, seed, model_backend, uploaded_pdfs, doi_links, bibtex_file, session_id)
            return new_session_id, gr.update(value="🛑 Cancel Research", variant="stop")

    def update_ui(session_id):
        logs, status_md, _ = read_logs(session_id)
        session = get_session(session_id)

        # Button State Logic
        btn_update = gr.update()
        if session.running:
             if session.stop_event.is_set():
                 btn_update = gr.update(value="🛑 Cancelling...", variant="stop", interactive=False)
             else:
                 btn_update = gr.update(value="🛑 Cancel Research", variant="stop", interactive=True)
        else:
             btn_update = gr.update(value="🚀 Start Research", variant="primary", interactive=True)

        return logs, status_md, btn_update, session.session_id

    start_btn.click(
        toggle_process,
        inputs=[api_key, topic, max_papers, temp, seed, model_backend, uploaded_pdfs, doi_links, bibtex_file, session_state],
        outputs=[session_state, start_btn]
    )

    # Timer for updates
    timer = gr.Timer(1)

    timer.tick(
        update_ui,
        inputs=[session_state],
        outputs=[logs_output, status_output, start_btn, session_state]
    )

    timer.tick(
        get_results,
        inputs=[session_state],
        outputs=[survey_output, synthesis_output, verification_output, session_state]
    )

    # Auto-refresh paper list
    timer.tick(
        get_paper_list,
        inputs=[session_state],
        outputs=[paper_dropdown, session_state]
    )

    # Manual refresh
    refresh_papers_btn.click(
        get_paper_list,
        inputs=[session_state],
        outputs=[paper_dropdown, session_state]
    )

    # Load details on selection
    paper_dropdown.change(
        get_paper_details,
        inputs=[session_state, paper_dropdown],
        outputs=[paper_summary_md, paper_pdf_viewer, session_state]
    )

if __name__ == "__main__":
    # Ensure sessions directory exists for allowed_paths
    os.makedirs("sessions", exist_ok=True)
    app.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[os.path.abspath("sessions")])

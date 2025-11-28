# Research Copilot Bot

A sophisticated multi-agent system for automating software engineering research, powered by the **MOYA Framework**. This tool helps researchers find, summarize, synthesize, and verify academic papers using LLMs (OpenAI GPT-4o or Google Gemini).

## ✨ Features

-   **Multi-Agent Orchestration**: Coordinates specialized agents for searching, reading, summarizing, and synthesizing research.
-   **Dual Model Support**: Switch seamlessly between **OpenAI** and **Google Gemini** models.
-   **Interactive UI**: Modern Gradio interface for managing research sessions, visualizing logs, and exploring results.
-   **Automated Literature Review**:
    -   **Arxiv Integration**: Automatically searches and downloads relevant papers based on a topic.
    -   **PDF Processing**: Ingests and chunks PDF documents for analysis.
    -   **Synthesis & Verification**: Generates a mini-survey and verifies claims against source text.
-   **Session Management**: Persists research sessions, allowing you to resume or review past work.
-   **Live Tracing**: Watch the agents think and act in real-time via the UI.

## 🚀 Getting Started

### Prerequisites

-   Python 3.10+
-   An API Key for **OpenAI** (`sk-...`) or **Google Gemini** (`AIza...`).

### Installation

1.  **Clone the repository** (if applicable) and navigate to the bot directory:
    ```bash
    cd research_copilot_bot
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure the `moya` framework is installed or available in your Python path.*

### Running the Application

1.  **Start the Gradio Interface**:
    ```bash
    python app.py
    ```

2.  **Open your browser**:
    Navigate to `http://localhost:7860`.

3.  **Start Researching**:
    -   Select your **Model Provider** (OpenAI or Gemini).
    -   Enter your **API Key**.
    -   Provide a **Seed Topic** (e.g., "Large Language Models in Software Testing") OR upload your own **PDFs**.
    -   Click **🚀 Start Research**.

## 📂 Project Structure

```
research_copilot_bot/
├── app.py                 # Main Gradio application entry point
├── main.py                # CLI entry point (alternative to UI)
├── config.yaml            # Default configuration
├── requirements.txt       # Python dependencies
├── src/
│   ├── agents/            # Agent implementations (Gemini, OpenAI)
│   ├── tools/             # Tools for Arxiv, PDF processing, Embeddings
│   ├── orchestrator.py    # Main logic coordinating the research workflow
│   └── storage.py         # Database and Vector Store management
├── sessions/              # (Generated) Stores session data, PDFs, and logs
└── scripts/               # Utility scripts
```

## 🛠️ Configuration

You can adjust default settings in `config.yaml` or directly in the UI:
-   **Max Papers**: Limit the number of papers to download/process.
-   **Temperature**: Control the creativity of the model.
-   **Seed**: Set a random seed for reproducibility.

## 📊 Outputs

For each session, the bot generates:
-   **`mini_survey.txt`**: A coherent literature review article.
-   **`synthesis.json`**: Structured insights, themes, and gaps.
-   **`verification.json`**: Fact-checking results linking claims to source quotes.
-   **`trace.jsonl`**: Detailed execution logs of agent thoughts and actions.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the agents, add new tools, or enhance the UI!

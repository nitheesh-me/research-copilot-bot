<div align="center">

![Header](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=300&section=header&text=Research%20Copilot%20Bot&fontSize=70)

</div>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 🤖 A sophisticated multi-agent system for automating software engineering research using the MOYA Framework.
    <br>
</p>

> **⚠️ Prototype Disclaimer**: This project is a **prototype** developed for an academic assignment. It is designed to demonstrate the capabilities of multi-agent systems in research automation but is **not a full-fledged production system**. It may have limitations in error handling, scalability, and edge-case coverage.

## 📝 Table of Contents

- [About](#about)
- [Features](#features)
- [Documentation](#documentation)
- [Repository Structure](#structure)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>The **Research Copilot Bot** is an advanced AI-driven tool designed to assist researchers in the field of software engineering. Built upon the **MOYA Framework**, it orchestrates multiple specialized agents to automate the tedious parts of literature review.

The system leverages state-of-the-art Large Language Models (LLMs) like **OpenAI GPT-4o** and **Google Gemini** to:
1.  **Search** for relevant academic papers (via Arxiv).
2.  **Read and Summarize** PDF documents.
3.  **Synthesize** findings into a coherent literature review.
4.  **Verify** claims against source texts.
5.  **Chat** with the synthesized knowledge.

## 🚀 Features <a name = "features"></a>

-   **Multi-Agent Workflow**: Orchestrated by MOYA to handle complex research tasks autonomously.
-   **Dual Backend Support**: Switch seamlessly between OpenAI and Google Gemini models.
-   **Interactive UI**: A modern web interface built with Gradio for easy interaction.
-   **Live Tracing**: Visualize the agent's thought process, tool usage, and decision-making in real-time. (hover over steps to see more details)
-   **Session Persistence**: Automatically saves research sessions, allowing you to resume work later.
-   **Automated Synthesis**: Generates structured summaries and mini-surveys from multiple papers.
-   **Claim Verification**: Cross-checks synthesized claims against original documents for accuracy.
-  **Various Ingestion Methods**: Supports both topic-based searches and direct PDF/DOI/BIB uploads.

## 📚 Documentation <a name = "documentation"></a>

For more detailed information about the system's internals, please refer to the documentation in the `research_copilot_bot/docs/` folder:

-   [**System Architecture**](research_copilot_bot/docs/architecture.md): Overview of the orchestrator, storage, and data flow.
-   [**Agents & Tools**](research_copilot_bot/docs/agents.md): Details about the specific agents (OpenAI, Gemini) and tools (Arxiv, PDF) used.

## Screenshots

![Preview](<preview.png>)

### Sequence Diagram
![Sequence Diagram](<sequence_diagram.svg>)

## 📂 Repository Structure <a name = "structure"></a>

-   **`moya/`**: The core multi-agent framework library. Contains the abstractions for Agents, Orchestrators, Memory, and Tools.
-   **`research_copilot_bot/`**: The main application code for the Research Copilot.
    -   `app.py`: The Gradio-based web interface.
    -   `src/`: Source code for the bot's specific agents and tools.
    -   `sessions/`: Directory where research session data is stored. (for debugging)

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   **Python 3.10+**
-   **API Keys**: You will need an API key for either OpenAI (`OPENAI_API_KEY`) or Google Gemini (`GEMINI_API_KEY`).

### Installation

1.  Clone the repository and navigate to the bot directory:
    ```bash
    cd research_copilot_bot
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🎈 Usage <a name = "usage"></a>

To start the interactive web interface, run the following command from the `research_copilot_bot` directory:

```bash
python app.py
```

Once the server starts, open your web browser and navigate to:
`http://localhost:7860`

### Using the Bot
1.  **Select Model**: Choose between OpenAI and Gemini.
2.  **Enter API Key**: Provide your secure API key.
3.  **Define Topic**: Enter a research topic (e.g., "LLMs in Software Testing") or upload PDF files directly.
4.  **Start**: Click "Start Research" and watch the agents work in the "Live Trace" panel.

## ⛏️ Built Using <a name = "built_using"></a>

-   [MOYA Framework](https://github.com/montycloud/moya) - Multi-Agent Orchestration
-   [Gradio](https://gradio.app/) - Web Interface
-   [OpenAI](https://openai.com/) - LLM Provider
-   [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM Provider
-   [LangChain](https://www.langchain.com/) - Utilities
-   [FAISS](https://github.com/facebookresearch/faiss) - Vector Store

## ✍️ Authors <a name = "authors"></a>

-   **Nitheesh** - *Initial work* - [As part of "Topics in Software Engineering" Course, IIIT-Hyderabad](https://www.iiit.ac.in/)

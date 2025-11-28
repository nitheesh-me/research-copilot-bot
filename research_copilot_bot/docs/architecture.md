# System Architecture

## Overview
The Research Copilot Bot is designed as a modular multi-agent system using the MOYA framework. It follows a centralized orchestration pattern where a `ResearchOrchestrator` manages the lifecycle of the research task.

## Components

### 1. Orchestrator (`src/orchestrator.py`)
The central brain of the system. It:
- Receives the user's topic or uploaded files.
- Decides which steps to execute (Search -> Download -> Read -> Summarize -> Synthesize).
- Manages the shared state (memory) across different agents.
- Handles error recovery and logging.

### 2. Storage Layer (`src/storage.py`)
- **SQLite**: Stores structured metadata about papers (title, authors, abstract) and the generated summaries.
- **FAISS**: A vector database used to store embeddings of paper chunks for semantic search during the "Chat" phase.
- **File System**: Stores raw PDFs and generated reports (`mini_survey.txt`, `synthesis.json`).

### 3. Data Flow
1.  **Ingestion**: Papers are downloaded from Arxiv or uploaded by the user.
2.  **Processing**: `PDFTool` converts PDFs to text and chunks them.
3.  **Analysis**: Agents read the chunks and generate summaries.
4.  **Synthesis**: The Orchestrator aggregates summaries and prompts an agent to write a survey.
5.  **Verification**: The system cross-references claims in the survey against the vector store to ensure accuracy.

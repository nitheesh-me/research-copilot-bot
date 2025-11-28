# Agents and Tools

## Agents

The system is designed to be model-agnostic, supporting both OpenAI and Google Gemini backends.

### 1. OpenAIAgent
- **Model**: `gpt-4o-mini`
- **Role**: General-purpose reasoning, summarization, and synthesis.
- **Configuration**: Uses standard OpenAI API.

### 2. GeminiAgent
- **Model**: `gemini-1.5-flash`
- **Role**: An alternative backend offering large context windows, suitable for reading full papers.
- **Configuration**: Uses Google Generative AI SDK.

## Tools

### 1. ArxivTool
- **Function**: Searches Arxiv.org for papers matching a query.
- **Output**: Downloads PDFs and extracts metadata (BibTeX).

### 2. PDFTool
- **Function**: Parses PDF files.
- **Capabilities**:
    - Text Extraction
    - Smart Chunking (overlapping windows)
    - Reference Extraction

### 3. Embedder
- **Function**: Converts text chunks into vector embeddings.
- **Model**: `all-MiniLM-L6-v2` (local HuggingFace model) for efficiency.

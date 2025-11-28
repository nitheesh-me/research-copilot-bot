#!/bin/bash
set -e

cd "$(dirname "$0")"

# Generate dummy PDFs
python scripts/generate_dummy_pdfs.py

# Run the orchestrator
python main.py


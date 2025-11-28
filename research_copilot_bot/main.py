import sys
import os

# Add moya to path if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../moya")))

from src.orchestrator import ResearchOrchestrator

def main():
    # Ensure directories exist
    os.makedirs("storage", exist_ok=True)
    os.makedirs("pdfs", exist_ok=True)

    # Check for config
    if not os.path.exists("config.yaml"):
        print("Config file not found.")
        return

    orchestrator = ResearchOrchestrator("config.yaml")
    orchestrator.run()

if __name__ == "__main__":
    main()

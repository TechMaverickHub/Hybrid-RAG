"""
CLI entry point for the Hybrid RAG system.

Usage:
    python main.py ingest    # Ingest documents from data/ folder
    python main.py           # Start interactive Q&A
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import settings


def run_ingest():
    """Run the document ingestion pipeline."""
    from src.ingest import ingest_pipeline

    print("=" * 60)
    print("  HYBRID RAG — Document Ingestion")
    print("=" * 60)
    ingest_pipeline()
    print("\nIngestion complete! You can now run 'python main.py' to query.\n")


def run_interactive():
    """Run the interactive Q&A loop."""
    from src.graph import build_graph

    # Check if FAISS index exists
    if not os.path.exists(settings.index_path):
        print("No FAISS index found!")
        print(f"  1. Place your .txt or .pdf files in '{settings.data_directory}/'")
        print("  2. Run: python main.py ingest")
        return

    print("=" * 60)
    print("  HYBRID RAG — Interactive Q&A")
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  Confidence Threshold: {settings.similarity_threshold}")
    print("=" * 60)

    print("\nBuilding graph...")
    graph = build_graph()
    print("Ready! Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        # Initialize state and invoke the graph
        initial_state = {
            "query": query,
            "route": "",
            "route_reasoning": "",
            "doc_results": [],
            "web_results": "",
            "confidence_met": False,
            "fused_context": "",
            "answer": "",
            "sources": [],
        }

        print("\nProcessing...")
        result = graph.invoke(initial_state)

        # Display results
        print(f"\n{'─' * 50}")
        print(f"  Route:      {result['route']}  ({result['route_reasoning']})")
        print(f"  Confidence: {'MET' if result['confidence_met'] else 'LOW (fallback used)'}")
        print(f"{'─' * 50}")
        print(f"\nAnswer:\n{result['answer']}")

        if result.get("sources"):
            print("\nSources:")
            for src in result["sources"]:
                print(f"  - {src}")

        print(f"\n{'═' * 60}\n")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "ingest":
            run_ingest()
        elif command == "help":
            print("Usage:")
            print("  python main.py ingest   — Ingest documents from data/ folder")
            print("  python main.py          — Start interactive Q&A")
        else:
            print(f"Unknown command: {command}")
            print("Run 'python main.py help' for usage.")
    else:
        run_interactive()


if __name__ == "__main__":
    main()

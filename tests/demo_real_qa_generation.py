"""
Real QA Generation Demo - Uses OpenRouter Llama 3.3 70B Instruct
This script makes actual API calls and shows real LLM outputs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from backend.src.subgraphs.QA_generation import create_qa_generation_graph


def demo_real_qa_generation():
    """Run demo with real OpenRouter LLM."""

    # Batch text
    batch_text = """
    Quantum computing represents a revolutionary approach to information processing 
    that leverages quantum mechanical phenomena. Unlike classical computers that use 
    bits (0 or 1), quantum computers use quantum bits or "qubits" which can exist in 
    a superposition of both 0 and 1 simultaneously.
    
    The power of quantum computing comes from three key principles: superposition, 
    entanglement, and interference. Superposition allows qubits to be in multiple 
    states at once. Entanglement links qubits together so that the state of one 
    instantly affects others. Interference allows quantum algorithms to amplify 
    correct answers while canceling out wrong ones.
    
    Quantum computers excel at solving certain types of problems that would take 
    classical computers an impractically long time, such as factoring large numbers, 
    simulating molecular behavior, and optimization problems. However, they are not 
    universally faster than classical computers for all tasks.
    
    Current challenges include maintaining quantum coherence (qubits losing their 
    quantum properties), error correction, and scaling up to larger numbers of qubits.
    """

    print("=" * 80)
    print("REAL QA GENERATION DEMO - OpenRouter Llama 3.3 70B")
    print("=" * 80)
    print(f"\nBatch Text:\n{batch_text}\n")
    print("-" * 80)
    print("\n⏳ Generating QA pairs using real LLM (this may take a moment)...\n")

    try:
        # Create graph with REAL LLM (no dummy LLMs passed)
        graph = create_qa_generation_graph(max_attempts=3, max_qas=5)

        # Invoke
        result = graph.invoke({"text": batch_text})

        print("\n" + "=" * 80)
        print("GENERATION COMPLETE ✓")
        print("=" * 80)
        print(f"\nTotal QA Pairs Generated: {len(result['questions'])}")

        print("\n" + "=" * 80)
        print("GENERATED QA PAIRS")
        print("=" * 80)

        for idx, (question, answer) in enumerate(zip(result["questions"], result["answers"]), 1):
            print(f"\n[QA Pair #{idx}]")
            print(f"  ❓ Question: {question}")
            print(f"  ✓ Answer:   {answer}")

        print("\n" + "=" * 80)
        print("FULL RESULT (JSON)")
        print("=" * 80)
        print(json.dumps(result, indent=2))

        print("\n" + "=" * 80)
        print("DEMO COMPLETE ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. .env file exists at: backend/.env")
        print("2. LLM_API_KEY is set correctly")
        print("3. You have internet connection")
        print("4. OpenRouter API is accessible")


if __name__ == "__main__":
    demo_real_qa_generation()

"""
Demo script to test QA generation with dummy data and print detailed output.
This script shows iterations, attempts, feedback, and generated questions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from backend.src.subgraphs.QA_generation import create_qa_generation_graph
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class DemoDummyLLM(BaseChatModel):
    """Mock LLM that returns predefined responses for demo purposes."""

    def __init__(self, responses):
        super().__init__()
        self._iter = iter(responses)
        object.__setattr__(self, 'call_count', 0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            content = next(self._iter)
        except StopIteration:
            content = "DONE"
        object.__setattr__(self, 'call_count', object.__getattribute__(self, 'call_count') + 1)
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "demo_dummy"

    @property
    def _identifying_params(self):
        return {}


def demo_qa_generation():
    """Run demo with dummy batch text and print detailed output."""

    # Dummy batch text
    batch_text = """
    Machine learning is a subset of artificial intelligence (AI) that enables 
    systems to learn and improve from experience without being explicitly programmed. 
    
    There are three main types of machine learning: supervised learning, unsupervised 
    learning, and reinforcement learning.
    
    Supervised learning uses labeled data to train models, while unsupervised learning 
    finds patterns in unlabeled data. Reinforcement learning trains agents through 
    rewards and penalties.
    
    Common applications include image recognition, natural language processing, 
    recommendation systems, and autonomous vehicles. Deep learning, using neural 
    networks, has revolutionized many of these fields.
    """

    # Demo responses: some bad, some good, then DONE
    gen_responses = [
        "What is machine learning? Machine learning is a subset of AI that enables systems to learn from experience without explicit programming.",
        "What are the three types of machine learning? The three types are supervised learning, unsupervised learning, and reinforcement learning.",
        "How does supervised learning work? Supervised learning uses labeled data to train models.",
        "DONE"
    ]

    judge_responses = [
        "no - too vague and generic definition",
        "yes",
        "yes",
        "yes"
    ]

    # Create mock LLMs
    gen = DemoDummyLLM(gen_responses)
    judge = DemoDummyLLM(judge_responses)

    # Create graph
    graph = create_qa_generation_graph(max_attempts=3, max_qas=20, llm=gen, judge_model=judge)

    print("=" * 80)
    print("QA GENERATION DEMO")
    print("=" * 80)
    print(f"\nBatch Text:\n{batch_text}\n")
    print("-" * 80)

    # Invoke graph
    result = graph.invoke({"text": batch_text})

    print("\nGENERATION SUMMARY")
    print("-" * 80)
    print(f"Total Generator Calls: {object.__getattribute__(gen, 'call_count')}")
    print(f"Total Judge Calls: {object.__getattribute__(judge, 'call_count')}")
    print(f"Total QA Pairs Generated: {len(result['questions'])}")

    print("\n" + "=" * 80)
    print("GENERATED QA PAIRS")
    print("=" * 80)

    for idx, (question, answer) in enumerate(zip(result["questions"], result["answers"]), 1):
        print(f"\n[QA Pair #{idx}]")
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")

    print("\n" + "=" * 80)
    print("FULL RESULT JSON")
    print("=" * 80)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo_qa_generation()

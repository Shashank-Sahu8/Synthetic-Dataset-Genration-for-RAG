from backend.src.subgraphs.QA_generation import create_qa_generation_graph
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class DummyLLM(BaseChatModel):
    def __init__(self, responses):
        super().__init__()
        self._iter = iter(responses)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            content = next(self._iter)
        except StopIteration:
            content = "DONE"
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "dummy"

    @property
    def _identifying_params(self):
        return {}


def test_multiple_qa_generation():
    # Generator: produces two QAs then DONE
    gen_responses = ["What is X? X is Y", "How does Z work? Z works by W", "DONE"]
    gen = DummyLLM(gen_responses)

    # Judge: both good
    judge_responses = ["yes", "yes", "yes"]
    judge = DummyLLM(judge_responses)

    graph = create_qa_generation_graph(max_attempts=3, max_qas=10, llm=gen, judge_model=judge)

    out = graph.invoke({"text": "dummy context"})
    assert len(out["questions"]) == 2
    assert len(out["answers"]) == 2
    assert out["questions"][0] == "What is X?"
    assert out["answers"][0] == "X is Y"
    assert out["questions"][1] == "How does Z work?"
    assert out["answers"][1] == "Z works by W"


def test_max_qas_limit():
    # Generator: keeps producing, but max_qas=2
    gen_responses = ["Q1? A1", "Q2? A2", "Q3? A3"]
    gen = DummyLLM(gen_responses)

    judge_responses = ["yes"] * 3
    judge = DummyLLM(judge_responses)

    graph = create_qa_generation_graph(max_attempts=3, max_qas=2, llm=gen, judge_model=judge)

    out = graph.invoke({"text": "dummy"})
    assert len(out["questions"]) == 2  # stopped at max_qas
    assert len(out["answers"]) == 2


def test_judge_retry():
    # Generator: first bad, then good on retry
    gen_responses = ["Bad Q? Bad A", "Good Q? Good A", "DONE"]
    gen = DummyLLM(gen_responses)

    judge_responses = ["no - bad", "yes", "yes"]
    judge = DummyLLM(judge_responses)

    graph = create_qa_generation_graph(max_attempts=3, max_qas=10, llm=gen, judge_model=judge)

    out = graph.invoke({"text": "dummy"})
    assert len(out["questions"]) == 1
    assert out["questions"][0] == "Good Q?"
    assert out["answers"][0] == "Good A"

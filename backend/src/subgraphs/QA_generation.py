from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import END
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# **State, input, and output schemas for the QA subgraph**
class QAState(TypedDict, total=False):
    """Mutable state that flows through the QA generation subgraph."""

    # the original batch of text we are generating QA for
    text: str

    # current candidate question (or "DONE")
    current_question: str

    # current candidate answer
    current_answer: str

    # accumulated questions
    questions: list[str]

    # accumulated answers
    answers: list[str]

    # previously asked questions (for diversity)
    previous_questions: list[str]

    # how many times we've tried the current QA
    attempt: int

    # whether the judge thought the current QA was good
    good: bool

    # feedback from judge if not good
    feedback: str

    # whether generator signaled done
    done: bool


class QAInput(TypedDict):
    """Input schema when the compiled graph is invoked."""

    text: str


class QAOutput(TypedDict):
    """Output schema returned when the graph completes."""

    questions: list[str]
    answers: list[str]


# ---------------------------------------------------------------------------
# builder function that constructs (and compiles) the subgraph
# ---------------------------------------------------------------------------
def create_qa_generation_graph(
    max_attempts: int = 3,
    max_qas: int = 20,
    llm: ChatOpenAI | None = None,
    judge_model: ChatOpenAI | None = None
) -> StateGraph[QAState, Any, QAInput, QAOutput]:
    """Return a compiled subgraph that generates multiple QA pairs with judge validation.

    Args:
        max_attempts: Max retries per QA pair.
        max_qas: Max QA pairs to generate before stopping.
        llm: Optional generator LLM. If None, uses OpenRouter Llama 3.3 70B from env.
        judge_model: Optional judge LLM. If None, uses same as llm.
    """

    if llm is None:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))  # Load backend/.env
        # Load model config from env
        model_name = os.getenv("LLM_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("LLM_API_KEY")  # API key from env
        )

    if judge_model is None:
        judge_model = llm

    # ------------------------------------------------------------------
    # nodes
    # ------------------------------------------------------------------

    def _generate_candidate(state: QAState, runtime: Runtime) -> dict:
        # initialize lists if not present
        if "questions" not in state:
            state["questions"] = []
        if "answers" not in state:
            state["answers"] = []
        if "previous_questions" not in state:
            state["previous_questions"] = []

        num_existing = len(state["questions"])
        if num_existing >= max_qas:
            return {"current_question": "DONE", "current_answer": "", "done": True}

        state.setdefault("attempt", 0)

        feedback = state.get("feedback", "").strip()
        feedback_part = f"\n\nPrevious feedback from judge: {feedback}" if feedback else ""

        # Build list of previously asked questions for diversity
        previous_q = state.get("previous_questions", [])
        previous_list = "\n".join([f"- {q}" for q in previous_q]) if previous_q else "None yet"
        diversity_part = f"""\n\nPreviously asked questions (DO NOT repeat these):
{previous_list}

Generate a question about a DIFFERENT topic or aspect of the text. Focus on areas not yet covered.
"""

        done_part = f"\n\nYou have already generated {num_existing} QA pairs." if num_existing > 0 else ""

        prompt = f"""
You are given the following batch of text. Generate a single question and its corresponding answer based on the content.{feedback_part}{diversity_part}{done_part}

If you cannot generate a useful NEW question (all topics exhausted), reply with just 'DONE'.

Format your answer as:
Question? Answer

Text:
{{body}}
""".format(body=state["text"])

        response = llm.invoke(prompt)
        text = response.content.strip()

        if text.upper() == "DONE":
            return {"current_question": "DONE", "current_answer": "", "done": True}
        else:
            # split into Q and A
            if "?" in text:
                q, a = text.split("?", 1)
                q = q.strip() + "?"
                a = a.strip()
            else:
                q = text
                a = ""
            return {"current_question": q, "current_answer": a, "done": False, "attempt": 0, "previous_questions": previous_q + [q]}

    def _check_if_done(state: QAState, runtime: Runtime) -> dict:
        if state.get("current_question") == "DONE":
            return {"done": True}
        return {"done": False}

    def _judge(state: QAState, runtime: Runtime) -> dict:
        if state.get("done"):
            return {"good": True}  # skip if done

        prompt = f"""
Here is the original context:
{state["text"]}

Here is a candidate question:
{state.get("current_question", "")}

Answer:
{state.get("current_answer", "")}

Is the question/answer accurate and clearly grounded in the context? Reply with just "yes" or "no". If "no", provide brief feedback on why.
"""

        response = judge_model.invoke(prompt)
        text = response.content.strip()
        good = text.lower().startswith("yes")
        feedback = text if not good else ""
        return {"good": good, "feedback": feedback}

    def _increment_attempt(state: QAState, runtime: Runtime) -> dict:
        return {"attempt": state.get("attempt", 0) + 1}

    def _append_qa(state: QAState, runtime: Runtime) -> dict:
        if state.get("good") and not state.get("done"):
            questions = state.get("questions", [])
            answers = state.get("answers", [])
            questions.append(state.get("current_question", ""))
            answers.append(state.get("current_answer", ""))
            # Keep track of accepted questions for diversity
            previous = state.get("previous_questions", [])
            return {"questions": questions, "answers": answers, "previous_questions": previous}
        return {}

    def _final(state: QAState, runtime: Runtime) -> dict:
        return {
            "questions": state.get("questions", []),
            "answers": state.get("answers", []),
        }

    # ------------------------------------------------------------------
    # assemble the graph
    # ------------------------------------------------------------------
    graph = StateGraph(
        state_schema=QAState, input_schema=QAInput, output_schema=QAOutput
    )

    graph.add_node("generate_candidate", _generate_candidate)
    graph.add_node("check_if_done", _check_if_done)
    graph.add_node("judge", _judge)
    graph.add_node("increment_attempt", _increment_attempt)
    graph.add_node("append_qa", _append_qa)
    graph.add_node("final", _final)

    graph.set_entry_point("generate_candidate")
    graph.add_edge("generate_candidate", "check_if_done")
    graph.add_edge("check_if_done", "judge")
    graph.add_edge("judge", "increment_attempt")
    graph.add_edge("increment_attempt", "append_qa")

    def _router(state: QAState) -> str:
        if state.get("done"):
            return "final"
        if state.get("good"):
            return "generate_candidate"
        if state.get("attempt", 0) < max_attempts:
            return "generate_candidate"
        # skip bad QA after max attempts
        return "generate_candidate"

    graph.add_conditional_edges("append_qa", _router, path_map=["generate_candidate", "final"])

    return graph.compile()


# convenience alias for external import
create_qa_generation_graph = create_qa_generation_graph

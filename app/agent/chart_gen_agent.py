"""Tool: Decide the best chart type for a given dataset and question."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.envsetup.env_loader import load_env
from app.prompt.prompts import sample_chart_prompt
from app.utils.llm_factory import create_chat_llm

logger = logging.getLogger(__name__)

load_env()


class ChartDecision(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "histogram", "pie", "box", "area"]
    x_axis: str
    y_axis: str
    reasoning: str


def decide_chart(question: str, context: dict | str) -> dict[str, Any]:
    """Decide the best chart type for a dataset.

    Parameters
    ----------
    question: The user's original question.
    context: Dataset summary (columns, dtypes, sample rows).

    Returns
    -------
    {"success": True, "chart": ChartDecision, "response": <AIMessage>} or {"success": False, "error": "..."}
    """
    try:
        llm = create_chat_llm(purpose="chart_generation", temperature=0)
        parser = JsonOutputParser(pydantic_object=ChartDecision)
        prompt = ChatPromptTemplate.from_template(sample_chart_prompt)
        messages = prompt.format_messages(question=question, context=context)
        response = llm.invoke(messages)
        parsed = parser.parse(str(response.content))
        decision = ChartDecision(**parsed)
        logger.debug("Chart decision: %s", decision)
        return {"success": True, "chart": decision, "response": response}
    except Exception as e:
        logger.exception("Chart decision failed")
        return {"success": False, "error": str(e)}


# --- Backward compatibility ---
class ChartAgent:
    """Deprecated: use decide_chart() directly."""

    def __init__(self):
        pass

    def run(self, question: str, context: dict | str) -> ChartDecision:
        result = decide_chart(question, context)
        if result["success"]:
            return result["chart"]
        raise RuntimeError(result["error"])


# ====== Unit Tests ======
if __name__ == "__main__":
    test_question = "show me year wise movie count as a bar chart"
    test_context = {
        "columns": ["year", "movie_count"],
        "dtypes": {"year": "int64", "movie_count": "int64"},
        "row_count": 10,
        "sample_rows": [
            {"year": 2020, "movie_count": 15},
            {"year": 2021, "movie_count": 22},
        ],
    }

    # Test new function API
    print("--- Testing decide_chart() ---")
    result = decide_chart(test_question, test_context)
    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Chart type: {result['chart'].chart_type}")
        print(f"X axis: {result['chart'].x_axis}")
        print(f"Y axis: {result['chart'].y_axis}")
        print(f"Reasoning: {result['chart'].reasoning}")
    else:
        print(f"Error: {result['error']}")

    # Test old class API
    print("\n--- Testing ChartAgent (old API) ---")
    agent = ChartAgent()
    old_result = agent.run(test_question, test_context)
    print(f"Result: {old_result}")

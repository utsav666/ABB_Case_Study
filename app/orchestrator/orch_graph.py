"""
LangGraph Agentic Workflow Orchestrator.

Same flow as orch_exec.py but with:
- Retry loop on SQL errors (max 2)
- Skip table identification if only 1 table
- Result verification
- Typed state flowing between nodes
- Each node has its own focused prompt

Flow:
  Router → Table Resolver → Generate SQL → Validate → Execute → (Verify → Chart) → END
                                  ↑              │
                                  └── retry ─────┘ (on error, max 2)
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langgraph.graph import END, StateGraph
from pandas import DataFrame

from app.agent.chart_gen_agent import decide_chart
from app.agent.fall_back import general_answer
from app.agent.schema_info_agent import explain_schema
from app.agent.sql_generation_agent import generate_sql
from app.agent.table_identification_agent import identify_tables
from app.envsetup.env_loader import load_env
from app.router.route_improvisation import Router
from app.utils.char_context import ChartContext
from app.utils.schema_summarizer import Summarizer
from app.utils.sql_executor import SqlExec
from app.utils.sql_validator import validator

logger = logging.getLogger(__name__)
load_env()

MAX_RETRIES = 2


# ============================================================
# State — flows between all nodes
# ============================================================


class GraphState(TypedDict, total=False):
    """Typed state that flows through the graph."""

    question: str
    route: str
    route_reasoning: str
    tables: list[str]
    sql: str
    sql_error: str
    retry_count: int
    data: Any  # DataFrame or None
    chart_figure: Any
    final_response: dict[str, Any]


# ============================================================
# Nodes — each does ONE thing
# ============================================================


def router_node(state: GraphState) -> GraphState:
    """Node 1: Classify user intent."""
    question = state["question"]
    router = Router()
    decision = router.route_question(question)
    logger.info("Router: route=%s, reasoning=%s", decision.route, decision.reasoning)
    return {
        "route": decision.route,
        "route_reasoning": decision.reasoning,
    }


def table_resolver_node(state: GraphState) -> GraphState:
    """Node 2: Identify relevant tables. Skip LLM if only 1 table exists."""
    question = state["question"]

    # Check how many tables exist in schema
    schema_summary = Summarizer.summarize_schema()
    # Quick check: if schema has only 1 table, skip LLM
    from app.sampledata.data_loader import DataLoader

    data = DataLoader.load_data()
    all_tables = list(data.get("tables", {}).keys())

    if len(all_tables) == 1:
        logger.info("Only 1 table in schema — skipping table identification LLM call")
        return {"tables": all_tables}

    # Multiple tables — use LLM to identify relevant ones
    result = identify_tables(question)
    if result["success"]:
        logger.info("Tables identified: %s", result["tables"])
        return {"tables": result["tables"]}
    else:
        logger.warning("Table identification failed: %s", result["error"])
        # Fallback: use all tables
        return {"tables": all_tables}


def generate_sql_node(state: GraphState) -> GraphState:
    """Node 3: Generate SQL using LLM."""
    question = state["question"]
    tables = state.get("tables", [])
    error_context = state.get("sql_error", "")

    result = generate_sql(question, tables, error_context=error_context)

    if result["success"]:
        logger.info("SQL generated: %s", result["sql"][:100])
        return {"sql": result["sql"], "sql_error": ""}
    else:
        logger.warning("SQL generation failed: %s", result["error"])
        return {"sql": "", "sql_error": result["error"]}


def validate_sql_node(state: GraphState) -> GraphState:
    """Node 4: Validate SQL syntax (no LLM — uses sqlglot)."""
    sql = state.get("sql", "")

    if not sql:
        return {"sql_error": "No SQL was generated."}

    val = validator(sql)
    if val is True:
        logger.info("SQL validation passed")
        return {"sql_error": ""}
    else:
        logger.warning("SQL validation failed: %s", val)
        return {"sql_error": f"SQL syntax error: {val}"}


def execute_sql_node(state: GraphState) -> GraphState:
    """Node 5: Execute SQL on database (no LLM)."""
    sql = state.get("sql", "")

    try:
        result = SqlExec().sql_executor(sql)
        if isinstance(result, DataFrame):
            logger.info("SQL executed: %d rows returned", len(result))
            return {"data": result, "sql_error": ""}
        else:
            logger.warning("SQL execution error: %s", result)
            return {"data": None, "sql_error": str(result)}
    except Exception as e:
        logger.exception("SQL execution exception")
        return {"data": None, "sql_error": str(e)}


def verify_result_node(state: GraphState) -> GraphState:
    """Node 6: Verify result makes sense (simple checks, no LLM)."""
    data = state.get("data")
    question = state.get("question", "")

    if data is None:
        return {}

    # Basic sanity checks
    warnings = []
    if len(data) == 0:
        warnings.append("Query returned 0 rows.")
    if data.isnull().all().any():
        null_cols = [col for col in data.columns if data[col].isnull().all()]
        warnings.append(f"Columns entirely NULL: {null_cols}")

    if warnings:
        logger.warning("Result verification warnings: %s", warnings)

    return {}


def chart_decision_node(state: GraphState) -> GraphState:
    """Node 7: Decide chart type (LLM)."""
    question = state["question"]
    data = state.get("data")

    if data is None or not isinstance(data, DataFrame):
        return {"final_response": {"type": "error", "message": "No data to chart."}}

    chart_context = ChartContext.build_chart_context(data)
    result = decide_chart(question, chart_context)

    if not result["success"]:
        # Chart decision failed — return data only
        return {
            "final_response": {"type": "data", "data": data, "sql": state.get("sql", "")}
        }

    decision = result["chart"]

    # Validate axes
    if decision.x_axis not in data.columns or decision.y_axis not in data.columns:
        logger.warning("Chart axes not in columns — returning data only")
        return {
            "final_response": {"type": "data", "data": data, "sql": state.get("sql", "")}
        }

    # Render chart
    fig, ax = plt.subplots()
    ct = decision.chart_type
    x, y = data[decision.x_axis], data[decision.y_axis]

    if ct == "line":
        ax.plot(x, y)
    elif ct == "bar":
        ax.bar(x, y)
    elif ct == "scatter":
        ax.scatter(x, y)
    elif ct == "histogram":
        ax.hist(x, bins=10)
    elif ct == "pie":
        ax.pie(y, labels=x.astype(str).tolist(), autopct="%1.1f%%")
    elif ct == "box":
        ax.boxplot(y)
    elif ct == "area":
        ax.fill_between(x, y)

    ax.set_xlabel(decision.x_axis)
    ax.set_ylabel(decision.y_axis)
    ax.set_title(f"{ct.title()} Chart")

    return {
        "chart_figure": fig,
        "final_response": {
            "type": "chart",
            "figure": fig,
            "data": data,
            "sql": state.get("sql", ""),
        },
    }


def schema_info_node(state: GraphState) -> GraphState:
    """Node: Answer schema questions (LLM)."""
    result = explain_schema(state["question"], context=state.get("route_reasoning", ""))
    if result["success"]:
        return {"final_response": {"type": "text", "message": result["answer"]}}
    return {"final_response": {"type": "error", "message": result["error"]}}


def general_answer_node(state: GraphState) -> GraphState:
    """Node: Answer general questions (LLM)."""
    result = general_answer(
        state["question"], router_reasoning=state.get("route_reasoning", "")
    )
    if result["success"]:
        return {"final_response": {"type": "text", "message": result["answer"]}}
    return {"final_response": {"type": "error", "message": result["error"]}}


def build_data_response_node(state: GraphState) -> GraphState:
    """Node: Package SQL result as final response."""
    data = state.get("data")
    sql = state.get("sql", "")

    if data is not None and isinstance(data, DataFrame):
        return {"final_response": {"type": "data", "data": data, "sql": sql}}
    return {
        "final_response": {
            "type": "error",
            "message": f"Query failed: {state.get('sql_error', 'Unknown error')}",
        }
    }


def increment_retry_node(state: GraphState) -> GraphState:
    """Node: Increment retry counter and set error context."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ============================================================
# Conditional Edges — decision logic (no LLM)
# ============================================================


def route_after_router(state: GraphState) -> str:
    """After router: which path to take?"""
    route = state.get("route", "")
    if route in ("sql_query", "sql_and_chart"):
        return "sql_path"
    elif route == "schema_info":
        return "schema_info"
    else:
        return "general"


def route_after_validate(state: GraphState) -> str:
    """After validation: proceed or retry?"""
    if not state.get("sql_error"):
        return "execute"
    # Has error — can we retry?
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "give_up"


def route_after_execute(state: GraphState) -> str:
    """After execution: success or retry?"""
    if state.get("data") is not None and isinstance(state.get("data"), DataFrame):
        return "success"
    # Execution failed — can we retry?
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "give_up"


def route_after_verify(state: GraphState) -> str:
    """After verification: need chart or return data?"""
    route = state.get("route", "")
    if route == "sql_and_chart":
        return "chart"
    return "data_response"


# ============================================================
# Build the Graph
# ============================================================


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("table_resolver", table_resolver_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("verify_result", verify_result_node)
    graph.add_node("chart_decision", chart_decision_node)
    graph.add_node("schema_info", schema_info_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_node("build_data_response", build_data_response_node)
    graph.add_node("increment_retry", increment_retry_node)

    # Entry point
    graph.set_entry_point("router")

    # Router → branching
    graph.add_conditional_edges("router", route_after_router, {
        "sql_path": "table_resolver",
        "schema_info": "schema_info",
        "general": "general_answer",
    })

    # SQL path: linear flow
    graph.add_edge("table_resolver", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    # After validation: proceed or retry
    graph.add_conditional_edges("validate_sql", route_after_validate, {
        "execute": "execute_sql",
        "retry": "increment_retry",
        "give_up": "build_data_response",
    })

    # After execution: success or retry
    graph.add_conditional_edges("execute_sql", route_after_execute, {
        "success": "verify_result",
        "retry": "increment_retry",
        "give_up": "build_data_response",
    })

    # Retry → back to generate_sql
    graph.add_edge("increment_retry", "generate_sql")

    # After verify: chart or data response
    graph.add_conditional_edges("verify_result", route_after_verify, {
        "chart": "chart_decision",
        "data_response": "build_data_response",
    })

    # Terminal nodes → END
    graph.add_edge("chart_decision", END)
    graph.add_edge("build_data_response", END)
    graph.add_edge("schema_info", END)
    graph.add_edge("general_answer", END)

    return graph


# ============================================================
# Orchestrator Class (same interface as orch_exec.py)
# ============================================================


# Compile graph once at module level
_compiled_graph = build_graph().compile()


class GraphOrchestrator:
    """LangGraph-based orchestrator. Same interface as Orchestrator in orch_exec.py."""

    def __init__(self, user_question: str):
        self.user_question = user_question

    def executor(self) -> dict[str, Any]:
        """Run the graph and return the result."""
        initial_state: GraphState = {
            "question": self.user_question,
            "route": "",
            "route_reasoning": "",
            "tables": [],
            "sql": "",
            "sql_error": "",
            "retry_count": 0,
            "data": None,
            "chart_figure": None,
            "final_response": {},
        }

        # Run the graph
        final_state = _compiled_graph.invoke(initial_state)

        # Return the final response
        response = final_state.get("final_response", {})
        if not response:
            return {"type": "error", "message": "No response generated."}
        return response


# ====== Unit Tests ======
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    print("=" * 60)
    print("TEST 1: SQL data query")
    print("=" * 60)
    r = GraphOrchestrator("list all movies released after 2015").executor()
    print(f"Type: {r.get('type')}")
    if r["type"] == "data":
        print(f"Rows: {len(r['data'])}")
        print(f"SQL: {r.get('sql', 'N/A')}")
    elif r["type"] == "error":
        print(f"Error: {r['message']}")
    else:
        print(f"Message: {str(r.get('message', ''))[:200]}")

    print("\n" + "=" * 60)
    print("TEST 2: Schema question")
    print("=" * 60)
    r = GraphOrchestrator("what tables are in the database?").executor()
    print(f"Type: {r.get('type')}")
    print(f"Message: {str(r.get('message', ''))[:200]}")

    print("\n" + "=" * 60)
    print("TEST 3: Chart question")
    print("=" * 60)
    r = GraphOrchestrator("show me the unique category").executor()
    print(f"Type: {r.get('type')}")
    if r["type"] == "chart":
        print("Chart generated!")
    elif r["type"] == "error":
        print(f"Error: {r['message']}")

    print("\n" + "=" * 60)
    print("TEST 4: General question")
    print("=" * 60)
    r = GraphOrchestrator("what is machine learning?").executor()
    print(f"Type: {r.get('type')}")
    print(f"Message: {str(r.get('message', ''))[:200]}")

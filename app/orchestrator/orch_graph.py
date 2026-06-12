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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.agent.chart_gen_agent import decide_chart
from app.agent.fall_back import general_answer
from app.agent.schema_info_agent import explain_schema
from app.agent.sql_generation_agent import generate_sql
from app.agent.table_identification_agent import identify_tables
from app.agent.value_resolver_agent import (  ### HITL
    get_column_distinct_values,
    get_fuzzy_matches,
    resolve_filter_value,
)
from app.envsetup.env_loader import load_env
from app.router.route_improvisation import Router
from app.utils.char_context import ChartContext
from app.utils.schema_summarizer import Summarizer
from app.utils.sql_executor import SqlExec
from app.utils.sql_parser import extract_where_filters, rewrite_sql_filter  ### HITL
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
    connection_id: str  # Selected DB connection scoping this run (optional)
    route: str
    route_reasoning: str
    tables: list[str]
    sql: str
    sql_error: str
    retry_count: int
    data: Any  # DataFrame or None
    chart_figure: Any
    final_response: dict[str, Any]
    ### HITL — additional state fields
    hitl_triggered: bool  # Whether HITL was triggered in this run
    hitl_type: str  # "cardinality_filter" or "chart_approval"
    chart_suggestion: dict[str, Any]  # Suggested chart config before human approval


# ============================================================
# Nodes — each does ONE thing
# ============================================================


def _connection_graph(state: GraphState):
    """Return the stored schema graph for the state's connection, or None."""
    connection_id = state.get("connection_id")
    if not connection_id:
        return None
    try:
        from app.target_db.controllers import load_schema_graph

        return load_schema_graph(connection_id)
    except Exception:  # noqa: BLE001 - fall back to default schema on any error
        logger.exception("Failed to load schema graph for connection %s", connection_id)
        return None


def _is_google_analytics(graph) -> bool:
    """Return True if the stored graph is a Google Analytics context."""
    return bool(graph) and graph.get("connection_type") == "google_analytics"


def router_node(state: GraphState) -> GraphState:
    """Node 1: Classify user intent."""
    question = state["question"]
    graph = _connection_graph(state)

    # Google Analytics connections always route to the GA agent — there is no
    # SQL path for them.
    if _is_google_analytics(graph):
        logger.info("Router: connection is Google Analytics → route=ga_query")
        return {"route": "ga_query", "route_reasoning": "Google Analytics connection."}

    # When a SQL connection is selected, route against its real schema graph.
    if graph is not None:
        from app.target_db.controllers import render_graph_summary

        router = Router(schema_summary=render_graph_summary(graph))
    else:
        router = Router()

    decision = router.route_question(question)
    logger.info("Router: route=%s, reasoning=%s", decision.route, decision.reasoning)
    return {
        "route": decision.route,
        "route_reasoning": decision.reasoning,
    }


def ga_agent_node(state: GraphState) -> GraphState:
    """Node: answer a question against a Google Analytics connection.

    Generates a GA4 runReport spec from the question + GA context, executes it,
    and packages the result rows in the same shape build_data_response uses.
    """
    question = state["question"]
    connection_id = state.get("connection_id", "")
    graph = _connection_graph(state)

    from app.agent.ga_query_agent import generate_ga_report_spec
    from app.ga_analytics.context_builder import render_ga_summary
    from app.ga_analytics.report_service import run_ga_report

    ga_context = render_ga_summary(graph) if graph else ""
    spec_result = generate_ga_report_spec(question, ga_context)
    if not spec_result["success"]:
        return {
            "final_response": {
                "type": "error",
                "message": f"Could not build the analytics query: {spec_result['error']}",
            }
        }

    try:
        df = run_ga_report(connection_id, spec_result["spec"])
    except Exception as error:  # noqa: BLE001 - surface to the user
        logger.exception("GA report execution failed")
        return {
            "final_response": {
                "type": "error",
                "message": f"Analytics query failed: {error}",
            }
        }

    return {
        "data": df.to_dict(orient="records"),
        "final_response": {
            "type": "data",
            "data": df.to_dict(orient="records"),
            "ga_spec": spec_result["spec"],
        },
    }


def table_resolver_node(state: GraphState) -> GraphState:
    """Node 2: Identify relevant tables. Skip LLM if only 1 table exists."""
    question = state["question"]

    # When a connection is selected, resolve tables from its schema graph.
    graph = _connection_graph(state)
    if graph is not None:
        from app.target_db.controllers import render_graph_summary

        all_tables = [node["qualified_name"] for node in graph.get("nodes", [])]
        if len(all_tables) <= 1:
            logger.info("Connection has <=1 table — skipping table identification")
            return {"tables": all_tables}
        # Identify against the connection's real schema, not sample data.
        result = identify_tables(
            question, schema_summary=render_graph_summary(graph)
        )
        if result["success"] and result["tables"]:
            logger.info("Tables identified: %s", result["tables"])
            return {"tables": result["tables"]}
        logger.warning(
            "Table identification empty/failed; using all connection tables"
        )
        return {"tables": all_tables}

    # Default (sample/config) schema path
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

    # When a connection is selected, feed the agent its real schema graph.
    graph = _connection_graph(state)
    if graph is not None:
        from app.target_db.controllers import render_graph_table_structure

        table_structure, all_tables = render_graph_table_structure(graph)
        result = generate_sql(
            question,
            tables or all_tables,
            error_context=error_context,
            table_structure=table_structure,
        )
    else:
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
            # ### HITL — Store as dict (records) for MemorySaver serialization
            return {"data": result.to_dict(orient="records"), "sql_error": ""}
        else:
            logger.warning("SQL execution error: %s", result)
            return {"data": None, "sql_error": str(result)}
    except Exception as e:
        logger.exception("SQL execution exception")
        return {"data": None, "sql_error": str(e)}


def _state_data_to_dataframe(data: Any) -> DataFrame | None:
    """### HITL — Helper: convert state['data'] (dict records) back to DataFrame."""
    if data is None:
        return None
    if isinstance(data, DataFrame):
        return data
    if isinstance(data, list):
        return DataFrame(data)
    return None


def verify_result_node(state: GraphState) -> GraphState:
    """Node 6: Verify result makes sense (simple checks, no LLM)."""
    data = _state_data_to_dataframe(state.get("data"))
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


# NOTE: chart_decision_node replaced by chart_approval_node (### HITL)
# The new node is defined below with interrupt() for human approval.


# ============================================================
# ### HITL — Cardinality Resolver Node
# ============================================================


def cardinality_resolver_node(state: GraphState) -> GraphState:
    """### HITL — Check if filter values can be confidently resolved by LLM.

    For each WHERE clause filter:
      1. Get column cardinality + sample values from DB
      2. Ask LLM: can you confidently resolve this value?
      3. If YES → auto-resolve, rewrite SQL
      4. If NO  → interrupt() → ask human to pick
    """
    sql = state.get("sql", "")
    if not sql:
        return {}

    # Parse WHERE clause filters
    filters = extract_where_filters(sql)
    if not filters:
        logger.info("### HITL — No WHERE filters found, skipping cardinality check")
        return {}

    rewritten_sql = sql

    for col_filter in filters:
        # Skip non-string equality/LIKE filters (numeric comparisons don't need HITL)
        if col_filter.operator not in ("=", "LIKE", "ILIKE", "IN"):
            continue

        # Determine table name (use first table from state if not qualified)
        table_name = col_filter.table or (
            state.get("tables", [""])[0] if state.get("tables") else ""
        )
        if not table_name:
            continue

        # Get actual distinct values and cardinality from DB
        actual_values, cardinality = get_column_distinct_values(
            table_name, col_filter.column
        )

        if cardinality == 0 or not actual_values:
            continue

        # Get user's filter value as string
        user_value = col_filter.filter_value
        if isinstance(user_value, list):
            user_value = user_value[0]  # Take first for IN clause
        user_value = str(user_value)

        # Sort by fuzzy similarity
        fuzzy_sorted = get_fuzzy_matches(user_value, actual_values)

        # Ask LLM for confidence
        resolution = resolve_filter_value(
            column_name=col_filter.column,
            table_name=table_name,
            user_value=user_value,
            actual_values=fuzzy_sorted,
            cardinality=cardinality,
        )

        if resolution["confidence"] == "high" and resolution["match"]:
            # LLM is confident → auto-resolve
            logger.info(
                "### HITL — Auto-resolved: '%s' → '%s' (confidence=high)",
                user_value,
                resolution["match"],
            )
            rewritten_sql = rewrite_sql_filter(
                rewritten_sql, col_filter, resolution["match"]
            )

        else:
            # LLM is NOT confident → trigger HITL interrupt
            logger.info(
                "### HITL — Low confidence for '%s' in column '%s'. Triggering HITL.",
                user_value,
                col_filter.column,
            )
            human_choice = interrupt(
                {
                    "hitl_type": "cardinality_filter",
                    "column": col_filter.column,
                    "table": table_name,
                    "user_typed": user_value,
                    "suggestions": resolution["alternatives"][:10],
                    "reasoning": resolution["reasoning"],
                    "message": (
                        f"I'm not sure which '{col_filter.column}' value you mean. "
                        f"You typed '{user_value}'. Please pick from the suggestions "
                        f"or type the exact value."
                    ),
                }
            )

            # Human responded — rewrite SQL with their choice
            rewritten_sql = rewrite_sql_filter(
                rewritten_sql, col_filter, str(human_choice)
            )
            logger.info("### HITL — Human selected: '%s'", human_choice)

    if rewritten_sql != sql:
        return {"sql": rewritten_sql, "hitl_triggered": True}
    return {}


# ============================================================
# ### HITL — Chart Approval Node
# ============================================================


def chart_approval_node(state: GraphState) -> GraphState:
    """### HITL — Ask human to approve or change the suggested chart type.

    LLM suggests a chart configuration, then we pause for human approval.
    Human can approve as-is or pick a different chart type/axes.
    """
    question = state["question"]
    data = _state_data_to_dataframe(state.get("data"))

    if data is None:
        return {"final_response": {"type": "error", "message": "No data to chart."}}

    # Get LLM's chart suggestion
    chart_context = ChartContext.build_chart_context(data)
    result = decide_chart(question, chart_context)

    if not result["success"]:
        return {
            "final_response": {
                "type": "data",
                "data": state.get("data"),
                "sql": state.get("sql", ""),
            }
        }

    decision = result["chart"]
    available_columns = list(data.columns)

    # ### HITL — Interrupt for human chart approval
    human_choice = interrupt(
        {
            "hitl_type": "chart_approval",
            "suggested_chart_type": decision.chart_type,
            "suggested_x_axis": decision.x_axis,
            "suggested_y_axis": decision.y_axis,
            "reasoning": decision.reasoning,
            "available_columns": available_columns,
            "available_chart_types": [
                "bar",
                "line",
                "scatter",
                "histogram",
                "pie",
                "box",
                "area",
            ],
            "message": (
                f"I suggest a **{decision.chart_type}** chart "
                f"(x={decision.x_axis}, y={decision.y_axis}). "
                f"Approve or choose differently."
            ),
        }
    )

    # Parse human response
    if isinstance(human_choice, dict):
        chart_type = human_choice.get("chart_type", decision.chart_type)
        x_axis = human_choice.get("x_axis", decision.x_axis)
        y_axis = human_choice.get("y_axis", decision.y_axis)
    elif isinstance(human_choice, str) and human_choice.lower() == "approve":
        chart_type = decision.chart_type
        x_axis = decision.x_axis
        y_axis = decision.y_axis
    else:
        # Assume it's a chart type string
        chart_type = str(human_choice)
        x_axis = decision.x_axis
        y_axis = decision.y_axis

    # Validate axes exist
    if x_axis not in data.columns or y_axis not in data.columns:
        logger.warning("### HITL — Chart axes not in columns after human choice")
        return {
            "final_response": {
                "type": "data",
                "data": state.get("data"),
                "sql": state.get("sql", ""),
            }
        }

    # ### HITL — Store chart config (serializable), render Figure in executor()
    logger.info(
        "### HITL — Chart approved: %s (x=%s, y=%s)", chart_type, x_axis, y_axis
    )

    return {
        "hitl_triggered": True,
        "final_response": {
            "type": "chart",
            "chart_config": {
                "chart_type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis,
            },
            "data": state.get("data"),  # already records (list of dicts)
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
    data = state.get("data")  # Already list of dicts (records)
    sql = state.get("sql", "")

    if data is not None:
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
    if route == "ga_query":
        return "ga_agent"
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
    data = state.get("data")
    if data is not None and isinstance(data, (DataFrame, list)):
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
    """Construct the LangGraph workflow.

    ### HITL — Two interrupt points added:
    1. cardinality_resolver — between generate_sql and validate_sql
    2. chart_approval — replaces chart_decision (adds human approval)
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("table_resolver", table_resolver_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("cardinality_resolver", cardinality_resolver_node)  ### HITL
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("verify_result", verify_result_node)
    graph.add_node(
        "chart_approval", chart_approval_node
    )  ### HITL (replaces chart_decision)
    graph.add_node("schema_info", schema_info_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_node("build_data_response", build_data_response_node)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("ga_agent", ga_agent_node)  # Google Analytics path

    # Entry point
    graph.set_entry_point("router")

    # Router → branching
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sql_path": "table_resolver",
            "schema_info": "schema_info",
            "general": "general_answer",
            "ga_agent": "ga_agent",
        },
    )

    # SQL path: linear flow
    graph.add_edge("table_resolver", "generate_sql")
    graph.add_edge(
        "generate_sql", "cardinality_resolver"
    )  ### HITL — check filter values
    graph.add_edge("cardinality_resolver", "validate_sql")  ### HITL — then validate

    # After validation: proceed or retry
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validate,
        {
            "execute": "execute_sql",
            "retry": "increment_retry",
            "give_up": "build_data_response",
        },
    )

    # After execution: success or retry
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "success": "verify_result",
            "retry": "increment_retry",
            "give_up": "build_data_response",
        },
    )

    # Retry → back to generate_sql
    graph.add_edge("increment_retry", "generate_sql")

    # After verify: chart or data response
    graph.add_conditional_edges(
        "verify_result",
        route_after_verify,
        {
            "chart": "chart_approval",  ### HITL — human approves chart
            "data_response": "build_data_response",
        },
    )

    # Terminal nodes → END
    graph.add_edge("chart_approval", END)  ### HITL
    graph.add_edge("build_data_response", END)
    graph.add_edge("schema_info", END)
    graph.add_edge("general_answer", END)
    graph.add_edge("ga_agent", END)  # Google Analytics path terminates here

    return graph


# ============================================================
# Orchestrator Class (same interface as orch_exec.py)
# ============================================================


# ### HITL — Compile graph with MemorySaver for interrupt/resume support
_checkpointer = MemorySaver()
_compiled_graph = build_graph().compile(checkpointer=_checkpointer)


class GraphOrchestrator:
    """LangGraph-based orchestrator. Same interface as Orchestrator in orch_exec.py.

    ### HITL — Now supports interrupt/resume:
    - Pass thread_id to maintain state across HITL pauses
    - Use resume() class method to continue after human input
    """

    def __init__(
        self,
        user_question: str,
        thread_id: str | None = None,
        connection_id: str | None = None,
    ):
        self.user_question = user_question
        # ### HITL — thread_id enables state persistence for interrupt/resume
        self.thread_id = thread_id or f"thread-{id(self)}"
        # Optional DB connection that scopes this run to a real schema graph.
        self.connection_id = connection_id

    def executor(self) -> dict[str, Any]:
        """Run the graph and return the result.

        ### HITL — If an interrupt is hit, returns a 'pending_hitl' response
        instead of the final answer. The caller must then call resume().
        """
        initial_state: GraphState = {
            "question": self.user_question,
            "connection_id": self.connection_id or "",
            "route": "",
            "route_reasoning": "",
            "tables": [],
            "sql": "",
            "sql_error": "",
            "retry_count": 0,
            "data": None,
            "chart_figure": None,
            "final_response": {},
            "hitl_triggered": False,
            "hitl_type": "",
            "chart_suggestion": {},
        }

        config = {"configurable": {"thread_id": self.thread_id}}

        # Run the graph (may pause at interrupt)
        final_state = _compiled_graph.invoke(initial_state, config)

        # ### HITL — Check if graph is paused at an interrupt
        pending = _compiled_graph.get_state(config)
        for task in pending.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                hitl_info = task.interrupts[0].value
                return {
                    "type": "pending_hitl",
                    "thread_id": self.thread_id,
                    **hitl_info,
                }

        # No interrupt — return final response
        response = final_state.get("final_response", {})
        if not response:
            return {"type": "error", "message": "No response generated."}
        return _hydrate_response(response)

    @classmethod
    def resume(cls, thread_id: str, human_input: Any) -> dict[str, Any]:
        """### HITL — Resume a paused graph after human provides input.

        Parameters
        ----------
        thread_id: The thread_id from the pending_hitl response.
        human_input: The human's decision (string, dict, etc.)

        Returns
        -------
        Final response dict, or another pending_hitl if multiple interrupts.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}

        # Resume the graph with human's response
        final_state = _compiled_graph.invoke(Command(resume=human_input), config)

        # Check if there's another interrupt (e.g., cardinality then chart)
        pending = _compiled_graph.get_state(config)
        for task in pending.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                hitl_info = task.interrupts[0].value
                return {
                    "type": "pending_hitl",
                    "thread_id": thread_id,
                    **hitl_info,
                }

        # Done — return final response
        response = final_state.get("final_response", {})
        if not response:
            return {"type": "error", "message": "No response generated."}
        return _hydrate_response(response)

    @staticmethod
    def _hydrate_response(response: dict[str, Any]) -> dict[str, Any]:
        """### HITL — Convert serializable state back to rich objects for the caller."""
        return _hydrate_response(response)


def _hydrate_response(response: dict[str, Any]) -> dict[str, Any]:
    """### HITL — Render chart Figure and convert data records to DataFrame.

    State stores everything as serializable (list of dicts, chart_config).
    This converts back to DataFrame + matplotlib Figure for the UI layer.
    """
    # Convert data records → DataFrame
    data = response.get("data")
    if isinstance(data, list) and data:
        response["data"] = DataFrame(data)

    # Render chart from config
    if response.get("type") == "chart" and "chart_config" in response:
        cfg = response.pop("chart_config")
        df = response.get("data")
        if df is not None and isinstance(df, DataFrame):
            chart_type = cfg["chart_type"]
            x_axis = cfg["x_axis"]
            y_axis = cfg["y_axis"]

            fig, ax = plt.subplots()
            x, y = df[x_axis], df[y_axis]

            if chart_type == "line":
                ax.plot(x, y)
            elif chart_type == "bar":
                ax.bar(x, y)
            elif chart_type == "scatter":
                ax.scatter(x, y)
            elif chart_type == "histogram":
                ax.hist(x, bins=10)
            elif chart_type == "pie":
                ax.pie(y, labels=x.astype(str).tolist(), autopct="%1.1f%%")
            elif chart_type == "box":
                ax.boxplot(y)
            elif chart_type == "area":
                ax.fill_between(x, y)

            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"{chart_type.title()} Chart")
            response["figure"] = fig

    return response


# ====== Unit Tests ======
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )

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

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
import time
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
    value_exists_exactly,
)
from app.envsetup.env_loader import get_llm_model, load_env
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
    schema_graph: dict[str, Any] | None  # Loaded ONCE per run; reused by every node
    conversation_history: str  # Prior conversation context from LangChain memory
    thread_id: str  # Conversation thread ID for LLM tracking
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
    ### LLM Cost Tracking
    user_message_id: str | None  # UUID of user message for LLM tracking
    ### Per-request LLM provider/model override (from agent/workspace settings)
    llm_provider: str | None  # e.g. "openai", "azure_openai", "claude_merck"
    llm_model: str | None  # e.g. "gpt-4o", "claude-opus-4-6-v1"


# ============================================================
# Nodes — each does ONE thing
# ============================================================


def _connection_graph(state: GraphState):
    """Return the schema graph for the state's connection, or None.

    The graph is loaded ONCE per run (in executor()/resume()) and stashed in
    state["schema_graph"], because it is immutable for the life of a request.
    Every node calls this, so re-querying the DB here would fetch the same
    bytes 3-5x — a multi-second waste against a remote settings DB.

    We use a sentinel: the key is absent until loaded. `None` is a valid loaded
    value (no connection / no graph), so we only fall back to a DB read when the
    key is missing entirely — e.g. a resume path that didn't seed it.
    """
    if "schema_graph" in state:
        return state["schema_graph"]
    return _load_schema_graph(state.get("connection_id"))


def _load_schema_graph(connection_id: str | None):
    """Read the stored schema graph for a connection from the DB, or None."""
    if not connection_id:
        # Loud, because everything downstream then falls back to the bundled
        # sample schema and will describe tables the user never connected to.
        logger.warning(
            "SCHEMA GRAPH — no connection_id on this run; downstream nodes will "
            "fall back to the bundled sample schema"
        )
        return None
    try:
        from app.target_db.controllers import load_schema_graph

        graph = load_schema_graph(connection_id)
        if not graph:
            logger.warning(
                "SCHEMA GRAPH — connection %s has NO stored graph; nodes will fall "
                "back to the bundled sample schema. Has this connection been profiled?",
                connection_id,
            )
            return None

        tables = [n.get("qualified_name") or n.get("table") for n in graph.get("nodes", [])]
        # Log the table names, not just a count: this is the line that makes a
        # wrong-database answer obvious at a glance.
        logger.info(
            "SCHEMA GRAPH — loaded %d table(s), %d relation(s) for connection %s: %s",
            len(tables),
            len(graph.get("edges", [])),
            connection_id,
            ", ".join(str(t) for t in tables[:15]) + (" …" if len(tables) > 15 else ""),
            extra={"table_count": len(tables), "relation_count": len(graph.get("edges", []))},
        )
        return graph
    except Exception:  # noqa: BLE001 - fall back to default schema on any error
        logger.exception("Failed to load schema graph for connection %s", connection_id)
        return None


def router_node(state: GraphState) -> GraphState:
    """Node 1: Classify user intent."""
    logger.info("NODE router — entering (classifying intent via LLM)")
    question = state["question"]
    graph = _connection_graph(state)
    _provider = state.get("llm_provider")
    _model = state.get("llm_model")

    # When a SQL connection is selected, route against its real schema graph.
    if graph is not None:
        from app.target_db.controllers import render_graph_summary

        router = Router(schema_summary=render_graph_summary(graph), provider=_provider, model=_model)
    else:
        router = Router(provider=_provider, model=_model)

    decision, response = router.route_question(question)

    # Track LLM call for routing
    if state.get("user_message_id"):
        from app.utils.llm_tracker import track_llm_call, extract_langchain_usage, get_model_from_langchain_response
        from app.settings_db.repositories import ConversationRepository

        usage = extract_langchain_usage(response)
        model = get_model_from_langchain_response(
            response, default=get_llm_model("router")
        )

        repo = ConversationRepository()
        track_llm_call(
            repo=repo,
            message_id=state["user_message_id"],
            thread_id=state.get("thread_id", ""),
            llm_purpose="routing",
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )

    logger.info("Router: route=%s, reasoning=%s", decision.route, decision.reasoning)
    return {
        "route": decision.route,
        "route_reasoning": decision.reasoning,
    }


def table_resolver_node(state: GraphState) -> GraphState:
    """Node 2: Identify relevant tables. Skip LLM if only 1 table exists."""
    logger.info("NODE table_resolver — entering")
    question = state["question"]
    _provider = state.get("llm_provider")
    _model = state.get("llm_model")

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
            question, schema_summary=render_graph_summary(graph),
            provider=_provider, model=_model,
        )

        # Track LLM call for table identification
        if result.get("success") and result.get("response") and state.get("user_message_id"):
            from app.utils.llm_tracker import track_llm_call, extract_langchain_usage, get_model_from_langchain_response
            from app.settings_db.repositories import ConversationRepository

            response = result["response"]
            usage = extract_langchain_usage(response)
            model = get_model_from_langchain_response(
                response, default=get_llm_model("table_identification")
            )

            repo = ConversationRepository()
            track_llm_call(
                repo=repo,
                message_id=state["user_message_id"],
                thread_id=state.get("thread_id", ""),
                llm_purpose="table-identifier",
                model=model,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
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
    result = identify_tables(question, provider=_provider, model=_model)

    # Track LLM call for table identification
    if result.get("success") and result.get("response") and state.get("user_message_id") :
        from app.utils.llm_tracker import track_llm_call, extract_langchain_usage, get_model_from_langchain_response
        from app.settings_db.repositories import ConversationRepository

        response = result["response"]
        usage = extract_langchain_usage(response)
        model = get_model_from_langchain_response(
            response, default=get_llm_model("table_identification")
        )

        repo = ConversationRepository()
        track_llm_call(
            repo=repo,
            message_id=state["user_message_id"],
            thread_id=state.get("thread_id", ""),
            llm_purpose="table-identifier",
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )

    if result["success"]:
        logger.info("Tables identified: %s", result["tables"])
        return {"tables": result["tables"]}
    else:
        logger.warning("Table identification failed: %s", result["error"])
        # Fallback: use all tables
        return {"tables": all_tables}


def generate_sql_node(state: GraphState) -> GraphState:
    """Node 3: Generate SQL using LLM."""
    logger.info("NODE generate_sql — entering (building table structure + LLM call)")
    question = state["question"]
    tables = state.get("tables", [])
    error_context = state.get("sql_error", "")
    conversation_history = state.get("conversation_history", "")
    _provider = state.get("llm_provider")
    _model = state.get("llm_model")

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
            conversation_history=conversation_history,
            provider=_provider,
            model=_model,
        )
    else:
        result = generate_sql(
            question,
            tables,
            error_context=error_context,
            conversation_history=conversation_history,
            provider=_provider,
            model=_model,
        )

    # Track LLM call for SQL generation
    if result.get("success") and state.get("user_message_id") :
        from app.utils.llm_tracker import track_llm_call
        from app.settings_db.repositories import ConversationRepository

        usage = result.get("usage", {})
        cost_info = result.get("cost", {})
        retry_count = state.get("retry_count", 0)

        repo = ConversationRepository()
        track_llm_call(
            repo=repo,
            message_id=state["user_message_id"],
            thread_id=state.get("thread_id", ""),
            llm_purpose="sql-generator",
            model=cost_info.get("model", "gpt-4"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            metadata={"retry_attempt": retry_count} if retry_count > 0 else None,
        )

    if result["success"]:
        # Log the FULL statement. Truncating at 100 chars hid the interesting
        # part of every non-trivial query — the JOINs and the WHERE clause are
        # exactly what you need when an answer looks wrong.
        logger.info(
            "SQL generated (attempt %d):\n%s",
            state.get("retry_count", 0) + 1,
            result["sql"],
            extra={"sql": result["sql"]},
        )
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
    started = time.perf_counter()

    try:
        result = SqlExec().sql_executor(sql)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if isinstance(result, DataFrame):
            row_count = len(result)
            # A zero-row result is the signature of a wrong filter value or an
            # INNER JOIN that should have been LEFT — worth a WARNING, not an
            # INFO buried among successes.
            log = logger.warning if row_count == 0 else logger.info
            log(
                "SQL executed in %.0f ms — %d row(s), %d column(s)%s",
                elapsed_ms,
                row_count,
                len(result.columns),
                " — ZERO ROWS: check filter values and join types" if row_count == 0 else "",
                extra={"row_count": row_count, "duration_ms": round(elapsed_ms, 1)},
            )
            # ### HITL — Store as dict (records) for MemorySaver serialization
            return {"data": result.to_dict(orient="records"), "sql_error": ""}
        else:
            logger.warning(
                "SQL execution error after %.0f ms: %s\nSQL was:\n%s",
                elapsed_ms, result, sql,
            )
            return {"data": None, "sql_error": str(result)}
    except Exception as e:
        logger.exception(
            "SQL execution raised after %.0f ms. SQL was:\n%s",
            (time.perf_counter() - started) * 1000, sql,
        )
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
    # Resolve table schemas from settings.schema_graphs, not the legacy JSON
    # fixture. Loaded once per run and reused for every filter below.
    schema_graph = _connection_graph(state)

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

        # Get user's filter value as string
        user_value = col_filter.filter_value
        if isinstance(user_value, list):
            user_value = user_value[0]  # Take first for IN clause
        user_value = str(user_value)

        # THE gate: does the typed value already match a value in the column?
        # This is the only signal that decides whether disambiguation is worth
        # doing, and it is TYPE-AGNOSTIC — year (INTEGER), pincode (INTEGER or
        # TEXT) and country (TEXT) are all checked identically. An exact match
        # means there is nothing to disambiguate → skip fuzzy/LLM/HITL. Only a
        # miss (typo, alias, wrong casing) falls through to resolution.
        if value_exists_exactly(
            table_name, col_filter.column, user_value, schema_graph=schema_graph
        ):
            logger.info(
                "### HITL — '%s' exact-matches '%s.%s'; no disambiguation needed",
                user_value,
                table_name,
                col_filter.column,
            )
            continue

        # No exact match → fetch distinct values to resolve against
        actual_values, cardinality = get_column_distinct_values(
            table_name, col_filter.column, schema_graph=schema_graph
        )

        if cardinality == 0 or not actual_values:
            continue

        # Sort by fuzzy similarity. Values with no real resemblance are dropped,
        # so this can come back empty — string similarity cannot see aliases
        # ("usa" vs "United States"). When it does, hand the LLM a plain sample of
        # real values instead: it CAN resolve aliases, and it's the only step that
        # can. What we must not do is let the fuzzy scorer's noise floor pass off
        # an unrelated value as a near miss.
        fuzzy_sorted = get_fuzzy_matches(user_value, actual_values)
        candidates = fuzzy_sorted or [str(v) for v in actual_values[:20]]
        has_near_miss = bool(fuzzy_sorted)

        # Ask LLM for confidence
        resolution = resolve_filter_value(
            column_name=col_filter.column,
            table_name=table_name,
            user_value=user_value,
            actual_values=candidates,
            cardinality=cardinality,
            provider=state.get("llm_provider"),
            model=state.get("llm_model"),
        )

        # Only auto-resolve to a value that genuinely exists in the column. A
        # confident-sounding match the LLM invented would silently produce a
        # zero-row answer with no hint that the filter was fabricated.
        actual_value_set = {str(v) for v in actual_values}
        match_is_real = str(resolution.get("match") or "") in actual_value_set
        if resolution["confidence"] == "high" and resolution["match"] and match_is_real:
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
            if resolution["match"] and not match_is_real:
                logger.warning(
                    "### HITL — Discarding hallucinated match '%s' (not present in %s.%s)",
                    resolution["match"],
                    table_name,
                    col_filter.column,
                )
            # LLM is NOT confident → trigger HITL interrupt
            logger.info(
                "### HITL — Low confidence for '%s' in column '%s'. Triggering HITL.",
                user_value,
                col_filter.column,
            )
            # Always give the user something real to pick. Prefer the LLM's
            # alternatives, but keep only values that exist in the column, then
            # fall back to actual candidates. De-dupe while preserving order.
            suggestions = [
                s
                for s in (resolution.get("alternatives") or [])
                if s and str(s) in actual_value_set
            ]
            if not suggestions:
                suggestions = candidates[:10]
            seen: set[str] = set()
            deduped_suggestions: list[str] = []
            for value in suggestions:
                value_str = str(value)
                if value_str not in seen:
                    seen.add(value_str)
                    deduped_suggestions.append(value_str)
            suggestions = deduped_suggestions[:10]

            logger.info(
                "### HITL — Offering %d suggestion(s) for column '%s'",
                len(suggestions),
                col_filter.column,
            )
            human_choice = interrupt(
                {
                    "hitl_type": "cardinality_filter",
                    "column": col_filter.column,
                    "table": table_name,
                    "user_typed": user_value,
                    "suggestions": suggestions,
                    "reasoning": resolution["reasoning"],
                    # Say which situation this is. "I'm not sure which you mean"
                    # implies a near miss; when nothing resembles their input,
                    # claiming ambiguity is misleading — the value simply isn't there.
                    "message": (
                        (
                            f"I'm not sure which '{col_filter.column}' value you "
                            f"mean. You typed '{user_value}'. Please pick from the "
                            f"suggestions or type the exact value."
                        )
                        if has_near_miss
                        else (
                            f"I couldn't find anything like '{user_value}' in "
                            f"'{col_filter.column}'. Here's what that column "
                            f"actually contains — pick one, or type an exact value."
                        )
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
    """Auto-generate and approve chart based on LLM suggestion.

    LLM analyzes the data and question to suggest the best chart type,
    then automatically generates it without requiring approval.
    Users can request changes via follow-up questions if needed.
    """
    question = state["question"]
    data = _state_data_to_dataframe(state.get("data"))

    if data is None:
        return {"final_response": {"type": "error", "message": "No data to chart."}}

    # Detect and pivot year-over-year comparison data
    def detect_and_pivot_yoy_comparison(df, question_text):
        """Detect YoY comparison and pivot data for multi-series chart."""
        import pandas as pd
        import re

        # Check if question indicates comparison
        comparison_keywords = ['compare', 'vs', 'versus', 'compared to', 'last year', 'previous year', 'year over year', 'yoy']
        is_comparison = any(keyword in question_text.lower() for keyword in comparison_keywords)

        if not is_comparison:
            return df, False

        # Check if data is already pivoted (has multiple numeric columns)
        # SQL agent might have already done the pivot with CASE WHEN
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            # Data is already pivoted, just format it for multi-series chart
            logger.info(f"Data already pivoted by SQL: {len(numeric_cols)} metric columns detected")
            # Find the categorical column (likely date-based)
            cat_cols = [col for col in df.columns if col not in numeric_cols]
            if cat_cols:
                return df, True  # Return as-is, already in the right format

        return df, False

        # Check if data has a date column with multiple time periods
        date_col = None
        comparison_type = None  # 'year' or 'month'

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'day' in col.lower() or 'month' in col.lower():
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    if dates.notna().any():
                        years = dates.dt.year.dropna().unique()
                        months = dates.dt.to_period('M').dropna().unique()

                        # Check for year-over-year comparison
                        if len(years) >= 2:
                            date_col = col
                            comparison_type = 'year'
                            break
                        # Check for month-over-month comparison (same year, different months)
                        elif len(months) >= 2 and len(years) == 1:
                            date_col = col
                            comparison_type = 'month'
                            break
                except Exception:
                    continue

        if not date_col or not comparison_type:
            return df, False

        # Find the metric column (numeric, not the date)
        metric_col = None
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                metric_col = col
                break

        if not metric_col:
            return df, False

        # Pivot the data based on comparison type
        df_copy = df.copy()
        df_copy['date_parsed'] = pd.to_datetime(df_copy[date_col])

        if comparison_type == 'year':
            # Year-over-year: compare same days across different years
            df_copy['year'] = df_copy['date_parsed'].dt.year
            df_copy['month_day'] = df_copy['date_parsed'].dt.strftime('%m-%d')

            # Pivot: rows = month_day, columns = year, values = metric
            pivoted = df_copy.pivot_table(
                index='month_day',
                columns='year',
                values=metric_col,
                aggfunc='mean'
            ).reset_index()

            # Rename columns to indicate years
            pivoted.columns = [str(col) if col == 'month_day' else f"{metric_col}_{col}" for col in pivoted.columns]
            logger.info(f"Pivoted YoY comparison data: {len(df)} rows → {len(pivoted)} rows, {len(pivoted.columns)} columns")

        elif comparison_type == 'month':
            # Month-over-month: compare same days across different months
            df_copy['month_name'] = df_copy['date_parsed'].dt.strftime('%B')  # Full month name
            df_copy['day_of_month'] = df_copy['date_parsed'].dt.day

            # Pivot: rows = day_of_month, columns = month_name, values = metric
            pivoted = df_copy.pivot_table(
                index='day_of_month',
                columns='month_name',
                values=metric_col,
                aggfunc='mean'
            ).reset_index()

            # Rename columns to indicate months
            pivoted.columns = [str(col) if col == 'day_of_month' else f"{metric_col}_{col.lower()}_2018" for col in pivoted.columns]
            logger.info(f"Pivoted MoM comparison data: {len(df)} rows → {len(pivoted)} rows, {len(pivoted.columns)} columns")

        return pivoted, True

    data, is_pivoted = detect_and_pivot_yoy_comparison(data, question)

    # Get column logical names from schema for better labels
    connection_id = state.get("connection_id")
    column_labels = {}
    if connection_id:
        try:
            graph_json = _connection_graph(state)
            if graph_json:
                # Parse JSON if it's a string
                if isinstance(graph_json, str):
                    import json
                    graph_json = json.loads(graph_json)

                # Extract column display names from schema graph
                # Graph structure: {"nodes": [{"table": "...", "columns": [{"raw_name": "...", "logical_name": "..."}]}]}
                for node in graph_json.get("nodes", []):
                    for col_info in node.get("columns", []):
                        raw_name = col_info.get("raw_name", "")
                        logical_name = col_info.get("logical_name", "")
                        if logical_name and logical_name != raw_name:
                            column_labels[raw_name.lower()] = logical_name
        except Exception as e:
            logger.warning("Could not fetch column labels: %s", e)

    # Get LLM's chart suggestion
    chart_context = ChartContext.build_chart_context(data)
    result = decide_chart(question, chart_context, provider=state.get("llm_provider"), model=state.get("llm_model"))

    # Track LLM call for chart generation
    if result.get("success") and result.get("response") and state.get("user_message_id") :
        from app.utils.llm_tracker import track_llm_call, extract_langchain_usage, get_model_from_langchain_response
        from app.settings_db.repositories import ConversationRepository

        response = result["response"]
        usage = extract_langchain_usage(response)
        model = get_model_from_langchain_response(
            response, default=get_llm_model("chart_generation")
        )

        repo = ConversationRepository()
        track_llm_call(
            repo=repo,
            message_id=state["user_message_id"],
            thread_id=state.get("thread_id", ""),
            llm_purpose="chart-generator",
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )

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

    # Override chart config for pivoted comparison data
    if is_pivoted:
        # Find categorical (X-axis) and numeric (Y-axis) columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        cat_cols = [col for col in data.columns if col not in numeric_cols]

        # Check if it's YoY (month_day) or MoM (day_of_month) comparison
        if 'month_day' in data.columns:
            # Year-over-year: convert month_day (05-01) to day number (1)
            data['day_label'] = data['month_day'].apply(lambda x: str(int(x.split('-')[1])))
            x_axis = 'day_label'
            y_axis = [col for col in numeric_cols]  # All numeric columns are series
        elif 'day_of_month' in data.columns:
            # Month-over-month: use day_of_month directly
            x_axis = 'day_of_month'
            y_axis = [col for col in numeric_cols]  # All numeric columns are series
        else:
            # Already pivoted by SQL: use first categorical column as X
            x_axis = cat_cols[0] if cat_cols else data.columns[0]
            y_axis = numeric_cols  # All numeric columns are series

        chart_type = 'line'  # Always use line for time comparisons
        y_axis_secondary = None
        logger.info(f"Comparison detected: x={x_axis}, y={y_axis}")
    else:
        # Auto-approve the chart (HITL disabled for smoother UX)
        # User can still modify chart via follow-up questions if needed
        chart_type = decision.chart_type
        x_axis = decision.x_axis
        y_axis = decision.y_axis
        y_axis_secondary = getattr(decision, 'y_axis_secondary', None)

    # Validate axes exist in data
    # Handle both single y_axis and list (for multi-series charts)
    y_axes_to_check = y_axis if isinstance(y_axis, list) else [y_axis]
    if y_axis_secondary:
        y_axes_to_check.append(y_axis_secondary)

    if x_axis not in data.columns or not all(y in data.columns for y in y_axes_to_check if y):
        logger.warning("Chart axes not in columns, falling back to table view")
        return {
            "final_response": {
                "type": "data",
                "data": state.get("data"),
                "sql": state.get("sql", ""),
            }
        }

    # Chart auto-approved
    logger.info(
        "Chart auto-approved: %s (x=%s, y=%s%s)",
        chart_type, x_axis, y_axis,
        f", y2={y_axis_secondary}" if y_axis_secondary else ""
    )

    chart_config = {
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis,
    }
    # Add secondary axis for dual-axis charts
    if y_axis_secondary:
        chart_config["y_axis_secondary"] = y_axis_secondary

    # Detect date granularity for smart X-axis formatting
    def detect_date_granularity(df, col_name):
        """Detect if column is date and what granularity (day/week/month/year)."""
        try:
            import pandas as pd
            if col_name not in df.columns:
                return None

            col = df[col_name]
            # Try to parse as datetime
            if pd.api.types.is_datetime64_any_dtype(col):
                dates = col
            else:
                try:
                    dates = pd.to_datetime(col, errors='coerce')
                except Exception:
                    return None

            # Filter out NaT values
            dates = dates.dropna()
            if len(dates) < 2:
                return None

            # Sort and calculate average difference
            dates_sorted = dates.sort_values()
            diffs = dates_sorted.diff().dropna()
            avg_diff_days = diffs.mean().total_seconds() / (60 * 60 * 24)

            # Classify granularity
            if avg_diff_days < 2:
                return 'day'
            elif avg_diff_days < 10:
                return 'week'
            elif avg_diff_days < 60:
                return 'month'
            else:
                return 'year'
        except Exception:
            return None

    x_date_format = detect_date_granularity(data, x_axis)
    if x_date_format:
        chart_config["x_axis_format"] = x_date_format

    # Get display configuration from connection settings
    def get_display_config():
        """Fetch display_config from the connection settings."""
        if not connection_id:
            return None
        try:
            from app.settings_db.repositories import DBConnectionRepository
            conn = DBConnectionRepository().get_connection_by_id(connection_id)
            return conn.get("display_config") if conn else None
        except Exception:
            return None

    display_config = get_display_config()

    # Detect number formatting for Y-axis
    def detect_number_format(df, col_names, config):
        """Detect appropriate number formatting based on value ranges and config."""
        try:
            import pandas as pd

            # If config disables number formatting, return None
            if config and not config.get("number_format_enabled", True):
                return None

            # Collect all numeric values from specified columns
            all_values = []
            for col in col_names:
                if col and col in df.columns:
                    col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    all_values.extend(col_data.values)

            if not all_values:
                return None

            max_val = max(abs(v) for v in all_values)

            # Use thresholds and suffixes from config, or defaults
            if config:
                thresholds = config.get("number_format_thresholds", {})
                suffixes = config.get("number_format_suffixes", {})
                decimals = config.get("decimal_places", 1)
            else:
                thresholds = {"billion": 1e9, "million": 1e6, "thousand": 1e3}
                suffixes = {"billion": "B", "million": "M", "thousand": "K"}
                decimals = 1

            # Determine format based on max value
            if max_val >= thresholds.get("billion", 1e9):
                return {'suffix': suffixes.get("billion", "B"), 'divisor': thresholds.get("billion", 1e9), 'decimals': decimals}
            elif max_val >= thresholds.get("million", 1e6):
                return {'suffix': suffixes.get("million", "M"), 'divisor': thresholds.get("million", 1e6), 'decimals': decimals}
            elif max_val >= thresholds.get("thousand", 1e3):
                return {'suffix': suffixes.get("thousand", "K"), 'divisor': thresholds.get("thousand", 1e3), 'decimals': decimals}
            else:
                return {'suffix': '', 'divisor': 1, 'decimals': 0}
        except Exception:
            return None

    # Collect all Y-axis columns
    y_cols = []
    if isinstance(y_axis, list):
        y_cols.extend(y_axis)
    else:
        y_cols.append(y_axis)
    if y_axis_secondary:
        y_cols.append(y_axis_secondary)

    y_number_format = detect_number_format(data, y_cols, display_config)
    if y_number_format:
        chart_config["y_axis_format"] = y_number_format

    # Add color configuration from display_config
    if display_config and "chart_colors" in display_config:
        chart_config["colors"] = display_config["chart_colors"]
    else:
        # Default colors if not configured
        chart_config["colors"] = {
            "primary": "#3b82f6",
            "secondary": "#ef4444",
            "palette": ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
        }

    # Add human-readable labels from column configuration
    # These will override the automatic snake_case → Title Case formatting
    if column_labels:
        def get_display_name(col_name):
            """Get display name for a column, stripping aggregation prefixes."""
            col_lower = col_name.lower()
            agg_type = None

            # Try exact match first
            if col_lower in column_labels:
                return column_labels[col_lower]

            # Strip common aggregation prefixes and try again
            # Map abbreviated prefixes to their display names
            prefix_map = {
                'average_': 'Average',
                'total_': 'Total',
                'sum_': 'Sum',
                'avg_': 'Average',
                'count_': 'Count',
                'max_': 'Maximum',
                'min_': 'Minimum',
            }

            for prefix, agg_display in prefix_map.items():
                if col_lower.startswith(prefix):
                    agg_type = agg_display
                    base_name = col_lower[len(prefix):]

                    # Try exact match on base name
                    if base_name in column_labels:
                        return f"{agg_type} {column_labels[base_name]}"

                    # Try partial match: find columns ending with base_name
                    # e.g., "sales" matches "actual_sales"
                    for col_key, display_name in column_labels.items():
                        if col_key.endswith(base_name) or base_name in col_key:
                            return f"{agg_type} {display_name}"

                    break  # Found aggregation prefix, don't try others

            return None

        labels = {}
        # Map each axis to its display name
        # For comparisons with day labels, use "Day" as X-axis label
        if is_pivoted and x_axis in ['day_label', 'day_of_month']:
            labels["x_axis_label"] = "Day"
        else:
            display = get_display_name(x_axis)
            if display:
                labels["x_axis_label"] = display

        # Handle y_axis (could be string or list)
        y_axes = y_axis if isinstance(y_axis, list) else [y_axis]
        for i, y in enumerate(y_axes):
            display = get_display_name(y)
            if display:
                key = "y_axis_label" if i == 0 else f"y_axis_{i}_label"
                labels[key] = display

        if y_axis_secondary:
            display = get_display_name(y_axis_secondary)
            if display:
                labels["y_axis_secondary_label"] = display

        if labels:
            chart_config["labels"] = labels

    # Convert DataFrame back to records for response
    # If data was pivoted, use the pivoted DataFrame; otherwise use original
    response_data = data.to_dict('records') if is_pivoted else state.get("data")

    return {
        "hitl_triggered": False,  # No longer requiring approval
        "final_response": {
            "type": "chart",
            "chart_config": chart_config,
            "data": response_data,  # records (list of dicts)
            "sql": state.get("sql", ""),
        },
    }


def schema_info_node(state: GraphState) -> GraphState:
    """Node: Answer schema questions (LLM).

    Passes the CONNECTED database's schema graph. Without it, explain_schema
    falls back to the bundled movie-database fixture and confidently describes
    tables the user has never connected to.
    """
    result = explain_schema(
        state["question"],
        context=state.get("route_reasoning", ""),
        provider=state.get("llm_provider"),
        model=state.get("llm_model"),
        schema_graph=_connection_graph(state),
    )
    if result["success"]:
        return {"final_response": {"type": "text", "message": result["answer"]}}
    return {"final_response": {"type": "error", "message": result["error"]}}


def explanation_node(state: GraphState) -> GraphState:
    """Node: Answer "why is X high/low?" with grounded comparative analysis.

    A single SELECT cannot answer a "why" question — you need the focal value,
    a baseline of peers to judge it against, and a breakdown of what drove the
    gap. analyze_why() runs those queries and does the arithmetic in pandas,
    so the narrative only ever restates verified numbers.
    """
    logger.info("NODE explanation — entering (comparative analysis)")
    from app.agent.analysis_agent import analyze_why
    from app.target_db.controllers import render_graph_summary

    graph = _connection_graph(state)
    if not graph:
        return {
            "final_response": {
                "type": "error",
                "message": (
                    "I need a connected database to analyse this. Please select a "
                    "connection and try again."
                ),
            }
        }

    result = analyze_why(
        state["question"],
        schema_summary=render_graph_summary(graph),
        provider=state.get("llm_provider"),
        model=state.get("llm_model"),
        # "Why was Nov high?" is nearly always a follow-up; without the prior
        # turns the planner drops filters the user already established.
        conversation_history=state.get("conversation_history", ""),
    )

    # Track both LLM calls (plan + synthesis) so cost reporting stays accurate.
    if state.get("user_message_id"):
        from app.settings_db.repositories import ConversationRepository
        from app.utils.llm_tracker import (
            extract_langchain_usage,
            get_model_from_langchain_response,
            track_llm_call,
        )

        for index, response in enumerate(result.get("responses") or []):
            usage = extract_langchain_usage(response)
            track_llm_call(
                repo=ConversationRepository(),
                message_id=state["user_message_id"],
                thread_id=state.get("thread_id", ""),
                llm_purpose="analysis-plan" if index == 0 else "analysis-synthesis",
                model=get_model_from_langchain_response(
                    response, default=get_llm_model("analysis")
                ),
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )

    if not result["success"]:
        return {"final_response": {"type": "error", "message": result["error"]}}

    # Return the period series as data so the UI can chart the comparison the
    # explanation refers to — the numbers and the picture stay in sync.
    evidence = result.get("evidence") or {}
    return {
        "sql": "\n\n".join(result.get("sql") or []),
        "final_response": {
            "type": "data",
            "message": result["answer"],
            "data": evidence.get("period_series") or [],
            "sql": "\n\n".join(result.get("sql") or []),
            # Surfaced so the UI can show which scope the analysis actually used.
            # A follow-up that inherited "Product A" looks identical to one that
            # silently didn't, unless we say so.
            "analysis_scope": {
                "metric": result.get("metric_description") or "",
                "inherited_filters": result.get("inherited_filters") or [],
            },
        },
    }


def general_answer_node(state: GraphState) -> GraphState:
    """Node: Answer general questions (LLM)."""
    result = general_answer(
        state["question"], router_reasoning=state.get("route_reasoning", ""),
        provider=state.get("llm_provider"), model=state.get("llm_model"),
    )

    # Track LLM call for fallback/general answers
    if result.get("success") and result.get("response") and state.get("user_message_id") :
        from app.utils.llm_tracker import track_llm_call, extract_langchain_usage, get_model_from_langchain_response
        from app.settings_db.repositories import ConversationRepository

        response = result["response"]
        usage = extract_langchain_usage(response)
        model = get_model_from_langchain_response(
            response, default=get_llm_model("general_answer")
        )

        repo = ConversationRepository()
        track_llm_call(
            repo=repo,
            message_id=state["user_message_id"],
            thread_id=state.get("thread_id", ""),
            llm_purpose="fallback-answer",
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )

    if result["success"]:
        return {"final_response": {"type": "text", "message": result["answer"]}}
    return {"final_response": {"type": "error", "message": result["error"]}}


def build_data_response_node(state: GraphState) -> GraphState:
    """Node: Package SQL result as final response."""
    data = state.get("data")  # Already list of dicts (records)
    sql = state.get("sql", "")

    if data is not None:
        # A zero-row result is a SUCCESS, not an error — but "Here are the query
        # results." with an empty table below it reads as a broken app. Say so
        # explicitly instead, so the user knows the filter matched nothing.
        row_count = 0 if data is None else len(data)
        if row_count == 0:
            return {
                "final_response": {
                    "type": "data",
                    "data": data,
                    "sql": sql,
                    "message": (
                        "The query ran successfully but returned no rows — "
                        "nothing in the data matches those filters."
                    ),
                }
            }
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
    elif route == "explanation":
        # Previously fell through to "general" — an LLM with NO database access —
        # so "why was Nov high?" got an answer that could not see the data.
        return "explanation"
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
    graph.add_node("explanation", explanation_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_node("build_data_response", build_data_response_node)
    graph.add_node("increment_retry", increment_retry_node)

    # Entry point
    graph.set_entry_point("router")

    # Router → branching
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sql_path": "table_resolver",
            "schema_info": "schema_info",
            "explanation": "explanation",
            "general": "general_answer",
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
    graph.add_edge("explanation", END)
    graph.add_edge("general_answer", END)

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
        conversation_history: str = "",
        user_message_id: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        self.user_question = user_question
        # ### HITL — thread_id enables state persistence for interrupt/resume
        self.thread_id = thread_id or f"thread-{id(self)}"
        # Optional DB connection that scopes this run to a real schema graph.
        self.connection_id = connection_id
        # Prior conversation history formatted as text for LLM context
        self.conversation_history = conversation_history
        # LLM cost tracking
        self.user_message_id = user_message_id
        # Per-request LLM provider/model override from agent/workspace settings
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def executor(self) -> dict[str, Any]:
        """Run the graph and return the result.

        ### HITL — If an interrupt is hit, returns a 'pending_hitl' response
        instead of the final answer. The caller must then call resume().
        """
        initial_state: GraphState = {
            "question": self.user_question,
            "thread_id": self.thread_id,
            "connection_id": self.connection_id or "",
            # Load the schema graph ONCE here; every node reads it from state
            # instead of re-querying the (remote) settings DB per node.
            "schema_graph": _load_schema_graph(self.connection_id),
            "conversation_history": self.conversation_history,
            "user_message_id": self.user_message_id,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
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
    """### HITL — Convert data records to DataFrame for legacy compatibility.

    State stores everything as serializable (list of dicts, chart_config).
    For charts, we now keep chart_config and send it to frontend as-is.
    Data stays as list of dicts for frontend rendering.
    """
    # For non-chart responses, convert data records → DataFrame
    if response.get("type") != "chart":
        data = response.get("data")
        if isinstance(data, list) and data:
            response["data"] = DataFrame(data)

    # For chart responses: keep chart_config and data as list of dicts
    # Frontend will handle rendering via Recharts
    # No matplotlib rendering needed anymore

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

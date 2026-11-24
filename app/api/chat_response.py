"""Response shaping for the chat API.

Extracted from main.py, which had grown to ~1600 lines mixing 23 route handlers
with the pure functions that shape their responses. Everything here is
side-effect free apart from one settings-DB read (display_config), so it is
directly unit-testable — see tests/test_filters_and_formatting.py.

Covers three concerns the UI depends on:
  * applied_filters  — describe the executed WHERE clause so an answer states
                       which slice of data it covers
  * number_format    — one display scale per response, from the connection's
                       display_config, so tables/charts/prose agree
  * persisted content — the payload the frontend re-renders from on reload
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def serialize_dataframe(data_frame: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame into a JSON-friendly structure.

    Args:
        data_frame: Pandas DataFrame returned by the orchestrator.

    Returns:
        A dictionary containing rows, columns, and row count.
    """
    safe_frame = data_frame.astype(object).where(data_frame.notna(), None)
    return {
        "columns": safe_frame.columns.tolist(),
        "rows": safe_frame.to_dict(orient="records"),
        "row_count": len(safe_frame.index),
    }


def serialize_figure_to_base64(figure: Any) -> str | None:
    """Encode a Matplotlib figure as a base64 PNG string."""
    if figure is None:
        return None

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _humanize_identifier(name: str) -> str:
    """Turn a raw column/table identifier into a readable label.

    ``product_category`` → ``Product Category``. Kept in the backend so the
    frontend stays a pure display layer.
    """
    return " ".join(part for part in str(name).replace("_", " ").split()).title()


def build_applied_filters(sql: str | None) -> list[dict[str, Any]]:
    """Describe a query's WHERE clause for display in the UI.

    Users cannot tell from a chart alone which slice of data they are looking
    at — whether "sales in Nov 2016" actually filtered to November, or which
    product category is plotted. Surfacing the filters makes the answer
    self-explanatory and makes a wrong filter immediately visible.

    Reuses the same sqlglot-based extractor the HITL flow uses, so what the UI
    shows is exactly what the query did — not a re-parse that could disagree.
    Returns [] for any SQL that fails to parse; this is display metadata and
    must never break a response.
    """
    if not sql or not isinstance(sql, str):
        return []

    # The explanation route returns TWO statements joined by a blank line, and
    # sqlglot.parse_one handles exactly one. Parse each separately, then dedupe —
    # the baseline and breakdown queries share most of their filters.
    statements = [part.strip() for part in sql.split("\n\n") if part.strip()]

    filters = []
    try:
        from app.utils.sql_parser import extract_where_filters

        for statement in statements:
            filters.extend(extract_where_filters(statement))
    except Exception:
        logger.exception("Failed to extract filters for display")
        return []

    applied: list[dict[str, Any]] = []
    for column_filter in filters:
        value = column_filter.filter_value
        # IN clauses arrive as a list; render them as a comma-joined string so
        # the frontend needs no type-branching.
        display_value = (
            ", ".join(str(item) for item in value)
            if isinstance(value, (list, tuple))
            else str(value)
        )
        applied.append(
            {
                "table": column_filter.table,
                "column": column_filter.column,
                "label": _humanize_identifier(column_filter.column),
                "operator": column_filter.operator,
                "value": display_value,
            }
        )

    for statement in statements:
        applied.extend(_extract_display_only_filters(statement))

    # Dedupe on column+operator+value, preserving order. The two analysis
    # statements repeat the same filters, and showing each chip twice reads as
    # a bug.
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for item in applied:
        key = (item["column"], item["operator"], item["value"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _extract_display_only_filters(sql: str) -> list[dict[str, Any]]:
    """Find filters the HITL extractor intentionally skips.

    ``extract_where_filters`` only handles ``=``/``LIKE``/``IN`` on bare columns,
    because those are the only shapes HITL can safely rewrite. But date filters
    are what users most want confirmed — a "sales in Nov 2016" question hinges on
    ``EXTRACT(year FROM orderdate) = 2016``, which that extractor drops.

    This adds those shapes for DISPLAY ONLY. Nothing here feeds SQL rewriting, so
    being approximate is acceptable where being wrong would not be.
    """
    extra: list[dict[str, Any]] = []
    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001 - display metadata only; never break a response
        return extra

    where = parsed.find(exp.Where)
    if where is None:
        return extra

    for node in where.find_all(exp.EQ):
        # EXTRACT(year FROM col) = 2016  /  date_trunc('month', col) = '...'
        func = node.left if isinstance(node.left, exp.Func) else None
        literal = node.right if isinstance(node.right, exp.Literal) else None
        if func is None or literal is None:
            continue

        column = func.find(exp.Column)
        if column is None:
            continue

        # EXTRACT stores the unit in `this`; fall back to the function name.
        unit = ""
        if isinstance(func, exp.Extract):
            unit = str(func.this).strip("'\" ")
        part_label = unit.title() if unit else func.sql_name().title()

        extra.append(
            {
                "table": column.table or None,
                "column": column.name,
                "label": f"{_humanize_identifier(column.name)} {part_label}".strip(),
                "operator": "=",
                "value": str(literal.this),
            }
        )

    for node in where.find_all(exp.Between):
        column = (
            node.this
            if isinstance(node.this, exp.Column)
            else node.this.find(exp.Column)
        )
        if column is None:
            continue
        low = node.args.get("low")
        high = node.args.get("high")
        extra.append(
            {
                "table": column.table or None,
                "column": column.name,
                "label": _humanize_identifier(column.name),
                "operator": "between",
                "value": f"{getattr(low, 'this', low)} → {getattr(high, 'this', high)}",
            }
        )

    # Range comparisons: `date >= '2016-01-01' AND date < '2017-01-01'` is the
    # most natural way to express a year filter, and was being dropped entirely.
    range_ops = {exp.GTE: ">=", exp.GT: ">", exp.LTE: "<=", exp.LT: "<"}
    ranges: dict[tuple[str | None, str], list[tuple[str, str]]] = {}
    for node_type, symbol in range_ops.items():
        for node in where.find_all(node_type):
            column = node.left if isinstance(node.left, exp.Column) else None
            literal = node.right if isinstance(node.right, exp.Literal) else None
            if column is None or literal is None:
                continue
            ranges.setdefault((column.table or None, column.name), []).append(
                (symbol, str(literal.this))
            )

    for (table, column_name), bounds in ranges.items():
        lower = next((v for op, v in bounds if op in (">=", ">")), None)
        upper = next((v for op, v in bounds if op in ("<=", "<")), None)
        if lower and upper:
            # Both bounds on one column read best as a single range chip.
            value = f"{lower} → {upper}"
            operator = "range"
        else:
            operator, value = bounds[0]
        extra.append(
            {
                "table": table,
                "column": column_name,
                "label": _humanize_identifier(column_name),
                "operator": operator,
                "value": value,
            }
        )

    return extra


def get_connection_display_config(connection_id: str | None) -> dict[str, Any] | None:
    """Return the connection's display_config from settings.db_connections.

    This is where a workspace configures its unit scale — e.g. Indian units
    (Cr/Lakh/K) instead of B/M/K — via number_format_thresholds and
    number_format_suffixes.
    """
    if not connection_id:
        return None
    try:
        from app.settings_db.repositories import DBConnectionRepository

        connection = DBConnectionRepository().get_connection_by_id(connection_id)
        return connection.get("display_config") if connection else None
    except Exception:
        logger.exception(
            "Could not load display_config for connection %s", connection_id
        )
        return None


def build_number_format(
    data: Any, display_config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Pick a display scale for a table's numeric values.

    Uses the SAME thresholds and suffixes as the chart path
    (``detect_number_format`` in orch_graph.py), read from the connection's
    ``display_config``. Hardcoding B/M/K here meant a table showed "983.04M"
    while the chart of the same figures showed "98.30Cr" — the config existed
    and only the chart honoured it.

    One scale is chosen for the whole result, from the largest magnitude
    present, so a column of figures stays comparable at a glance.
    """
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return None

    # An explicit opt-out means show raw values.
    if display_config and not display_config.get("number_format_enabled", True):
        return None

    magnitudes = [
        abs(value)
        for row in rows
        if isinstance(row, dict)
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not magnitudes:
        return None

    config = display_config or {}
    thresholds = config.get("number_format_thresholds") or {}
    suffixes = config.get("number_format_suffixes") or {}
    decimals = config.get("decimal_places", 2)

    # Descending order matters: the first threshold the value clears wins.
    # Names are the config's keys, so a workspace using crore/lakh is honoured
    # as long as its thresholds and suffixes use the same key names.
    default_thresholds = {"billion": 1e9, "million": 1e6, "thousand": 1e3}
    default_suffixes = {"billion": "B", "million": "M", "thousand": "K"}

    # When the config defines its own thresholds, use ONLY those — mixing in the
    # B/M/K defaults would let "million" outrank "crore" and undo the override.
    effective = thresholds or default_thresholds
    scales = sorted(
        (
            (name, float(value))
            for name, value in effective.items()
            if isinstance(value, (int, float)) and value > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    largest = max(magnitudes)
    for name, threshold in scales:
        if largest >= threshold:
            return {
                "divisor": threshold,
                "suffix": suffixes.get(name, default_suffixes.get(name, "")),
                "decimals": decimals,
            }
    return None


def build_chat_response(
    result: Any, logs: str, connection_id: str | None = None
) -> dict[str, Any]:
    """Normalize orchestrator output for API responses.

    ``connection_id`` is used to load the connection's display_config so table
    number formatting matches the chart's (Cr/Lakh vs B/M/K).
    """
    payload: dict[str, Any] = {
        "logs": logs.strip() or None,
        "data": None,
        "sql": None,
        "chart_image_base64": None,
    }

    # Extract SQL if present
    if isinstance(result, dict):
        payload["sql"] = result.get("sql")
        # Describe the WHERE clause so the UI can show what slice of data
        # this answer covers.
        payload["applied_filters"] = build_applied_filters(payload["sql"])

        # The explanation route reports the filters it carried over from earlier
        # turns. Prepend them: an inherited filter is the one a user is most
        # likely to have forgotten, and most needs confirming.
        scope = result.get("analysis_scope")
        if isinstance(scope, dict):
            inherited = [
                {
                    "table": None,
                    "column": "",
                    "label": "Carried over",
                    "operator": "",
                    "value": str(item),
                }
                for item in (scope.get("inherited_filters") or [])
                if str(item).strip()
            ]
            payload["applied_filters"] = inherited + payload["applied_filters"]

    # Load the connection's display_config ONCE. Each branch below fills
    # payload["data"] at a different point, so the scale is applied per-branch
    # via _attach_number_format() rather than here.
    display_config = (
        get_connection_display_config(connection_id)
        if isinstance(result, dict)
        else None
    )

    def _attach_number_format(response: dict[str, Any]) -> dict[str, Any]:
        """Add the display scale for whatever data this branch produced.

        Applied to EVERY response shape — a chart answer still renders its rows
        as a table and its figures in prose, so it needs the scale too.
        """
        response["number_format"] = build_number_format(
            response.get("data"), display_config
        )
        return response

    # ### HITL — Return pending_hitl response as-is (graph is paused)
    if isinstance(result, dict) and result.get("type") == "pending_hitl":
        payload["answer_type"] = "pending_hitl"
        payload["message"] = result.get("message", "Human input needed.")
        payload["hitl_type"] = result.get("hitl_type")
        payload["thread_id"] = result.get("thread_id")
        payload["suggestions"] = result.get("suggestions")
        payload["data"] = {
            k: v
            for k, v in result.items()
            if k
            not in ("type", "message", "hitl_type", "thread_id", "suggestions", "sql")
        }
        return payload

    # --- Typed results from GraphOrchestrator ---
    if isinstance(result, dict) and "type" in result:
        result_type = result["type"]

        if result_type == "chart":
            chart_figure = result.get("figure")
            chart_config = result.get("chart_config")
            chart_data = result.get("data")
            payload["answer_type"] = "chart"
            payload["message"] = result.get("message") or "Here is your chart."

            # New format: chart_config + data (for frontend rendering)
            if chart_config:
                payload["chart_config"] = chart_config
                if isinstance(chart_data, pd.DataFrame):
                    payload["data"] = chart_data.to_dict(orient="records")
                elif isinstance(chart_data, list):
                    payload["data"] = chart_data
                else:
                    payload["data"] = []
            # Legacy format: matplotlib figure as base64 image
            elif chart_figure:
                payload["chart_image_base64"] = serialize_figure_to_base64(chart_figure)
                if isinstance(chart_data, pd.DataFrame):
                    payload["data"] = serialize_dataframe(chart_data)
                else:
                    payload["data"] = chart_data

            return _attach_number_format(payload)

        if result_type == "data":
            data = result.get("data")
            payload["answer_type"] = "table"
            payload["message"] = result.get("message") or "Here are the query results."
            if isinstance(data, pd.DataFrame):
                payload["data"] = serialize_dataframe(data)
            else:
                payload["data"] = data
            # Tables need the same scale hint charts get, or the exact same
            # figures render as "1.4B" in a chart and "1,392,495,535.246" in a
            # table. Frontend stays a display layer: the scale is decided here.
            return _attach_number_format(payload)

        if result_type == "schema_info":
            payload["answer_type"] = "schema_info"
            payload["message"] = result.get("message", "Schema information retrieved.")
            payload["data"] = result.get("data")
            return _attach_number_format(payload)

        if result_type == "text":
            payload["answer_type"] = "text"
            payload["message"] = result.get("message", "Here is the response.")
            return _attach_number_format(payload)

        if result_type == "error":
            payload["answer_type"] = "error"
            payload["message"] = result.get("message", "An error occurred.")
            return _attach_number_format(payload)

    # --- Legacy fallbacks ---
    if isinstance(result, pd.DataFrame):
        payload["answer_type"] = "table"
        payload["message"] = "Here are the query results."
        payload["data"] = serialize_dataframe(result)
        return _attach_number_format(payload)

    if isinstance(result, (dict, list)):
        payload["answer_type"] = "json"
        payload["message"] = "Structured response generated."
        payload["data"] = result
        return _attach_number_format(payload)

    payload["answer_type"] = "text"
    payload["message"] = str(result)
    return _attach_number_format(payload)


def build_persisted_assistant_content(response_payload: dict[str, Any]) -> str:
    """Serialize a stable assistant payload for conversation history storage."""
    # Every field the frontend needs to re-render must be here: on reload,
    # buildAssistantParts() runs against THIS payload, not the live response.
    # Omitting number_format meant a reloaded table fell back to auto-detected
    # B/M/K (704.8M) while the live one honoured the connection's Cr config
    # (70.48Cr) — same data, two different answers.
    persisted_payload = {
        "answer_type": response_payload.get("answer_type"),
        "message": response_payload.get("message"),
        "data": response_payload.get("data"),
        "chart_config": response_payload.get("chart_config"),  # New chart format
        "chart_image_base64": response_payload.get("chart_image_base64"),  # Legacy
        "sql": response_payload.get("sql"),
        "number_format": response_payload.get("number_format"),
        "applied_filters": response_payload.get("applied_filters"),
    }

    try:
        return json.dumps(
            {
                "kind": "assistant_payload_v1",
                "payload": persisted_payload,
            },
            ensure_ascii=True,
            default=str,
        )
    except (TypeError, ValueError):
        fallback_message = response_payload.get("message")
        if isinstance(fallback_message, str) and fallback_message.strip():
            return fallback_message
        return "Request completed."

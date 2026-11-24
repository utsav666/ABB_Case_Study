"""Decide how a SQL result is displayed (chart, table or single value) and
whether the result looks wrong enough to regenerate the query.

The router reads the user's *display intent* from their wording ("plot",
"list", "what is X in Q4"). Only the executed result reveals the data's
*shape* (one row? hundreds? the same label repeated?). Combining the two here
means an explicit request is honoured when it can be, and "auto" only charts
data that genuinely has a chart shape.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

import pandas as pd

DisplayIntent = Literal["chart", "table", "value", "auto"]
Display = Literal["chart", "table", "value"]

MAX_AUTO_CHART_ROWS = 60
MAX_AUTO_CHART_LABEL_COLUMNS = 2
MAX_DUPLICATE_EXAMPLES = 3


def to_frame(data: Any) -> pd.DataFrame:
    """Return ``data`` (records or DataFrame) as a DataFrame with real numbers."""
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data or [])
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        non_null = frame[column].dropna()
        if len(non_null) and all(
            isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)
            for v in non_null
        ):
            frame[column] = pd.to_numeric(frame[column])
    return frame


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def label_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(numeric_columns(frame))
    return [column for column in frame.columns if column not in numeric]


def find_result_issue(data: Any) -> str | None:
    """Return a correction hint when a result is probably wrong, else None.

    Two signals, both generic:
    * zero rows — usually a filter value that does not exist in the data;
    * several rows sharing every label column but with different numbers —
      rows from different versions, scenarios, parents or metrics were mixed,
      which is how one entity ends up counted several times.
    """
    frame = to_frame(data)
    if frame.empty:
        return (
            "The query ran but returned 0 rows. A filter value probably does not "
            "exist in the data or a join is too strict. Use only values listed in "
            "the table structure (copy them exactly, mind case-sensitive values), "
            "and check the scenario/version defaults."
        )

    labels = label_columns(frame)
    if not labels or not numeric_columns(frame) or len(frame) < 2:
        return None
    duplicated = frame[frame.duplicated(subset=labels, keep=False)]
    if duplicated.empty:
        return None
    examples = (
        duplicated[labels]
        .drop_duplicates()
        .head(MAX_DUPLICATE_EXAMPLES)
        .astype(str)
        .to_dict("records")
    )
    return (
        f"The result has several rows with identical {labels} but different "
        f"numbers (e.g. {examples}). Rows from different versions, scenarios, "
        "projects or metrics were mixed. Return one row per "
        f"{labels}: filter scenario columns to their default value and keep only "
        "the latest version per entity, unless the user asked to compare them. "
        "If the duplicates are genuinely different records the user asked for, "
        "add the column that distinguishes them."
    )


def _chart_blocker(frame: pd.DataFrame, strict: bool) -> str | None:
    """Return why the frame can't be charted well, or None when it can."""
    numeric = numeric_columns(frame)
    labels = label_columns(frame)

    if not numeric:
        return "the result has no numeric column to plot"
    if len(frame) < 2:
        return "there is only one row"

    x_column = labels[0] if labels else frame.columns[0]
    if frame[x_column].duplicated().any():
        return (
            f"several rows share the same '{x_column}', so one point per "
            f"'{x_column}' would mix different series"
        )

    if strict:
        if len(frame) > MAX_AUTO_CHART_ROWS:
            return f"{len(frame)} rows is detail data, not a chart"
        if len(labels) > MAX_AUTO_CHART_LABEL_COLUMNS:
            return "the result is row-level detail with many descriptive columns"
    return None


def decide_display(intent: str | None, data: Any) -> tuple[Display, str]:
    """Return (display, reason) for a query result given the user's intent."""
    intent = intent if intent in ("chart", "table", "value", "auto") else "auto"
    frame = to_frame(data)

    if frame.empty:
        return "table", "the query returned no rows"

    numeric = numeric_columns(frame)
    if len(frame) == 1 and intent != "table":
        if len(numeric) == 1 and len(frame.columns) <= 2:
            return "value", "single value answer"
        return "table", "single row answer"

    if intent == "table":
        return "table", "the user asked for a list or table"
    if intent == "value":
        return "table", "a single value was expected but several rows came back"

    blocker = _chart_blocker(frame, strict=intent != "chart")
    if intent == "chart":
        if blocker:
            return "table", f"a chart would be misleading: {blocker}"
        return "chart", "the user asked for a chart"
    if blocker:
        return "table", blocker
    return "chart", "the result has a clear chart shape"


def format_single_value(data: Any) -> str | None:
    """Render a 1-row result with one number as "**Label:** 1,234"."""
    frame = to_frame(data)
    numeric = numeric_columns(frame)
    if len(frame) != 1 or len(numeric) != 1:
        return None
    column = numeric[0]
    value = frame.iloc[0][column]
    if pd.isna(value):
        return None
    number = float(value)
    rendered = f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"
    label = str(column).replace("_", " ").strip().capitalize()
    context = [
        str(frame.iloc[0][c])
        for c in label_columns(frame)
        if pd.notna(frame.iloc[0][c])
    ]
    suffix = f" ({', '.join(context)})" if context else ""
    return f"**{label}:** {rendered}{suffix}"

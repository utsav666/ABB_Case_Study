"""Chart data shaping and display formatting.

Extracted from the ~520-line ``chart_approval_node`` in orch_graph.py, where
these lived as nested functions. Nested inside a graph node they were unreachable
by tests and unusable anywhere else, even though all three are pure transforms
over a DataFrame plus config.

  * ``detect_and_pivot_yoy_comparison`` — reshape a long year-over-year result
    into one column per period, which a multi-series chart needs
  * ``detect_date_granularity`` — pick a date axis format from the row spacing
  * ``detect_number_format`` — pick one display scale for the whole result,
    honouring the connection's display_config (e.g. Indian crore/lakh)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_and_pivot_yoy_comparison(df, question_text):
    """Detect YoY comparison and pivot data for multi-series chart."""

    import pandas as pd

    # Check if question indicates comparison
    comparison_keywords = [
        "compare",
        "vs",
        "versus",
        "compared to",
        "last year",
        "previous year",
        "year over year",
        "yoy",
    ]
    is_comparison = any(
        keyword in question_text.lower() for keyword in comparison_keywords
    )

    if not is_comparison:
        return df, False

    # Check if data is already pivoted (has multiple numeric columns)
    # SQL agent might have already done the pivot with CASE WHEN
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Data is already pivoted, just format it for multi-series chart
        logger.info(
            f"Data already pivoted by SQL: {len(numeric_cols)} metric columns detected"
        )
        # Find the categorical column (likely date-based)
        cat_cols = [col for col in df.columns if col not in numeric_cols]
        if cat_cols:
            return df, True  # Return as-is, already in the right format

    return df, False

    # Check if data has a date column with multiple time periods
    date_col = None
    comparison_type = None  # 'year' or 'month'

    for col in df.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(df[col])
            or "date" in col.lower()
            or "day" in col.lower()
            or "month" in col.lower()
        ):
            try:
                dates = pd.to_datetime(df[col], errors="coerce")
                if dates.notna().any():
                    years = dates.dt.year.dropna().unique()
                    months = dates.dt.to_period("M").dropna().unique()

                    # Check for year-over-year comparison
                    if len(years) >= 2:
                        date_col = col
                        comparison_type = "year"
                        break
                    # Check for month-over-month comparison (same year, different months)
                    elif len(months) >= 2 and len(years) == 1:
                        date_col = col
                        comparison_type = "month"
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
    df_copy["date_parsed"] = pd.to_datetime(df_copy[date_col])

    if comparison_type == "year":
        # Year-over-year: compare same days across different years
        df_copy["year"] = df_copy["date_parsed"].dt.year
        df_copy["month_day"] = df_copy["date_parsed"].dt.strftime("%m-%d")

        # Pivot: rows = month_day, columns = year, values = metric
        pivoted = df_copy.pivot_table(
            index="month_day", columns="year", values=metric_col, aggfunc="mean"
        ).reset_index()

        # Rename columns to indicate years
        pivoted.columns = [
            str(col) if col == "month_day" else f"{metric_col}_{col}"
            for col in pivoted.columns
        ]
        logger.info(
            f"Pivoted YoY comparison data: {len(df)} rows → {len(pivoted)} rows, {len(pivoted.columns)} columns"
        )

    elif comparison_type == "month":
        # Month-over-month: compare same days across different months
        df_copy["month_name"] = df_copy["date_parsed"].dt.strftime(
            "%B"
        )  # Full month name
        df_copy["day_of_month"] = df_copy["date_parsed"].dt.day

        # Pivot: rows = day_of_month, columns = month_name, values = metric
        pivoted = df_copy.pivot_table(
            index="day_of_month",
            columns="month_name",
            values=metric_col,
            aggfunc="mean",
        ).reset_index()

        # Rename columns to indicate months
        pivoted.columns = [
            str(col) if col == "day_of_month" else f"{metric_col}_{col.lower()}_2018"
            for col in pivoted.columns
        ]
        logger.info(
            f"Pivoted MoM comparison data: {len(df)} rows → {len(pivoted)} rows, {len(pivoted.columns)} columns"
        )

    return pivoted, True


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
                dates = pd.to_datetime(col, errors="coerce")
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
            return "day"
        elif avg_diff_days < 10:
            return "week"
        elif avg_diff_days < 60:
            return "month"
        else:
            return "year"
    except Exception:
        return None


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
                col_data = pd.to_numeric(df[col], errors="coerce").dropna()
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
            return {
                "suffix": suffixes.get("billion", "B"),
                "divisor": thresholds.get("billion", 1e9),
                "decimals": decimals,
            }
        elif max_val >= thresholds.get("million", 1e6):
            return {
                "suffix": suffixes.get("million", "M"),
                "divisor": thresholds.get("million", 1e6),
                "decimals": decimals,
            }
        elif max_val >= thresholds.get("thousand", 1e3):
            return {
                "suffix": suffixes.get("thousand", "K"),
                "divisor": thresholds.get("thousand", 1e3),
                "decimals": decimals,
            }
        else:
            return {"suffix": "", "divisor": 1, "decimals": 0}
    except Exception:
        return None

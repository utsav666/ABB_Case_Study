"""Tool: Answer "why is X high/low compared to others?" questions.

Why this exists
---------------
A question like "why was Nov 2016 sales high compared to other months?" cannot
be answered by one query. It needs three things a single SELECT does not give
you:

  1. a BASELINE — the focal period is meaningless without the other periods
  2. a CONTRIBUTION breakdown — which dimension members drove the difference
  3. a MECHANISM — did the total move because of volume, or unit value?

Design principle: **all arithmetic happens in pandas, never in the LLM.** The
LLM is used twice — once to write the two SQL queries (it knows the schema), and
once to turn a table of already-computed deltas into prose. It is never asked
to compute or estimate a number, so it cannot invent one. Every figure in the
narrative traces back to a row this module calculated.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.envsetup.env_loader import load_env
from app.prompt.prompts import analysis_plan_prompt, analysis_synthesis_prompt
from app.utils.llm_factory import create_chat_llm
from app.utils.sql_executor import SqlExec

logger = logging.getLogger(__name__)

load_env()

# Guard rails on how much we pull back. An explanation needs aggregates, not
# raw rows; anything larger than this suggests the LLM wrote the wrong query.
MAX_ANALYSIS_ROWS = 500


def _strip_sql_fences(text: str) -> str:
    """Remove ```sql fences the LLM sometimes wraps SQL in."""
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def _first_numeric_column(frame: pd.DataFrame) -> str | None:
    """Return the name of the first numeric column, if any."""
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            return column
    return None


def _first_non_numeric_column(frame: pd.DataFrame) -> str | None:
    """Return the name of the first non-numeric (label) column, if any."""
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            return column
    return None


def compute_baseline_stats(
    periods: pd.DataFrame, focal_label: str | None = None
) -> dict[str, Any]:
    """Compare a focal period against every other period in the frame.

    Pure arithmetic — this is what keeps the narrative honest.

    Parameters
    ----------
    periods:
        One row per period, with a label column and a metric column.
    focal_label:
        Which period is being explained. When None, the largest is assumed
        (a "why was X high" question is usually about the peak).
    """
    label_column = _first_non_numeric_column(periods)
    metric_column = _first_numeric_column(periods)
    if not label_column or not metric_column or periods.empty:
        return {}

    frame = periods[[label_column, metric_column]].dropna()
    if frame.empty:
        return {}

    # Match the focal period by substring so "Nov 2016" finds "2016-11-01".
    focal_row = None
    if focal_label:
        matches = frame[
            frame[label_column]
            .astype(str)
            .str.contains(str(focal_label), case=False, regex=False)
        ]
        if not matches.empty:
            focal_row = matches.iloc[0]

    if focal_row is None:
        focal_row = frame.loc[frame[metric_column].idxmax()]

    focal_value = float(focal_row[metric_column])
    others = frame[frame[label_column] != focal_row[label_column]]

    stats: dict[str, Any] = {
        "focal_period": str(focal_row[label_column]),
        "focal_value": focal_value,
        "metric_name": metric_column,
        "period_count": int(len(frame)),
    }

    if not others.empty:
        others_mean = float(others[metric_column].mean())
        stats.update(
            {
                "baseline_mean": others_mean,
                "difference": focal_value - others_mean,
                "pct_vs_baseline": (
                    ((focal_value - others_mean) / others_mean * 100)
                    if others_mean
                    else None
                ),
                "max_other_period": str(
                    others.loc[others[metric_column].idxmax()][label_column]
                ),
                "max_other_value": float(others[metric_column].max()),
                "min_other_period": str(
                    others.loc[others[metric_column].idxmin()][label_column]
                ),
                "min_other_value": float(others[metric_column].min()),
                # Rank 1 = highest. Tells us whether "high" is even true.
                "rank": int((frame[metric_column] > focal_value).sum() + 1),
            }
        )

        # Standard deviations above the mean — distinguishes "a good month" from
        # "a genuine outlier worth explaining".
        others_std = float(others[metric_column].std())
        if others_std:
            stats["std_devs_from_mean"] = (focal_value - others_mean) / others_std

    return stats


def compute_contributions(
    breakdown: pd.DataFrame, focal_total: float | None = None, top_n: int = 5
) -> list[dict[str, Any]]:
    """Rank dimension members by how much each contributed to the focal total.

    Answers "Category D drove 62% of it" rather than merely "Category D is
    biggest" — share of total is the number that explains a gap.
    """
    label_column = _first_non_numeric_column(breakdown)
    metric_column = _first_numeric_column(breakdown)
    if not label_column or not metric_column or breakdown.empty:
        return []

    frame = (
        breakdown[[label_column, metric_column]]
        .dropna()
        .groupby(label_column, as_index=False)[metric_column]
        .sum()
        .sort_values(metric_column, ascending=False)
    )

    total = float(focal_total) if focal_total else float(frame[metric_column].sum())
    if not total:
        return []

    return [
        {
            "member": str(row[label_column]),
            "value": float(row[metric_column]),
            "share_pct": float(row[metric_column]) / total * 100,
        }
        for _, row in frame.head(top_n).iterrows()
    ]


def _run_query(sql: str, label: str) -> pd.DataFrame | None:
    """Execute one analysis query, returning None on failure.

    A failed sub-query degrades the analysis rather than killing it: a baseline
    with no breakdown still answers "was it actually high?".
    """
    if not sql:
        return None
    result = SqlExec().sql_executor(sql)
    if isinstance(result, pd.DataFrame):
        logger.info("ANALYSIS %s query — %d row(s)", label, len(result))
        return result.head(MAX_ANALYSIS_ROWS)
    logger.warning("ANALYSIS %s query failed: %s", label, result)
    return None


def analyze_why(
    user_question: str,
    schema_summary: str,
    provider: str | None = None,
    model: str | None = None,
    conversation_history: str = "",
) -> dict[str, Any]:
    """Answer a "why is X high/low" question with grounded analysis.

    ``conversation_history`` is REQUIRED for correctness, not a nicety: "why was
    Nov high?" is almost always a follow-up. If the previous turn was "trend of
    sales for Product A in 2016", the analysis must inherit the Product A filter
    or it silently explains the wrong number — a whole-database total presented
    as the answer to a Product A question.

    Returns
    -------
    {"success": True, "answer": str, "evidence": {...}, "sql": [...],
     "responses": [AIMessage, ...]}   # responses are for token tracking
    or {"success": False, "error": str}
    """
    responses: list[Any] = []
    try:
        # --- Step 1: plan. The LLM writes two queries; it knows the schema. ---
        planner = create_chat_llm(
            purpose="analysis_plan", temperature=0, provider=provider, model=model
        )
        plan_response = planner.invoke(
            [
                {"role": "system", "content": "You write PostgreSQL analysis queries."},
                {
                    "role": "user",
                    "content": analysis_plan_prompt.format(
                        schema_summary=schema_summary,
                        user_question=user_question,
                        conversation_history=conversation_history or "(no prior turns)",
                    ),
                },
            ]
        )
        responses.append(plan_response)

        import json

        plan_text = _strip_sql_fences(str(plan_response.content))
        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            logger.warning("ANALYSIS — plan was not valid JSON:\n%s", plan_text[:400])
            return {
                "success": False,
                "error": "Could not plan the analysis for this question.",
            }

        focal_label = plan.get("focal_period") or None
        inherited = plan.get("inherited_filters") or []
        # Log the inherited filters explicitly: a follow-up that silently drops
        # the user's earlier scope produces a plausible but wrong answer, and
        # this is the only place that becomes visible.
        logger.info(
            "ANALYSIS — plan: focal=%s metric=%s inherited_filters=%s",
            focal_label,
            plan.get("metric_description"),
            inherited or "NONE",
        )

        # --- Step 2: gather evidence. ---
        periods = _run_query(plan.get("baseline_sql", ""), "baseline")
        breakdown = _run_query(plan.get("breakdown_sql", ""), "breakdown")

        if periods is None and breakdown is None:
            return {
                "success": False,
                "error": "The analysis queries could not be executed against this database.",
            }

        # --- Step 3: compute. All arithmetic here, none in the LLM. ---
        baseline_stats = (
            compute_baseline_stats(periods, focal_label) if periods is not None else {}
        )
        contributions = (
            compute_contributions(breakdown, baseline_stats.get("focal_value"))
            if breakdown is not None
            else []
        )

        if not baseline_stats and not contributions:
            return {
                "success": False,
                "error": "The analysis returned no comparable figures.",
            }

        evidence = {
            "baseline": baseline_stats,
            "contributions": contributions,
            "period_series": (
                periods.to_dict(orient="records") if periods is not None else []
            ),
        }
        logger.info(
            "ANALYSIS — evidence: focal=%s value=%s vs baseline_mean=%s (%s%%), %d contributor(s)",
            baseline_stats.get("focal_period"),
            baseline_stats.get("focal_value"),
            baseline_stats.get("baseline_mean"),
            round(baseline_stats["pct_vs_baseline"], 1)
            if baseline_stats.get("pct_vs_baseline") is not None
            else "n/a",
            len(contributions),
        )

        # --- Step 4: narrate the computed numbers. ---
        writer = create_chat_llm(
            purpose="analysis_synthesis", temperature=0, provider=provider, model=model
        )
        synthesis_response = writer.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst. You explain figures that have "
                        "already been calculated. Never invent or recompute a number."
                    ),
                },
                {
                    "role": "user",
                    "content": analysis_synthesis_prompt.format(
                        user_question=user_question,
                        evidence=json.dumps(evidence, indent=2, default=str),
                    ),
                },
            ]
        )
        responses.append(synthesis_response)

        return {
            "success": True,
            "answer": str(synthesis_response.content).strip(),
            "evidence": evidence,
            "metric_description": plan.get("metric_description") or "",
            "inherited_filters": inherited,
            "sql": [
                s for s in (plan.get("baseline_sql"), plan.get("breakdown_sql")) if s
            ],
            "responses": responses,
        }

    except Exception as error:  # noqa: BLE001 - surface a message, never a 500
        logger.exception("ANALYSIS failed")
        return {"success": False, "error": str(error), "responses": responses}

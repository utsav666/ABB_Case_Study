"""Tool: Translate a natural-language question into a GA4 runReport request.

Parallels app/agent/sql_generation_agent.py but targets the Google Analytics
Data API. Given the question plus the property's available dimensions/metrics
(the GA context summary), the LLM returns a structured runReport spec:
dimensions, metrics, date range, optional dimension filter, and limit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

from app.envsetup.env_loader import get_openai_api_key, load_env

logger = logging.getLogger(__name__)

load_env()

_GA_SYSTEM_PROMPT = """\
You are a Google Analytics 4 (GA4) reporting expert. Given a user question and
the property's available dimensions and metrics, produce a single GA4 Data API
runReport request as JSON.

Return ONLY a JSON object (no prose, no markdown) with these keys:
  "dimensions"  - array of GA4 dimension API names (e.g. ["country", "date"]).
                  Use ONLY dimensions listed in the provided context. May be [].
  "metrics"     - array of GA4 metric API names (e.g. ["sessions", "activeUsers"]).
                  Use ONLY metrics listed in the provided context. At least one.
  "start_date"  - report start, "YYYY-MM-DD" or a GA relative date like
                  "7daysAgo", "30daysAgo", "yesterday", "today".
  "end_date"    - report end, same formats (default "today").
  "limit"       - integer max rows (default 100, cap 250).
  "order_by"    - optional metric API name to sort by, descending.

Rules:
- Use ONLY the exact API names provided in the context. Never invent names.
- If a previous attempt failed, return a corrected request, never an explanation.
"""

_GA_USER_TEMPLATE = """\
Available GA4 schema:
{ga_context}

------------------
User question:
{question}
------------------
Now return the runReport JSON:
"""


def generate_ga_report_spec(
    user_question: str,
    ga_context: str,
    error_context: str = "",
) -> dict[str, Any]:
    """Generate a GA4 runReport spec for a question.

    Returns {"success": True, "spec": {...}} or {"success": False, "error": ...}.
    """
    try:
        openai.api_key = get_openai_api_key()
        prompt = _GA_USER_TEMPLATE.format(
            ga_context=ga_context,
            question=user_question,
        )
        if error_context:
            prompt += (
                f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                f"Error: {error_context}\n"
                f"Return a corrected runReport JSON.\n"
            )

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _GA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        spec = json.loads(raw)

        # Light normalization / guardrails.
        if not spec.get("metrics"):
            return {"success": False, "error": "No metrics were selected."}
        spec.setdefault("dimensions", [])
        spec.setdefault("start_date", "30daysAgo")
        spec.setdefault("end_date", "today")
        limit = int(spec.get("limit", 100) or 100)
        spec["limit"] = max(1, min(limit, 250))

        logger.info("Generated GA report spec: %s", spec)
        return {"success": True, "spec": spec}
    except Exception as error:  # noqa: BLE001
        logger.exception("GA report spec generation failed")
        return {"success": False, "error": str(error)}

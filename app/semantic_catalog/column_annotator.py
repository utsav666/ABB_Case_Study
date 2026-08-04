"""Lightweight LLM annotator that fills logical_name + description for columns.

Unlike :mod:`semantic_annotator` (which expects statistical data profiles),
this annotator works from just the column name, data type, and table context.
It produces, per column:

  - ``logical_name``: a short human-friendly name (e.g. "Customer Email")
  - ``description``:  a one-sentence plain-English purpose

It makes a single batched LLM call per table. On any failure it returns empty
annotations so callers can safely leave the fields NULL.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.envsetup.env_loader import get_openai_api_key

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a data catalog expert. Given a database table name and its columns
(each with a raw name and SQL data type), produce a concise semantic
annotation for every column.

For each column return a JSON object with exactly these keys:
  "logical_name" — a short, human-friendly display name in Title Case
                   (e.g. "Customer Email", "Order Total"). 1-4 words.
  "description"  — one short plain-English sentence describing the column's
                   purpose. No more than 20 words.

Return ONLY a JSON object keyed by the column's raw name. No markdown,
no commentary. Every column listed must appear exactly once.
"""

_USER_TEMPLATE = """\
Table: {table_name}

Columns:
{columns_text}

Generate the annotation JSON for every column above.
"""

_DB_SYSTEM_PROMPT = """\
You are a data catalog expert. You will be given an entire database schema:
multiple tables, each with its columns (raw name and SQL data type). Seeing
the whole schema at once, produce a concise semantic annotation for every
column, keeping logical names consistent across related tables.

For each column return a JSON object with exactly these keys:
  "logical_name" — a short, human-friendly display name in Title Case
                   (e.g. "Customer Email", "Order Total"). 1-4 words.
  "description"  — one short plain-English sentence describing the column's
                   purpose. No more than 20 words.

Return ONLY a JSON object shaped as:
  {{ "schema.table_name": {{ "raw_column_name": {{ "logical_name": ..., "description": ... }} }} }}
Use the exact table keys and raw column names provided. No markdown,
no commentary. Every table and column listed must appear exactly once.
"""

_DB_USER_TEMPLATE = """\
Database schema ({table_count} tables):

{schema_text}

Generate the annotation JSON for every column in every table above.
"""


class ColumnAnnotator:
    """Generates logical_name + description for columns via a single LLM call."""

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        self._llm = ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o").strip().strip("'\""),
            temperature=temperature,
            api_key=SecretStr(get_openai_api_key()),
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("user", _USER_TEMPLATE),
            ]
        )
        self._db_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _DB_SYSTEM_PROMPT),
                ("user", _DB_USER_TEMPLATE),
            ]
        )

    def annotate_columns(
        self,
        table_name: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]]:
        """Return ``{raw_name: {"logical_name": ..., "description": ...}}``.

        Args:
            table_name: Qualified or bare table name for context.
            columns: List of dicts with at least ``raw_name`` and ``data_type``.

        Returns:
            Mapping keyed by raw column name. Empty dict on failure.
        """
        if not columns:
            return {}

        raw_names = [c["raw_name"] for c in columns if c.get("raw_name")]
        columns_text = "\n".join(
            f"  {c['raw_name']}: {c.get('data_type') or 'unknown'}"
            for c in columns
            if c.get("raw_name")
        )

        messages = self._prompt.format_messages(
            table_name=table_name,
            columns_text=columns_text,
        )

        try:
            logger.info("Requesting column annotation for table %s", table_name)
            response = self._llm.invoke(messages)
            raw_content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return _parse_response(raw_content, raw_names)
        except Exception:  # noqa: BLE001 - never let annotation crash the sync
            logger.exception("Column annotation failed for table %s", table_name)
            return {}

    def annotate_database(
        self,
        tables: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, str]]]:
        """Annotate every column across the whole database in ONE LLM call.

        Args:
            tables: List of dicts, each with:
                - "table_key": stable key the result is grouped by (e.g.
                  "public.movie")
                - "columns": list of {"raw_name", "data_type"}

        Returns:
            ``{table_key: {raw_name: {"logical_name", "description"}}}``.
            Empty dict on failure (callers leave fields NULL).
        """
        tables_with_columns = [
            t for t in tables if t.get("columns") and t.get("table_key")
        ]
        if not tables_with_columns:
            return {}

        schema_text = _format_database_schema(tables_with_columns)
        expected = {
            t["table_key"]: [c["raw_name"] for c in t["columns"] if c.get("raw_name")]
            for t in tables_with_columns
        }

        messages = self._db_prompt.format_messages(
            table_count=len(tables_with_columns),
            schema_text=schema_text,
        )

        try:
            logger.info(
                "Requesting DB-level annotation for %d tables",
                len(tables_with_columns),
            )
            response = self._llm.invoke(messages)
            raw_content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return _parse_db_response(raw_content, expected)
        except Exception:  # noqa: BLE001 - never let annotation crash the sync
            logger.exception("DB-level column annotation failed")
            return {}


def _parse_response(raw: str, expected_columns: list[str]) -> dict[str, dict[str, str]]:
    """Parse the LLM JSON response, tolerating markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rfind("```")]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse column annotation JSON:\n%s", cleaned[:500])
        return {}

    if not isinstance(parsed, dict):
        return {}

    # Normalize: keep only expected columns, coerce to strings.
    result: dict[str, dict[str, str]] = {}
    for col in expected_columns:
        entry = parsed.get(col)
        if isinstance(entry, dict):
            logical = str(entry.get("logical_name", "") or "").strip()
            description = str(entry.get("description", "") or "").strip()
            result[col] = {
                "logical_name": logical,
                "description": description,
            }
    return result


def _strip_code_fences(raw: str) -> str:
    """Remove surrounding markdown code fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def _format_database_schema(tables: list[dict[str, Any]]) -> str:
    """Render the whole schema compactly: one block per table."""
    blocks: list[str] = []
    for table in tables:
        lines = [f"Table: {table['table_key']}"]
        for column in table["columns"]:
            if not column.get("raw_name"):
                continue
            lines.append(
                f"  {column['raw_name']}: {column.get('data_type') or 'unknown'}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_db_response(
    raw: str,
    expected: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Parse the DB-level JSON response keyed by table then column."""
    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse DB annotation JSON:\n%s", cleaned[:500])
        return {}

    if not isinstance(parsed, dict):
        return {}

    result: dict[str, dict[str, dict[str, str]]] = {}
    for table_key, raw_names in expected.items():
        table_entry = parsed.get(table_key)
        if not isinstance(table_entry, dict):
            continue
        table_result: dict[str, dict[str, str]] = {}
        for raw_name in raw_names:
            entry = table_entry.get(raw_name)
            if isinstance(entry, dict):
                logical = str(entry.get("logical_name", "") or "").strip()
                description = str(entry.get("description", "") or "").strip()
                table_result[raw_name] = {
                    "logical_name": logical,
                    "description": description,
                }
        if table_result:
            result[table_key] = table_result
    return result

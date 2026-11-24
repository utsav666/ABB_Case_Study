"""Run golden questions through the full orchestrator and check the answers.

Usage (from the project root):

    python -m scripts.eval_questions eval/cap_table_questions.json \
        --connection-id <uuid> [--only Q2 Q5]

Each question runs the real graph — router, SQL generation, value checks,
execution, result checks, display — against the connection's database, with
SQL executed read-only. Checks are about the *answer* (rows and values), not the
SQL text, so a correct query written differently still passes.

Question file format::

    {"questions": [
      {"id": "Q1", "question": "...",
       "follow_up_of": "Q0",              # optional: run with Q0 as history
       "expect": {
         "display": "table",              # chart | table | value
         "min_rows": 1, "max_rows": 10,
         "must_contain": ["INSM"],        # any cell or column name, case-insensitive
         "must_not_contain": ["ATYR"],
         "no_result_issue": true,         # default true: no empty / mixed-grain result
         "reference_sql": "SELECT ...",   # hand-written answer; see below
         "same_row_count": true           # with reference_sql (default true)
       }}

With ``reference_sql`` the hand-written query is run read-only and every value
in its result (numbers rounded to 2 decimals) must appear in the answer, so the
answer may have extra columns or different aliases but not different data.
    ]}

Exit code is the number of failed questions, so it can gate CI or a deploy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    data = result.get("data")
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    return data if isinstance(data, list) else []


def _haystack(rows: list[dict[str, Any]]) -> str:
    cells = [str(key) for row in rows[:1] for key in row]
    cells += [str(value) for row in rows for value in row.values()]
    return "\n".join(cells).lower()


def _normalize(value: Any) -> str:
    """Compare numbers by value (2 dp) and text case/space-insensitively."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()
    if isinstance(value, bool):
        return str(value).lower()
    return f"{number:.2f}"


def _run_reference(sql: str, connection_id: str) -> list[dict[str, Any]]:
    from app.utils.sql_executor import SqlExec
    from app.utils.sql_validator import validator

    if validator(sql) is not True:
        message = f"reference_sql must be a single read-only query: {sql}"
        raise ValueError(message)
    result = SqlExec(connection_id=connection_id).sql_executor(sql)
    if not isinstance(result, pd.DataFrame):
        message = f"reference_sql failed: {result}"
        raise RuntimeError(message)  # noqa: TRY004 - a failed query, not a bad type
    return result.to_dict("records")


def check_answer(result: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """Return a list of failed expectations (empty when the answer passes)."""
    from app.orchestrator.display_policy import find_result_issue

    failures: list[str] = []
    answer_type = result.get("type")
    if answer_type in ("error", "pending_hitl"):
        return [f"answer type is {answer_type}: {result.get('message')}"]

    rows = _rows(result)
    display = "chart" if answer_type == "chart" else result.get("_display", "table")
    if expect.get("display") and expect["display"] != display:
        failures.append(f"display {display!r} != expected {expect['display']!r}")
    if "min_rows" in expect and len(rows) < expect["min_rows"]:
        failures.append(f"{len(rows)} rows < min_rows {expect['min_rows']}")
    if "max_rows" in expect and len(rows) > expect["max_rows"]:
        failures.append(f"{len(rows)} rows > max_rows {expect['max_rows']}")

    haystack = _haystack(rows)
    failures += [
        f"missing {needle!r}"
        for needle in expect.get("must_contain", [])
        if needle.lower() not in haystack
    ]
    failures += [
        f"unexpected {needle!r}"
        for needle in expect.get("must_not_contain", [])
        if needle.lower() in haystack
    ]
    reference = expect.get("_reference_rows")
    if reference is not None:
        if expect.get("same_row_count", True) and len(reference) != len(rows):
            failures.append(f"{len(rows)} rows != reference {len(reference)}")
        answer_values = {_normalize(v) for row in rows for v in row.values()}
        missing = sorted(
            {_normalize(v) for row in reference for v in row.values()}
            - answer_values
            - {"none"}
        )
        if missing:
            failures.append(
                f"{len(missing)} reference value(s) missing, e.g. {missing[:5]}"
            )
    if expect.get("no_result_issue", True):
        issue = find_result_issue(rows)
        if issue:
            failures.append(f"result issue: {issue[:160]}")
    return failures


def run_question(
    question: str, connection_id: str, history: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one question; return (final response, final graph state)."""
    from app.orchestrator.orch_graph import GraphOrchestrator, _compiled_graph
    from app.utils.logging_context import bind_log_context

    thread_id = f"eval-{uuid.uuid4().hex[:8]}"
    bind_log_context(connection_id=connection_id, thread_id=thread_id)
    result = GraphOrchestrator(
        question,
        thread_id=thread_id,
        connection_id=connection_id,
        conversation_history=history,
    ).executor()
    state = _compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    return result, dict(state.values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions_file", type=Path)
    parser.add_argument("--connection-id", required=True)
    parser.add_argument("--only", nargs="*", help="Question ids to run")
    parser.add_argument("--verbose", action="store_true", help="Show app logs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
    from app.envsetup.env_loader import load_env

    load_env()

    spec = json.loads(args.questions_file.read_text())
    answers: dict[str, tuple[str, dict[str, Any]]] = {}
    failed = 0
    for item in spec["questions"]:
        if args.only and item["id"] not in args.only:
            continue
        history = ""
        parent = item.get("follow_up_of")
        if parent in answers:
            parent_question, parent_result = answers[parent]
            history = (
                f"User: {parent_question}\n\nAssistant: "
                f"{parent_result.get('message') or 'Returned query results.'}\n"
                f"SQL: {parent_result.get('sql')}"
            )

        started = time.perf_counter()
        try:
            result, state = run_question(item["question"], args.connection_id, history)
        except Exception as error:  # noqa: BLE001 - report and continue the run
            result, state = {"type": "error", "message": str(error)}, {}
        elapsed = time.perf_counter() - started
        result["_display"] = state.get("display") or "table"
        answers[item["id"]] = (item["question"], result)

        expect = dict(item.get("expect", {}))
        if expect.get("reference_sql"):
            expect["_reference_rows"] = _run_reference(
                expect["reference_sql"], args.connection_id
            )
        failures = check_answer(result, expect)
        failed += bool(failures)
        rows = _rows(result)
        print(f"\n{'PASS' if not failures else 'FAIL'}  {item['id']}  ({elapsed:.0f}s)")
        print(f"  Q: {item['question']}")
        print(
            f"  route={state.get('route')} display={result['_display']} "
            f"retries={state.get('retry_count', 0)} rows={len(rows)}"
        )
        if result.get("message"):
            print(f"  message: {str(result['message'])[:300]}")
        print("  SQL: " + str(result.get("sql") or "").replace("\n", "\n       "))
        if rows:
            print(
                "  "
                + pd.DataFrame(rows)
                .head(8)
                .to_string(max_colwidth=28)
                .replace("\n", "\n  ")
            )
        for failure in failures:
            print(f"  ✗ {failure}")

    print(f"\n{failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(main())

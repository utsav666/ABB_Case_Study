from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# Statement types that can change data, schema, or session state. Resolved with
# getattr so the guard keeps working across sqlglot versions that rename them.
_WRITE_EXPRESSION_TYPES: tuple[type[exp.Expression], ...] = tuple(
    expression_type
    for expression_type in (
        getattr(exp, name, None)
        for name in (
            "Insert",
            "Update",
            "Delete",
            "Merge",
            "Drop",
            "Create",
            "Alter",
            "AlterTable",
            "TruncateTable",
            "Command",
            "Into",
            "Copy",
            "Grant",
            "Revoke",
            "Set",
            "Transaction",
            "Commit",
            "Rollback",
            "Use",
        )
    )
    if isinstance(expression_type, type)
)


def validator(sql_query: str, dialect: str | None = None) -> bool | str:
    """Return True for a single read-only query, otherwise an error message.

    Generated SQL is executed against customer databases, so syntax is not
    enough: exactly one statement is allowed, its root must be a query
    (SELECT / UNION / ...), and no write or DDL node may appear anywhere in the
    tree — data-modifying CTEs such as ``WITH x AS (DELETE ...) SELECT`` and
    ``SELECT ... INTO new_table`` included.
    """
    try:
        statements = [
            statement
            for statement in sqlglot.parse(sql_query, read=dialect)
            if statement is not None
        ]
    except SqlglotError as error:
        return str(error)

    if not statements:
        return "No SQL statement found."
    if len(statements) > 1:
        return "Only a single SQL statement is allowed."

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        return "Only read-only SELECT queries are allowed."

    forbidden = next(statement.find_all(*_WRITE_EXPRESSION_TYPES), None)
    if forbidden is not None:
        return (
            "Only read-only SELECT queries are allowed "
            f"(found {forbidden.key.upper()})."
        )
    return True

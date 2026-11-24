"""Per-connection semantic profile — the "what the data contains" layer.

The schema graph describes structure (tables, columns, joins). That is not
enough to query a real database: the SQL generator must also know which values
exist ('United States' vs 'US'), how many there are, and how tables are shaped (metrics
stored as rows, scenario and version columns that must never be summed
together). Without that it guesses, and guesses produce empty or silently wrong
answers.

The profile has two parts, both stored inside the connection's schema graph
under ``graph["profile"]`` so every request reads them with the graph it
already loads:

* **facts** — collected with two batched queries per table: row count, distinct
  counts, min/max, the real values of categorical columns, and values that
  differ only by letter case together with the column that tells them apart.
  Cheap and deterministic, so it is built lazily on the chat path when missing.
* **inferred** — one LLM call per connection that reads the facts and states
  each table's grain, metric-as-row columns, scenario columns (with their base
  value) and version columns. Every column and value it names is validated
  against the facts; anything unverifiable is dropped. Built when the graph is
  (re)built or on explicit refresh — never on the chat path, because it is slow.

Nothing here is specific to one database: all knowledge comes from the data.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.semantic_catalog.profiling_queries import classify_column_type

logger = logging.getLogger(__name__)

PROFILE_VERSION = 1

#: Columns with at most this many distinct values have every value stored.
VALUE_STORE_LIMIT = int(os.getenv("PROFILER_VALUE_STORE_LIMIT", "500"))
#: Columns with at most this many distinct values are listed in full in prompts.
PROMPT_FULL_VALUE_LIMIT = int(os.getenv("PROFILER_PROMPT_VALUE_LIMIT", "150"))
#: Larger columns show only their most frequent values in prompts.
PROMPT_TOP_VALUES = 25
#: Columns of long strings (formulas, tags) are rarely filtered on: show examples.
PROMPT_LONG_VALUE_LENGTH = 80
PROMPT_LONG_VALUE_EXAMPLES = 5
#: Text longer than this is free text, not a category.
MAX_CATEGORICAL_LENGTH = 200
#: Below this many distinct values a column is categorical even if every row differs.
ALWAYS_CATEGORICAL_DISTINCT = 30
#: Above this distinct/row ratio a text column is an identifier or free text.
MAX_CATEGORICAL_RATIO = 0.5
#: A column with at most this many values is checked as a case-variant separator.
SEPARATOR_MAX_DISTINCT = 12
PROFILE_QUERY_TIMEOUT_MS = int(os.getenv("PROFILER_QUERY_TIMEOUT_MS", "60000"))
MAX_PARALLEL_TABLES = 4

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()
_build_locks: dict[str, threading.Lock] = {}


# ─── SQL helpers ──────────────────────────────────────────────────────────────


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _qualified(node: dict[str, Any]) -> str:
    return f"{_quote(node.get('schema') or 'public')}.{_quote(node['table'])}"


def _family(data_type: str | None) -> str:
    normalized = (data_type or "").strip().lower()
    if normalized.startswith("bool"):
        return "boolean"
    if normalized == "uuid":
        return "identifier"
    return classify_column_type(normalized)


def _stats_sql(node: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """One query returning row count plus per-column distinct/null/min/max."""
    columns = []
    selects = ["COUNT(*) AS row_count"]
    for index, column in enumerate(node.get("columns", [])):
        family = _family(column.get("data_type"))
        if family == "other":
            continue
        quoted = _quote(column["raw_name"])
        columns.append({"name": column["raw_name"], "family": family, "i": index})
        selects.append(f"COUNT(DISTINCT {quoted}::text) AS d{index}")
        selects.append(f"COUNT(*) - COUNT({quoted}) AS n{index}")
        if family in ("numeric", "temporal"):
            selects.append(f"MIN({quoted})::text AS lo{index}")
            selects.append(f"MAX({quoted})::text AS hi{index}")
        elif family == "text":
            selects.append(f"MAX(LENGTH({quoted}::text)) AS len{index}")
    return f"SELECT {', '.join(selects)} FROM {_qualified(node)}", columns


def _values_sql(node: dict[str, Any], column_names: list[str]) -> str:
    """One query returning (column, value, frequency) for categorical columns."""
    parts = [
        f"SELECT {position} AS c, {_quote(name)}::text AS v, COUNT(*) AS n "
        f"FROM {_qualified(node)} WHERE {_quote(name)} IS NOT NULL GROUP BY 2"
        for position, name in enumerate(column_names)
    ]
    return " UNION ALL ".join(parts) + " ORDER BY 1, 3 DESC, 2"


def _separator_sql(
    node: dict[str, Any], column: str, variants: list[str], separators: list[str]
) -> str:
    """Co-occurrence of case-variant values with low-cardinality columns."""
    literals = ", ".join("'" + v.replace("'", "''") + "'" for v in variants)
    parts = [
        f"SELECT {position} AS s, {_quote(column)}::text AS v, "
        f"{_quote(separator)}::text AS o, COUNT(*) AS n FROM {_qualified(node)} "
        f"WHERE {_quote(column)}::text IN ({literals}) GROUP BY 2, 3"
        for position, separator in enumerate(separators)
    ]
    return " UNION ALL ".join(parts)


# ─── Fact collection ──────────────────────────────────────────────────────────


def _is_categorical(column: dict[str, Any], row_count: int) -> bool:
    distinct = column.get("distinct_count") or 0
    if column["family"] == "boolean":
        return distinct > 0
    if column["family"] != "text" or distinct == 0 or distinct > VALUE_STORE_LIMIT:
        return False
    if (column.get("max_length") or 0) > MAX_CATEGORICAL_LENGTH:
        return False
    if distinct <= ALWAYS_CATEGORICAL_DISTINCT:
        return True
    return row_count > 0 and distinct / row_count <= MAX_CATEGORICAL_RATIO


def _case_variant_groups(values: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for value in values:
        groups.setdefault(value.casefold(), []).append(value)
    return [group for group in groups.values() if len(group) > 1]


def _profile_table(node: dict[str, Any], execute) -> dict[str, Any]:
    """Collect facts for one table. ``execute(sql) -> DataFrame | str``."""
    stats_sql, stat_columns = _stats_sql(node)
    stats = execute(stats_sql)
    if isinstance(stats, str) or stats is None or stats.empty:
        raise RuntimeError(f"stats query failed for {node['qualified_name']}: {stats}")
    row = stats.iloc[0]
    row_count = int(row["row_count"])

    columns: dict[str, dict[str, Any]] = {}
    for column in stat_columns:
        i = column["i"]
        info: dict[str, Any] = {
            "family": column["family"],
            "distinct_count": int(row[f"d{i}"]),
            "null_count": int(row[f"n{i}"]),
        }
        if f"lo{i}" in row and row[f"lo{i}"] is not None:
            info["min"] = str(row[f"lo{i}"])
            info["max"] = str(row[f"hi{i}"])
        if f"len{i}" in row and pd.notna(row[f"len{i}"]):
            info["max_length"] = int(row[f"len{i}"])
        columns[column["name"]] = info

    categorical = [
        name for name, info in columns.items() if _is_categorical(info, row_count)
    ]
    if categorical:
        values = execute(_values_sql(node, categorical))
        if isinstance(values, str):
            raise RuntimeError(
                f"values query failed for {node['qualified_name']}: {values}"
            )
        for position, name in enumerate(categorical):
            subset = values[values["c"] == position]
            columns[name]["values"] = [
                [str(v), int(n)] for v, n in zip(subset["v"], subset["n"], strict=True)
            ]

    _attach_case_variant_separators(node, columns, execute)

    return {"row_count": row_count, "columns": columns}


def _attach_case_variant_separators(
    node: dict[str, Any], columns: dict[str, dict[str, Any]], execute
) -> None:
    """For values differing only by case, find the column that tells them apart.

    'Total' and 'total' look like one value to a reader but can be two
    different measures (e.g. monthly vs annual). Without saying which
    column separates them, the generator picks one at random.
    """
    separators = [
        name
        for name, info in columns.items()
        if info.get("values") and 2 <= info["distinct_count"] <= SEPARATOR_MAX_DISTINCT
    ]
    for name, info in columns.items():
        groups = _case_variant_groups([v for v, _ in info.get("values", [])])
        if not groups:
            continue
        info["case_variants"] = groups
        candidates = [s for s in separators if s != name]
        if not candidates:
            continue
        variants = [value for group in groups for value in group]
        result = execute(_separator_sql(node, name, variants, candidates))
        if isinstance(result, str) or result is None or result.empty:
            continue
        distinctions = []
        for position, separator in enumerate(candidates):
            subset = result[result["s"] == position]
            for group in groups:
                mapping = {}
                for variant in group:
                    others = subset[subset["v"] == variant]["o"].dropna().unique()
                    if len(others) == 1:
                        mapping[variant] = str(others[0])
                if len(mapping) == len(group) and len(set(mapping.values())) == len(
                    group
                ):
                    distinctions.append({"by": separator, "mapping": mapping})
        if distinctions:
            info["case_variant_separators"] = distinctions


def collect_facts(graph: dict[str, Any], connection_id: str) -> dict[str, Any]:
    """Profile every table in the graph against the target database."""
    from app.utils.sql_executor import SqlExec

    executor = SqlExec(connection_id=connection_id)
    if not executor.database_url.startswith("postgresql"):
        logger.warning(
            "SEMANTIC PROFILE — profiling supports PostgreSQL only; skipping %s",
            connection_id,
        )
        return {}

    def execute(sql: str):
        return executor.sql_executor(sql, timeout_ms=PROFILE_QUERY_TIMEOUT_MS)

    tables: dict[str, Any] = {}
    nodes = graph.get("nodes", [])
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TABLES) as pool:
        futures = {
            node["qualified_name"]: pool.submit(_profile_table, node, execute)
            for node in nodes
        }
        for qualified, future in futures.items():
            try:
                tables[qualified] = future.result()
            except Exception:
                logger.exception("SEMANTIC PROFILE — failed to profile %s", qualified)
    return tables


# ─── LLM inference (grain, scenarios, versions) ───────────────────────────────


def _profile_for_prompt(tables: dict[str, Any]) -> str:
    lines = []
    for qualified, table in tables.items():
        lines.append(f"TABLE {qualified} (rows={table['row_count']})")
        for name, info in table["columns"].items():
            description = (
                f"  - {name} [{info['family']}] distinct={info['distinct_count']}"
            )
            if "min" in info:
                description += f" range={info['min']}..{info['max']}"
            values = info.get("values")
            if values:
                shown = [v for v, _ in values[:25]]
                more = len(values) - len(shown)
                description += f" values={json.dumps(shown)}" + (
                    f" (+{more} more)" if more > 0 else ""
                )
            lines.append(description)
    return "\n".join(lines)


def _validated_inference(
    raw: dict[str, Any], tables: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Keep only inferred facts that reference real columns and values."""
    validated: dict[str, dict[str, Any]] = {}
    by_lower = {name.lower(): name for name in tables}
    for item in raw.get("tables") or []:
        qualified = by_lower.get(str(item.get("table", "")).lower())
        if not qualified:
            continue
        columns = tables[qualified]["columns"]

        def real(column: Any, columns: dict[str, Any] = columns) -> bool:
            return isinstance(column, str) and column in columns

        scenario_columns = []
        for scenario in item.get("scenario_columns") or []:
            column = scenario.get("column")
            values = {v for v, _ in columns.get(column, {}).get("values", [])}
            if real(column) and scenario.get("default") in values:
                scenario_columns.append(
                    {"column": column, "default": scenario["default"]}
                )

        version_columns = []
        for version in item.get("version_columns") or []:
            if real(version.get("column")) and real(version.get("latest_by")):
                version_columns.append(
                    {
                        "column": version["column"],
                        "latest_by": version["latest_by"],
                        "per": [c for c in version.get("per") or [] if real(c)],
                    }
                )

        validated[qualified] = {
            "grain": str(item.get("grain") or "").strip()[:300],
            "row_metric_columns": [
                c for c in item.get("row_metric_columns") or [] if real(c)
            ],
            "scenario_columns": scenario_columns,
            "version_columns": version_columns,
            "notes": [str(n).strip()[:300] for n in (item.get("notes") or [])[:3]],
        }
    return validated


def infer_table_semantics(
    tables: dict[str, Any], graph: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Ask an LLM to state each table's grain and scenario/version columns."""
    from langchain_core.output_parsers import JsonOutputParser

    from app.prompt.prompts import semantic_profile_inference_prompt
    from app.utils.llm_factory import create_chat_llm

    if not tables:
        return {}
    relationships = [
        f"{e['from_table']}.{e['from_column']} -> {e['to_table']}.{e['to_column']}"
        for e in graph.get("edges", [])
    ]
    prompt = semantic_profile_inference_prompt.format(
        profile=_profile_for_prompt(tables),
        relationships="\n".join(relationships) or "(none)",
    )
    llm = create_chat_llm(purpose="semantic_profile", temperature=0)
    response = llm.invoke([{"role": "user", "content": prompt}])
    raw = JsonOutputParser().parse(str(response.content))
    return _validated_inference(raw if isinstance(raw, dict) else {}, tables)


# ─── Build, persist, load ─────────────────────────────────────────────────────


def build_profile(
    graph: dict[str, Any], connection_id: str, *, infer: bool
) -> dict[str, Any]:
    tables = collect_facts(graph, connection_id)
    profile: dict[str, Any] = {
        "version": PROFILE_VERSION,
        "profiled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tables": tables,
        "inferred": {},
    }
    if infer and tables:
        try:
            profile["inferred"] = infer_table_semantics(tables, graph)
        except Exception:
            logger.exception(
                "SEMANTIC PROFILE — inference failed for %s", connection_id
            )
    return profile


def _profile_matches_graph(profile: Any, graph: dict[str, Any]) -> bool:
    if not isinstance(profile, dict) or profile.get("version") != PROFILE_VERSION:
        return False
    graph_tables = {node["qualified_name"] for node in graph.get("nodes", [])}
    return graph_tables <= set(profile.get("tables", {}))


def refresh_connection_profile(
    connection_id: str, graph: dict[str, Any] | None = None, *, infer: bool = True
) -> dict[str, Any]:
    """Rebuild the profile from the live data and persist it into the graph."""
    from app.settings_db.repositories import SchemaGraphRepository
    from app.target_db.controllers import load_schema_graph

    graph = graph or load_schema_graph(connection_id)
    if not graph:
        raise ValueError(f"No schema graph for connection {connection_id}.")
    profile = build_profile(graph, connection_id, infer=infer)
    if not profile["tables"]:
        return profile
    if not infer:
        # A facts-only refresh must not throw away inference from a full build.
        previous = (graph.get("profile") or {}).get("inferred") or {}
        profile["inferred"] = {
            table: notes
            for table, notes in previous.items()
            if table in profile["tables"]
        }
    SchemaGraphRepository().upsert_graph(connection_id, {**graph, "profile": profile})
    with _cache_lock:
        _cache[connection_id] = profile
    logger.info(
        "SEMANTIC PROFILE — stored for %s: %d table(s), inferred=%d",
        connection_id,
        len(profile["tables"]),
        len(profile["inferred"]),
    )
    return profile


def get_connection_profile(
    connection_id: str | None, graph: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Return the profile for a connection, building the facts if missing.

    Order: the profile stored in the graph, then the in-process cache, then a
    facts-only build (fast) that is persisted for next time. Inference is never
    run here — it belongs to graph builds and explicit refreshes.
    """
    if not connection_id or not graph:
        return None
    stored = graph.get("profile")
    if _profile_matches_graph(stored, graph):
        return stored
    with _cache_lock:
        cached = _cache.get(connection_id)
        lock = _build_locks.setdefault(connection_id, threading.Lock())
    if _profile_matches_graph(cached, graph):
        return cached
    with lock:
        with _cache_lock:
            cached = _cache.get(connection_id)
        if _profile_matches_graph(cached, graph):
            return cached
        try:
            logger.info(
                "SEMANTIC PROFILE — missing for %s; collecting facts", connection_id
            )
            return refresh_connection_profile(connection_id, graph, infer=False)
        except Exception:
            logger.exception("SEMANTIC PROFILE — build failed for %s", connection_id)
            return None


# ─── Lookups used by the orchestrator ─────────────────────────────────────────


def _table_entry(
    profile: dict[str, Any] | None, table_name: str
) -> tuple[str | None, dict[str, Any] | None]:
    if not profile:
        return None, None
    tables = profile.get("tables", {})
    wanted = table_name.lower()
    for qualified, entry in tables.items():
        if qualified.lower() == wanted or qualified.lower().split(".", 1)[-1] == wanted:
            return qualified, entry
    return None, None


def get_column_values(
    profile: dict[str, Any] | None, table_name: str, column_name: str
) -> tuple[list[str], int] | None:
    """Return (all values, cardinality) when the profile holds the full list.

    None means "unknown here" — the caller should fall back to a live query.
    """
    _, entry = _table_entry(profile, table_name)
    if not entry:
        return None
    columns = entry.get("columns", {})
    info = columns.get(column_name) or next(
        (v for k, v in columns.items() if k.lower() == column_name.lower()), None
    )
    if not info or "values" not in info:
        return None
    values = [value for value, _ in info["values"]]
    if len(values) < info.get("distinct_count", 0):
        return None
    return values, info["distinct_count"]


def render_table_profile(
    profile: dict[str, Any] | None, qualified: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (table-level, per-column) prompt payloads for one table."""
    _, entry = _table_entry(profile, qualified)
    if not entry:
        return {}, {}
    table_payload: dict[str, Any] = {"row_count": entry.get("row_count")}
    inferred = (profile or {}).get("inferred", {}).get(qualified) or {}
    for key in (
        "grain",
        "row_metric_columns",
        "scenario_columns",
        "version_columns",
        "notes",
    ):
        if inferred.get(key):
            table_payload[key] = inferred[key]

    column_payloads: dict[str, dict[str, Any]] = {}
    for name, info in entry.get("columns", {}).items():
        payload: dict[str, Any] = {"distinct": info["distinct_count"]}
        if "min" in info:
            payload["range"] = [info["min"], info["max"]]
        values = info.get("values")
        if values:
            if (info.get("max_length") or 0) > PROMPT_LONG_VALUE_LENGTH:
                payload["example_values"] = [
                    v for v, _ in values[:PROMPT_LONG_VALUE_EXAMPLES]
                ]
            elif len(values) <= PROMPT_FULL_VALUE_LIMIT:
                payload["values"] = [v for v, _ in values]
            else:
                payload["top_values"] = [v for v, _ in values[:PROMPT_TOP_VALUES]]
        if info.get("case_variants"):
            payload["case_sensitive_values"] = info["case_variants"]
        if info.get("case_variant_separators"):
            payload["distinguished_by"] = info["case_variant_separators"]
        column_payloads[name] = payload
    return table_payload, column_payloads

"""DB-mocked tests for the HITL value resolver.

The DB is mocked at ``SqlExec.sql_executor`` — the single seam every query goes
through. That exercises the real code paths (SQL construction, result handling,
error handling) without a database, which pure-function tests cannot reach.

Assertions check the generated SQL as well as the return value: a resolver that
returns the right answer from the wrong query is still broken, and the schema
qualification here is exactly what was silently wrong before
(`"public"."workorder"` for a table living in `production`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent.value_resolver_agent import (
    _lookup_schema_for_table,
    _qualify_table,
    get_column_distinct_values,
    get_fuzzy_matches,
    value_exists_exactly,
)

# A schema graph as stored in settings.schema_graphs.
SCHEMA_GRAPH = {
    "nodes": [
        {"table": "workorder", "schema": "production", "columns": []},
        {"table": "sample", "schema": "public", "columns": []},
    ],
    "edges": [],
}


class TestValueExistsExactly:
    def test_returns_true_when_the_query_returns_a_row(self, patch_executor) -> None:
        patch_executor(pd.DataFrame({"1": [1]}))

        assert (
            value_exists_exactly("sample", "brand_name", "Brand A", SCHEMA_GRAPH)
            is True
        )

    def test_returns_false_when_the_query_returns_no_rows(self, patch_executor) -> None:
        """A miss is what makes disambiguation worthwhile."""
        patch_executor(pd.DataFrame())

        assert (
            value_exists_exactly("sample", "brand_name", "Women", SCHEMA_GRAPH) is False
        )

    def test_qualifies_the_table_from_the_schema_graph(self, patch_executor) -> None:
        """The exact bug that shipped: workorder lives in production, not public.

        A query against the wrong schema either errors or returns nothing, and
        because this function fails open to False it silently triggered HITL for
        values that were perfectly valid.
        """
        recorder = patch_executor(pd.DataFrame({"1": [1]}))

        value_exists_exactly("workorder", "scrapreasonid", "7", SCHEMA_GRAPH)

        assert '"production"."workorder"' in recorder.queries[0]
        assert '"public"."workorder"' not in recorder.queries[0]

    def test_casts_the_column_to_text_so_the_check_is_type_agnostic(
        self, patch_executor
    ) -> None:
        """A year (INTEGER) and a country (TEXT) must be checked identically."""
        recorder = patch_executor(pd.DataFrame({"1": [1]}))

        value_exists_exactly("sample", "date_year", "2016", SCHEMA_GRAPH)

        assert 'CAST("date_year" AS TEXT)' in recorder.queries[0]

    def test_limits_to_one_row(self, patch_executor) -> None:
        """Existence needs one row; scanning the table would be wasteful."""
        recorder = patch_executor(pd.DataFrame({"1": [1]}))

        value_exists_exactly("sample", "brand_name", "Brand A", SCHEMA_GRAPH)

        assert "LIMIT 1" in recorder.queries[0]

    def test_escapes_single_quotes_in_the_value(self, patch_executor) -> None:
        """`O'Brien` must not break the literal or inject SQL."""
        recorder = patch_executor(pd.DataFrame())

        value_exists_exactly("sample", "customer", "O'Brien", SCHEMA_GRAPH)

        assert "'O''Brien'" in recorder.queries[0]

    def test_injection_attempt_is_neutralised(self, patch_executor) -> None:
        """A quote-escape attempt must stay inside the string literal."""
        recorder = patch_executor(pd.DataFrame())

        value_exists_exactly("sample", "c", "x' OR '1'='1", SCHEMA_GRAPH)

        sql = recorder.queries[0]
        # Every embedded quote is doubled, so no new statement or predicate opens.
        assert "'x'' OR ''1''=''1'" in sql

    def test_fails_open_when_the_query_errors(self, patch_executor) -> None:
        """SqlExec returns a str on failure. Failing open keeps resolution alive
        rather than trusting an unverified value."""
        patch_executor("relation does not exist")

        assert value_exists_exactly("sample", "bad_col", "x", SCHEMA_GRAPH) is False

    def test_empty_value_short_circuits_without_querying(self, patch_executor) -> None:
        recorder = patch_executor(pd.DataFrame({"1": [1]}))

        assert (
            value_exists_exactly("sample", "brand_name", "   ", SCHEMA_GRAPH) is False
        )
        assert recorder.queries == []

    def test_unknown_table_fails_open_without_querying(self, patch_executor) -> None:
        """No graph entry means no trustworthy schema, so do not guess."""
        recorder = patch_executor(pd.DataFrame({"1": [1]}))

        assert value_exists_exactly("mystery", "col", "x", SCHEMA_GRAPH) is False
        assert recorder.queries == []

    def test_already_qualified_table_is_used_as_given(self, patch_executor) -> None:
        recorder = patch_executor(pd.DataFrame())

        value_exists_exactly("sales.orders", "id", "1", SCHEMA_GRAPH)

        assert '"sales"."orders"' in recorder.queries[0]


class TestGetColumnDistinctValues:
    def test_returns_values_and_cardinality(self, patch_executor) -> None:
        patch_executor(
            pd.DataFrame({"cnt": [3]}),
            pd.DataFrame({"category": ["Category A", "Category B", "Category C"]}),
        )

        values, cardinality = get_column_distinct_values(
            "sample", "category", schema_graph=SCHEMA_GRAPH
        )

        assert cardinality == 3
        assert values == ["Category A", "Category B", "Category C"]

    def test_counts_distinct_before_fetching(self, patch_executor) -> None:
        """Cardinality decides whether all values or a sample are fetched."""
        recorder = patch_executor(
            pd.DataFrame({"cnt": [2]}), pd.DataFrame({"c": ["a", "b"]})
        )

        get_column_distinct_values("sample", "c", schema_graph=SCHEMA_GRAPH)

        assert "COUNT(DISTINCT" in recorder.queries[0]
        assert "SELECT DISTINCT" in recorder.queries[1]

    def test_uses_the_graph_schema_for_both_queries(self, patch_executor) -> None:
        recorder = patch_executor(
            pd.DataFrame({"cnt": [1]}),
            # The result column must be named for the column queried: the
            # implementation reads it by name.
            pd.DataFrame({"scrapreasonid": ["7"]}),
        )

        get_column_distinct_values(
            "workorder", "scrapreasonid", schema_graph=SCHEMA_GRAPH
        )

        assert all('"production"."workorder"' in sql for sql in recorder.queries)

    def test_applies_the_row_limit(self, patch_executor) -> None:
        recorder = patch_executor(
            pd.DataFrame({"cnt": [500]}), pd.DataFrame({"c": ["x"]})
        )

        get_column_distinct_values("sample", "c", limit=25, schema_graph=SCHEMA_GRAPH)

        assert "LIMIT 25" in recorder.queries[1]

    def test_excludes_nulls_from_the_value_list(self, patch_executor) -> None:
        """A NULL is not a value a user can be asked to pick."""
        recorder = patch_executor(
            pd.DataFrame({"cnt": [2]}), pd.DataFrame({"c": ["a", "b"]})
        )

        get_column_distinct_values("sample", "c", schema_graph=SCHEMA_GRAPH)

        assert "IS NOT NULL" in recorder.queries[1]

    def test_cardinality_query_failure_returns_empty(self, patch_executor) -> None:
        patch_executor("permission denied")

        assert get_column_distinct_values("sample", "c", schema_graph=SCHEMA_GRAPH) == (
            [],
            0,
        )

    def test_unknown_table_returns_empty_without_querying(self, patch_executor) -> None:
        recorder = patch_executor(pd.DataFrame({"cnt": [1]}))

        assert get_column_distinct_values(
            "mystery", "c", schema_graph=SCHEMA_GRAPH
        ) == ([], 0)
        assert recorder.queries == []

    def test_values_are_returned_as_strings(self, patch_executor) -> None:
        """Numeric values must be comparable against the user's typed text."""
        patch_executor(pd.DataFrame({"cnt": [2]}), pd.DataFrame({"year": [2015, 2016]}))

        values, _ = get_column_distinct_values(
            "sample", "year", schema_graph=SCHEMA_GRAPH
        )

        assert all(isinstance(value, str) for value in values)


class TestQualifyTable:
    def test_resolves_a_bare_table_from_the_graph(self) -> None:
        assert _qualify_table("workorder", SCHEMA_GRAPH) == '"production"."workorder"'

    def test_passes_through_an_already_qualified_name(self) -> None:
        assert _qualify_table("sales.orders", SCHEMA_GRAPH) == '"sales"."orders"'

    def test_matching_is_case_insensitive(self) -> None:
        assert _qualify_table("WorkOrder", SCHEMA_GRAPH) == '"production"."WorkOrder"'

    def test_raises_for_a_table_absent_from_the_graph(self) -> None:
        """Guessing "public" produced queries against the wrong schema."""
        with pytest.raises(ValueError, match="not in the connection's schema graph"):
            _qualify_table("mystery", SCHEMA_GRAPH)

    def test_raises_when_no_graph_is_supplied(self) -> None:
        with pytest.raises(ValueError, match="schema graph"):
            _qualify_table("workorder", None)

    def test_lookup_returns_none_for_a_missing_table(self) -> None:
        assert _lookup_schema_for_table("nope", SCHEMA_GRAPH) is None
        assert _lookup_schema_for_table("workorder", None) is None


class TestGetFuzzyMatches:
    def test_exact_match_ranks_first(self) -> None:
        result = get_fuzzy_matches("Category B", ["Category A", "Category B"])

        assert result[0] == "Category B"

    def test_typo_still_matches_by_prefix(self) -> None:
        result = get_fuzzy_matches("categry c", ["Category A", "Category C"])

        assert "Category C" in result

    def test_unrelated_value_returns_nothing(self) -> None:
        """The bug this floor was added for: "Women" against Category A/B/C.

        Without a minimum score every value gets a small nonzero rank from
        character overlap, and the top one was offered as though it matched.
        """
        assert (
            get_fuzzy_matches("Women", ["Category A", "Category B", "Category C"]) == []
        )

    def test_case_insensitive(self) -> None:
        assert get_fuzzy_matches("CATEGORY B", ["Category B"]) == ["Category B"]

    def test_respects_top_n(self) -> None:
        values = [f"Category {letter}" for letter in "ABCDEFGH"]

        assert len(get_fuzzy_matches("Category", values, top_n=3)) == 3

    def test_min_score_is_configurable(self) -> None:
        """A caller can widen the net; the default is deliberately strict."""
        unrelated = ["Category A"]

        assert get_fuzzy_matches("Women", unrelated) == []
        assert get_fuzzy_matches("Women", unrelated, min_score=0.0) == ["Category A"]

    def test_empty_candidate_list_returns_empty(self) -> None:
        assert get_fuzzy_matches("anything", []) == []

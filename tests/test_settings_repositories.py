"""Repository tests using an injected fake settings-DB gateway.

The repositories take ``gateway`` as a constructor argument, so no patching is
required — a FakeGateway goes straight in. That injection point is what makes the
whole settings-DB layer testable; worth keeping as new repositories are added.

What these tests cover that pure-function tests cannot: UUID parsing and its
failure modes, row-to-dict conversion, whether a write commits, and the
None-versus-empty-list contracts callers branch on.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.settings_db.repositories import (
    DBConnectionRepository,
    SchemaGraphRepository,
)

GRAPH = {
    "nodes": [{"table": "orders", "schema": "sales", "qualified_name": "sales.orders"}],
    "edges": [],
}


class TestSchemaGraphRepository:
    """settings.schema_graphs is now the ONLY schema source, so its read path is
    load-bearing for every question asked."""

    def test_returns_the_stored_graph(self, fake_gateway) -> None:
        connection_id = uuid4()
        row_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": row_id,
                    "db_connection_id": connection_id,
                    "graph": GRAPH,
                    "updated_at": None,
                }
            ]
        )

        result = SchemaGraphRepository(gateway).get_graph_by_connection(
            str(connection_id)
        )

        assert result["graph"] == GRAPH

    def test_uuid_fields_are_stringified_for_json_serialisation(
        self, fake_gateway
    ) -> None:
        """A raw UUID breaks jsonable_encoder downstream."""
        connection_id, row_id = uuid4(), uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": row_id,
                    "db_connection_id": connection_id,
                    "graph": GRAPH,
                    "updated_at": None,
                }
            ]
        )

        result = SchemaGraphRepository(gateway).get_graph_by_connection(
            str(connection_id)
        )

        assert result["id"] == str(row_id)
        assert isinstance(result["db_connection_id"], str)

    def test_returns_none_when_no_row_exists(self, fake_gateway) -> None:
        """An unprofiled connection must be distinguishable from an empty graph."""
        gateway = fake_gateway(None)

        assert (
            SchemaGraphRepository(gateway).get_graph_by_connection(str(uuid4())) is None
        )

    def test_malformed_connection_id_returns_none_without_querying(
        self, fake_gateway
    ) -> None:
        """A bad id is a caller error, not a DB round-trip."""
        gateway = fake_gateway([{"id": uuid4()}])

        assert (
            SchemaGraphRepository(gateway).get_graph_by_connection("not-a-uuid") is None
        )
        assert gateway.session.statements == []

    def test_read_path_issues_exactly_one_statement(self, fake_gateway) -> None:
        """Guards against an N+1 creeping in: this runs on every request."""
        connection_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": uuid4(),
                    "db_connection_id": connection_id,
                    "graph": GRAPH,
                    "updated_at": None,
                }
            ]
        )

        SchemaGraphRepository(gateway).get_graph_by_connection(str(connection_id))

        assert len(gateway.session.statements) == 1

    def test_read_path_does_not_commit(self, fake_gateway) -> None:
        """A read holding a write transaction open would be wasteful."""
        connection_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": uuid4(),
                    "db_connection_id": connection_id,
                    "graph": GRAPH,
                    "updated_at": None,
                }
            ]
        )

        SchemaGraphRepository(gateway).get_graph_by_connection(str(connection_id))

        assert gateway.session.committed is False

    def test_upsert_commits_the_write(self, fake_gateway) -> None:
        connection_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": uuid4(),
                    "db_connection_id": connection_id,
                    "graph": GRAPH,
                    "updated_at": None,
                }
            ]
        )

        SchemaGraphRepository(gateway).upsert_graph(str(connection_id), GRAPH)

        assert gateway.session.committed is True

    def test_upsert_returns_the_stored_row(self, fake_gateway) -> None:
        connection_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": uuid4(),
                    "db_connection_id": connection_id,
                    "graph": GRAPH,
                    "updated_at": None,
                }
            ]
        )

        result = SchemaGraphRepository(gateway).upsert_graph(str(connection_id), GRAPH)

        assert result["graph"] == GRAPH
        assert isinstance(result["id"], str)

    def test_upsert_returns_none_when_nothing_comes_back(self, fake_gateway) -> None:
        gateway = fake_gateway(None)

        assert SchemaGraphRepository(gateway).upsert_graph(str(uuid4()), GRAPH) is None

    def test_upsert_rejects_a_malformed_connection_id(self, fake_gateway) -> None:
        """Unlike the read path, the write raises: silently dropping a graph
        would leave the connection permanently unprofiled with no signal."""
        with pytest.raises(ValueError):
            SchemaGraphRepository(fake_gateway()).upsert_graph("not-a-uuid", GRAPH)


class TestDBConnectionRepository:
    def test_returns_the_connection_row(self, fake_gateway) -> None:
        connection_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": connection_id,
                    "engine": "postgres",
                    "db_host": "db.internal",
                    "db_port": 5432,
                    "db_name": "analytics",
                    "db_user": "reader",
                    "db_password": "secret",
                    "db_schema": "public",
                    "use_ssl": True,
                    "extra_options": None,
                    "display_config": {"number_format_suffixes": {"billion": "Cr"}},
                }
            ]
        )

        result = DBConnectionRepository(gateway).get_connection_by_id(
            str(connection_id)
        )

        assert result["engine"] == "postgres"

    def test_display_config_is_returned(self, fake_gateway) -> None:
        """The Cr/Lakh number formatting is read from here on every response."""
        connection_id = uuid4()
        gateway = fake_gateway(
            [
                {
                    "id": connection_id,
                    "engine": "postgres",
                    "display_config": {
                        "number_format_suffixes": {"billion": "Cr"},
                        "number_format_thresholds": {"billion": 10000000},
                    },
                }
            ]
        )

        result = DBConnectionRepository(gateway).get_connection_by_id(
            str(connection_id)
        )

        assert result["display_config"]["number_format_suffixes"]["billion"] == "Cr"

    def test_returns_none_for_an_unknown_connection(self, fake_gateway) -> None:
        gateway = fake_gateway(None)

        assert (
            DBConnectionRepository(gateway).get_connection_by_id(str(uuid4())) is None
        )

    def test_read_issues_one_statement_and_does_not_commit(self, fake_gateway) -> None:
        connection_id = uuid4()
        gateway = fake_gateway([{"id": connection_id, "engine": "postgres"}])

        DBConnectionRepository(gateway).get_connection_by_id(str(connection_id))

        assert len(gateway.session.statements) == 1
        assert gateway.session.committed is False

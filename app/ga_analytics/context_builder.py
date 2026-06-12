"""Discover a GA4 property's available dimensions and metrics ("schema graph").

Analogous to the SQL schema graph: we fetch the property metadata, shape it into
a JSON artifact stored in settings.schema_graphs, and render a compact summary
the LLM uses to translate NL questions into GA4 report requests.
"""

from __future__ import annotations

import logging
from typing import Any

from app.ga_analytics.ga_client import (
    get_ga_client_for_connection,
    normalize_property_id,
)

logger = logging.getLogger(__name__)


def build_ga_graph(connection_id: str) -> dict[str, Any]:
    """Fetch GA4 dimensions/metrics for a connection and persist as a graph.

    Returns the stored graph payload. Best-effort: logs and returns an empty
    graph on failure so a background trigger never crashes.
    """
    from app.settings_db.controllers import get_settings_connection_credentials
    from app.settings_db.repositories import SchemaGraphRepository

    credentials = get_settings_connection_credentials(connection_id)
    property_id = normalize_property_id(
        str(credentials.get("ga_property_id", ""))
    )
    if not property_id:
        logger.warning("Connection %s has no GA property id", connection_id)
        return {}

    client = get_ga_client_for_connection(connection_id)
    if client is None:
        logger.warning("No GA client for connection %s", connection_id)
        return {}

    try:
        from google.analytics.data_v1beta.types import GetMetadataRequest

        metadata = client.get_metadata(
            GetMetadataRequest(name=f"{property_id}/metadata")
        )
    except Exception as error:  # noqa: BLE001 - best effort
        logger.exception("GA metadata fetch failed for %s: %s", connection_id, error)
        return {}

    dimensions = [
        {
            "api_name": d.api_name,
            "ui_name": d.ui_name,
            "description": d.description,
            "category": d.category,
        }
        for d in metadata.dimensions
    ]
    metrics = [
        {
            "api_name": m.api_name,
            "ui_name": m.ui_name,
            "description": m.description,
            "category": m.category,
            "type": getattr(m, "type_", None) and str(m.type_),
        }
        for m in metadata.metrics
    ]

    graph = {
        "connection_id": connection_id,
        "connection_type": "google_analytics",
        "property_id": property_id,
        "dimensions": dimensions,
        "metrics": metrics,
        "dimension_count": len(dimensions),
        "metric_count": len(metrics),
    }

    SchemaGraphRepository().upsert_graph(connection_id, graph)
    logger.info(
        "Stored GA graph for connection_id=%s: %d dimensions, %d metrics",
        connection_id,
        len(dimensions),
        len(metrics),
    )
    return graph


def render_ga_summary(graph: dict[str, Any]) -> str:
    """Render a GA graph into a compact summary string for the LLM/router.

    Lists dimensions and metrics by API name (the names the Data API expects),
    grouped, with UI names for context. Kept compact to fit the prompt.
    """
    if not graph or graph.get("connection_type") != "google_analytics":
        return ""

    def _fmt(items: list[dict[str, Any]], limit: int = 80) -> str:
        shown = items[:limit]
        parts = [
            f"{it['api_name']}"
            + (f" ({it['ui_name']})" if it.get("ui_name") else "")
            for it in shown
        ]
        suffix = f" … +{len(items) - limit} more" if len(items) > limit else ""
        return ", ".join(parts) + suffix

    dimensions = graph.get("dimensions", [])
    metrics = graph.get("metrics", [])

    return (
        f"Google Analytics 4 property {graph.get('property_id')}.\n"
        f"DIMENSIONS: {_fmt(dimensions)}\n"
        f"METRICS: {_fmt(metrics)}"
    )

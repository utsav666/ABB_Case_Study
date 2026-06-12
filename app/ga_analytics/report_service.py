"""Execute a GA4 runReport spec and normalize the response into a DataFrame."""

from __future__ import annotations

import logging
from typing import Any

from pandas import DataFrame

from app.ga_analytics.ga_client import (
    get_ga_client_for_connection,
    normalize_property_id,
)

logger = logging.getLogger(__name__)


def run_ga_report(connection_id: str, spec: dict[str, Any]) -> DataFrame:
    """Run a GA4 report for a connection from a spec dict.

    spec keys: dimensions[], metrics[], start_date, end_date, limit, order_by.
    Returns a pandas DataFrame (dimension + metric columns).
    Raises on API/credential errors so the orchestrator can retry.
    """
    from app.settings_db.controllers import get_settings_connection_credentials
    from google.analytics.data_v1beta.types import (
        DateRange,
        Dimension,
        Metric,
        MetricType,
        OrderBy,
        RunReportRequest,
    )

    credentials = get_settings_connection_credentials(connection_id)
    property_id = normalize_property_id(str(credentials.get("ga_property_id", "")))
    if not property_id:
        raise ValueError("Connection has no GA4 property id.")

    client = get_ga_client_for_connection(connection_id)
    if client is None:
        raise ValueError("Could not build a GA client for this connection.")

    dimension_names = list(spec.get("dimensions", []) or [])
    metric_names = list(spec.get("metrics", []) or [])
    if not metric_names:
        raise ValueError("At least one metric is required.")

    request_kwargs: dict[str, Any] = {
        "property": property_id,
        "dimensions": [Dimension(name=n) for n in dimension_names],
        "metrics": [Metric(name=n) for n in metric_names],
        "date_ranges": [
            DateRange(
                start_date=str(spec.get("start_date", "30daysAgo")),
                end_date=str(spec.get("end_date", "today")),
            )
        ],
        "limit": int(spec.get("limit", 100) or 100),
    }

    order_by = spec.get("order_by")
    if order_by and order_by in metric_names:
        request_kwargs["order_bys"] = [
            OrderBy(
                metric=OrderBy.MetricOrderBy(metric_name=order_by),
                desc=True,
            )
        ]

    response = client.run_report(RunReportRequest(**request_kwargs))

    # Build rows: dimension values then metric values, typed.
    metric_types = [mh.type_ for mh in response.metric_headers]
    records: list[dict[str, Any]] = []
    for row in response.rows:
        record: dict[str, Any] = {}
        for name, value in zip(dimension_names, row.dimension_values):
            record[name] = value.value
        for name, value, mtype in zip(
            metric_names, row.metric_values, metric_types
        ):
            record[name] = _coerce_metric(value.value, mtype, MetricType)
        records.append(record)

    logger.info(
        "GA report for %s returned %d rows", connection_id, len(records)
    )
    return DataFrame(records)


def _coerce_metric(raw: str, mtype: Any, MetricType: Any) -> Any:
    """Coerce a GA metric string to int/float where appropriate."""
    try:
        if mtype == MetricType.TYPE_INTEGER:
            return int(raw)
        if mtype in (
            MetricType.TYPE_FLOAT,
            MetricType.TYPE_SECONDS,
            MetricType.TYPE_CURRENCY,
            getattr(MetricType, "TYPE_MILLISECONDS", None),
        ):
            return float(raw)
    except (ValueError, TypeError):
        return raw
    return raw

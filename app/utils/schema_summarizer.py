from app.sampledata.data_loader import DataLoader


def _render_semantic_profile(column_name: str, profile: dict) -> str:
    """Render a single column's semantic profile into a compact string.

    This is the function that converts the enriched JSON data into
    prompt-friendly text consumed by the Router, TableIdentifier, and
    SQLGenerator.

    Parameters
    ----------
    column_name:
        Raw column name.
    profile:
        The ``semantic_profile`` dict attached to the column.

    Returns
    -------
    str
        A compact human-readable summary of the column's data.
    """
    parts: list[str] = []

    # Description from LLM annotation
    desc = profile.get("description", "").strip()
    if desc:
        parts.append(desc)

    # TEXT columns — show values / aliases
    distinct = profile.get("distinct_values")
    if distinct:
        # Show value descriptions if available, else raw values
        value_descs = profile.get("value_descriptions", {})
        if value_descs:
            mapped = [
                f"{v}={value_descs[v]}" if v in value_descs else v
                for v in distinct[:20]
            ]
        else:
            mapped = distinct[:20]
        suffix = f" +{len(distinct) - 20} more" if len(distinct) > 20 else ""
        parts.append(f"values=[{', '.join(mapped)}{suffix}]")
    elif profile.get("top_values"):
        top = [tv["value"] for tv in profile["top_values"][:15]]
        parts.append(f"top values=[{', '.join(top)} …]")

    # Aliases
    aliases = profile.get("natural_language_aliases", {})
    if aliases:
        alias_pairs = [f'"{k}"→{v}' for k, v in list(aliases.items())[:10]]
        parts.append(f"aliases: {', '.join(alias_pairs)}")

    # NUMERIC columns
    if profile.get("min_value") is not None:
        parts.append(
            f"range {profile.get('min_value')}–{profile.get('max_value')}"
            f", mean={profile.get('mean_value')}"
        )

    # TEMPORAL columns
    if profile.get("min_date"):
        parts.append(
            f"range {profile['min_date']} → {profile.get('max_date')}"
            f", {profile.get('granularity', 'unknown')} granularity"
        )

    # Query hints
    hints = profile.get("query_hints", "").strip()
    if hints:
        parts.append(f"⚠️ {hints}")

    return "; ".join(parts)


class Summarizer:
    @staticmethod
    def summarize_schema(include_semantic_profile: bool = True) -> str:
        """Return a schema summary enriched with any configured column labels.

        Parameters
        ----------
        include_semantic_profile:
            When ``True`` (default), appends semantic profile data (value
            descriptions, aliases, ranges, query hints) to each column's
            summary line.  Set to ``False`` for a compact structure-only
            summary.
        """
        data = DataLoader.load_data()
        table_summaries: list[str] = []

        for table_name, table_info in data.get("tables", {}).items():
            column_summaries: list[str] = []
            for column_name, column_info in table_info.get("columns", {}).items():
                if not column_info.get("include_in_assistant", True):
                    continue

                display_name = str(column_info.get("display_name", column_name)).strip()
                description = str(column_info.get("description", "")).strip()

                column_summary = column_name
                if display_name and display_name != column_name:
                    column_summary = f"{column_name} ({display_name})"
                if description:
                    column_summary = f"{column_summary}: {description}"

                # Append semantic profile when available and enabled
                if include_semantic_profile:
                    sem_profile = column_info.get("semantic_profile")
                    if sem_profile:
                        rendered = _render_semantic_profile(column_name, sem_profile)
                        if rendered:
                            column_summary = f"{column_summary} [{rendered}]"

                column_summaries.append(column_summary)

            if not column_summaries:
                column_summaries = list(table_info.get("columns", {}).keys())

            primary_keys = table_info.get("primary_key", [])
            foreign_keys = [
                f"{foreign_key['from'][0]}→{foreign_key['to_table']}.{foreign_key['to_columns'][0]}"
                for foreign_key in table_info.get("foreign_keys", [])
            ]

            schema_name = table_info.get("schema", "")
            display_name = f"{schema_name}.{table_name}" if schema_name else table_name
            summary = f"{display_name}({', '.join(column_summaries)})"
            if primary_keys:
                summary += f" PK: {', '.join(primary_keys)}"
            if foreign_keys:
                summary += f" FK: {', '.join(foreign_keys)}"

            table_summaries.append(summary)

        return " ".join(table_summaries)


##### Unit 3 #######
# if __name__ == "__main__":
#     res=Summarizer.summarize_schema()
#     print(res)

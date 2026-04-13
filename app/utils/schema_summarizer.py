from app.sampledata.data_loader import DataLoader


class Summarizer:
    @staticmethod
    def summarize_schema() -> str:
        """Return a schema summary enriched with any configured column labels."""
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

                column_summaries.append(column_summary)

            if not column_summaries:
                column_summaries = list(table_info.get("columns", {}).keys())

            primary_keys = table_info.get("primary_key", [])
            foreign_keys = [
                f"{foreign_key['from'][0]}→{foreign_key['to_table']}.{foreign_key['to_columns'][0]}"
                for foreign_key in table_info.get("foreign_keys", [])
            ]

            summary = f"{table_name}({', '.join(column_summaries)})"
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

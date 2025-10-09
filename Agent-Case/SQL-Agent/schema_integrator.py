import sqlite3
import json
from typing import Dict, Any, List


class SQLiteSchemaIngestor:
    def __init__(self, db_path: str, sample_rows: int = 3):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.sample_rows = sample_rows

    def get_tables_and_views(self) -> List[Dict[str, str]]:
        cursor = self.conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%';"
        )
        return [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]

    def get_columns_and_pk(self, table: str):
        cursor = self.conn.execute(f"PRAGMA table_info({table});")
        columns = {}
        pk_columns = []
        for col in cursor.fetchall():
            col_name = col[1]
            columns[col_name] = {
                "type": col[2],
                "notnull": bool(col[3]),
                "default": col[4],
                "is_pk": col[5] > 0
            }
            if col[5] > 0:
                pk_columns.append((col[5], col_name))  # (position, name)

        # Sort composite PK by position
        pk_columns = [name for _, name in sorted(pk_columns, key=lambda x: x[0])]
        return columns, pk_columns

    def get_foreign_keys(self, table: str):
        cursor = self.conn.execute(f"PRAGMA foreign_key_list({table});")
        fks = {}
        for row in cursor.fetchall():
            fid = row[0]  # foreign key id, same id = same composite FK
            if fid not in fks:
                fks[fid] = {
                    "from": [],
                    "to_table": row[2],
                    "to_columns": []
                }
            fks[fid]["from"].append(row[3])
            fks[fid]["to_columns"].append(row[4])
        return list(fks.values())

    def get_indices(self, table: str):
        cursor = self.conn.execute(f"PRAGMA index_list({table});")
        indices = []
        for idx in cursor.fetchall():
            index_name = idx[1]
            unique = bool(idx[2])
            idx_info_cursor = self.conn.execute(f"PRAGMA index_info({index_name});")
            indexed_cols = [col[2] for col in idx_info_cursor.fetchall()]
            indices.append({
                "name": index_name,
                "unique": unique,
                "columns": indexed_cols
            })
        return indices

    def get_sample_rows(self, table: str):
        try:
            cursor = self.conn.execute(f"SELECT * FROM {table} LIMIT {self.sample_rows};")
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.DatabaseError:
            return []  # views without data, etc.

    def extract(self) -> Dict[str, Any]:
        schema = {"tables": {}}
        for entry in self.get_tables_and_views():
            table = entry["name"]
            ttype = entry["type"]

            columns, pk_columns = self.get_columns_and_pk(table)
            fks = self.get_foreign_keys(table)
            indices = self.get_indices(table)
            samples = self.get_sample_rows(table) if ttype == "table" else []

            schema["tables"][table] = {
                "type": ttype,
                "columns": columns,
                "primary_key": pk_columns,
                "foreign_keys": fks,
                "indices": indices,
                "sample_rows": samples
            }
        return schema


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SQLite Schema Ingestor")
    parser.add_argument("db_path", help="Path to the SQLite database file")
    args = parser.parse_args()
    print(args,'---args-----')
    ingestor = SQLiteSchemaIngestor(args.db_path)
    schema = ingestor.extract()
    #print(json.dumps(schema, indent=2))
    with open('sample_data/schema_preserve.json','w') as f:
        json.dump(schema, f, indent=2)

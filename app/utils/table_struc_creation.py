from __future__ import annotations

from app.sampledata.data_loader import DataLoader


class TabStrucCreation:
    def __init__(self, table_list: list[str]):
        self.table_list = table_list

    def struc_creation(self) -> str:
        data = DataLoader.load_data()
        tables = data["tables"]

        qualified_to_key: dict[str, str] = {}
        for key, info in tables.items():
            schema_name = info.get("schema")
            if schema_name:
                qualified_to_key[f"{schema_name}.{key}"] = key

        table_sections: list[str] = []

        for table_name in self.table_list:
            lookup_key = qualified_to_key.get(table_name, table_name)
            table_info = dict(tables[lookup_key])
            schema_name = table_info.get("schema")
            if schema_name:
                table_info["qualified_name"] = f"{schema_name}.{lookup_key}"
            table_payload = {lookup_key: table_info}
            table_sections.append(f"\n{table_payload}")

        return "".join(table_sections)


####### unit test ##########
# if __name__ == '__main__':
#     tab_struc_gen=TabStrucCreation(['Movie', 'Person'])
#     res = tab_struc_gen.struc_creation()
#     print(res)

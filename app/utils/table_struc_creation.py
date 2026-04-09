from __future__ import annotations

from app.sampledata.data_loader import DataLoader


class TabStrucCreation:
    def __init__(self, table_list: list[str]):
        self.table_list = table_list

    def struc_creation(self) -> str:
        data = DataLoader.load_data()
        table_sections: list[str] = []

        for table_name in self.table_list:
            table_payload = {table_name: data["tables"][table_name]}
            table_sections.append(f"\n{table_payload}")

        return "".join(table_sections)


####### unit test ##########
# if __name__ == '__main__':
#     tab_struc_gen=TabStrucCreation(['Movie', 'Person'])
#     res = tab_struc_gen.struc_creation()
#     print(res)

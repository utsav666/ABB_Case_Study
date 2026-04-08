import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sampledata.data_loader import DataLoader


class Summarizer:
    @staticmethod
    def summarize_schema():
        data = DataLoader.load_data()
        summaries = []
        for tname, tinfo in data["tables"].items():
            cols = [c for c in tinfo["columns"].keys()]
            pk = tinfo.get("primary_key", [])
            fks = [
                f"{fk['from'][0]}→{fk['to_table']}.{fk['to_columns'][0]}"
                for fk in tinfo.get("foreign_keys", [])
            ]
            summary = f"{tname}({', '.join(cols)})"
            if pk:
                summary += f" PK: {', '.join(pk)}"
            if fks:
                summary += f" FK: {', '.join(fks)}"
            summaries.append(summary)
        return " ".join(summaries)


##### Unit 3 #######
# if __name__ == "__main__":
#     res=Summarizer.summarize_schema()
#     print(res)

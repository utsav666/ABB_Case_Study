import json

class Summarizer:
    def __init__(self,data_path):
        self.data_path = data_path

    def summarize_schema(self):
        with open(self.data_path) as f:
            data = json.load(f)
        
        summaries = []
        for tname, tinfo in data["tables"].items():
            cols = [c for c in tinfo["columns"].keys()]
            pk = tinfo.get("primary_key", [])
            fks = [
                f'{fk["from"][0]}→{fk["to_table"]}.{fk["to_columns"][0]}'
                for fk in tinfo.get("foreign_keys", [])
            ]
            summary = f"{tname}({', '.join(cols)})"
            if pk: summary += f" PK: {', '.join(pk)}"
            if fks: summary += f" FK: {', '.join(fks)}"
            summaries.append(summary)
        return ' '.join(summaries)
    
# summ = Summarizer('SQL-Agent/sample_data/schema_preserve.json')
# res=summ.summarize_schema()
# print(res)
    
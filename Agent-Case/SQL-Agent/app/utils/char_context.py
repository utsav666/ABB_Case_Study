import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
#from sampledata.data_loader import DataLoader
class ChartContext:
    def build_chart_context(df):
        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "row_count": len(df),
            "sample_rows": df.head(3).to_dict(orient="records")
                }
##### Unit 3 #######
# if __name__ == "__main__":
#     df = pd.DataFrame({
#     "month": ["Jan", "Feb", "Mar", "Apr", "May"],
#     "revenue": [12000, 15000, 14000, 17000, 16000]
#     })
#     char_context =ChartContext.build_chart_context(df)
#     print(char_context)
    
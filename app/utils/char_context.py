class ChartContext:
    def __init__(self):
        pass

    @staticmethod
    def build_chart_context(df):
        if (
            not hasattr(df, "columns")
            or not hasattr(df, "dtypes")
            or not hasattr(df, "head")
        ):
            raise TypeError(
                "ChartContext.build_chart_context expects a pandas DataFrame."
            )

        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "row_count": len(df),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }


##### Unit 3 #######
# if __name__ == "__main__":
#     df = pd.DataFrame({
#     "month": ["Jan", "Feb", "Mar", "Apr", "May"],
#     "revenue": [12000, 15000, 14000, 17000, 16000]
#     })
#     char_context =ChartContext.build_chart_context(df)
#     print(char_context)

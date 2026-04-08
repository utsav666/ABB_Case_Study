import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Union

import pandas as pd
from sampledata.data_loader import DBLoader
from sqlalchemy import create_engine


class SqlExec:
    def __init__(self):
        # self.data_path = 'SQL-Agent/sample_data/database_cred.json'
        self.data_path = DBLoader.db_loader()
        # self.connection = sqlite3.connect(self.data_path)
        # self.sql_query = sql_query

    def sql_executor(self, sql_query) -> Union[pd.DataFrame, str]:
        try:
            db_details = self.data_path["data_base_detail"]
            url = f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            engine = create_engine(url)
            df = pd.read_sql_query(sql_query, engine)
            engine.dispose()
            return df
        except Exception as e:
            return str(e)


# if __name__ == "__main__":
#     user_query = """SELECT * FROM Movie WHERE rating < 3"""
#     sql_exec= SqlExec()
#     res = sql_exec.sql_executor(user_query)
#     print(res)

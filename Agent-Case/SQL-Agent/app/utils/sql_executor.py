import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import sqlite3

import pandas as pd
import psycopg2
from sampledata.data_loader import DBLoader


class SqlExec:
    def __init__(self):
        # self.data_path = 'SQL-Agent/sample_data/database_cred.json'
        self.data_path = DBLoader.db_loader()
        # self.connection = sqlite3.connect(self.data_path)
        # self.sql_query = sql_query

    def sql_executor(self, sql_query):
        try:
            # connection = sqlite3.connect(self.data_path)
            connection = psycopg2.connect(
                host=self.data_path["data_base_detail"]["host"],
                database=self.data_path["data_base_detail"]["database"],
                user=self.data_path["data_base_detail"]["user"],
                password=self.data_path["data_base_detail"]["password"],
                port=self.data_path["data_base_detail"]["port"],
            )
            df = pd.read_sql_query(sql_query, connection)
            connection.close()
            return df
        except Exception as e:
            return str(e)


# if __name__ == "__main__":
#     user_query = """SELECT * FROM Movie WHERE rating < 3"""
#     sql_exec= SqlExec()
#     res = sql_exec.sql_executor(user_query)
#     print(res)

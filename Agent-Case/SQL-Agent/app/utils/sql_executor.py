import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import sqlite3
import pandas as pd


class SqlExec:
    def __init__(self):
        self.data_path = 'SQL-Agent/sample_data/Db-IMDB-Assignment.db'  
        #self.connection = sqlite3.connect(self.data_path)
        #self.sql_query = sql_query

    def sql_executor(self,sql_query):
        try:
            connection = sqlite3.connect(self.data_path)
            df = pd.read_sql_query(sql_query,connection)
            connection.close()
            return df
        except Exception as e:
            return str(e)
# if __name__ == "__main__":
#     user_query = """SELECT * FROM Movie WHERE rating < 3"""
#     sql_exec= SqlExec()
#     res = sql_exec.sql_executor(user_query)
#     print(res.shape)
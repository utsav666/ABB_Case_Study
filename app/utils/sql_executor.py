import logging
from typing import Union

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from app.sampledata.data_loader import DBLoader

logger = logging.getLogger(__name__)


class SqlExec:
    def __init__(self):
        # self.data_path = 'SQL-Agent/sample_data/database_cred.json'
        self.data_path = DBLoader.db_loader()
        # self.connection = sqlite3.connect(self.data_path)
        # self.sql_query = sql_query

    def sql_executor(self, sql_query: str) -> Union[pd.DataFrame, str]:
        try:
            db_details = self.data_path["data_base_detail"]
            # Build connection URL with search_path to include both public and other schemas
            url = f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            engine = create_engine(url)
            df = pd.read_sql_query(sql_query, engine)
            engine.dispose()
            return df
        except (KeyError, SQLAlchemyError, ValueError) as error:
            logger.exception("SQL execution failed")
            return str(error)


# if __name__ == "__main__":
#     user_query = """SELECT * FROM Movie WHERE rating < 3"""
#     sql_exec= SqlExec()
#     res = sql_exec.sql_executor(user_query)
#     print(res)

from __future__ import annotations

import sqlglot
from sqlglot.errors import SqlglotError


def validator(sql_query: str) -> bool | str:
    try:
        sqlglot.parse_one(sql_query)
        return True
    except SqlglotError as error:
        return str(error)


# if __name__ == "__main__":
#     res = validator("""SELECT Person.Name
# FROM Person
# JOIN M_Cast ON Person.PID = M_Cast.PID
# JOIN Movie ON M_Cast.MID = Movie.MID
# WHERE Movie.title = 'Anand'""")
#     print(res)

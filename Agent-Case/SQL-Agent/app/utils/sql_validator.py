import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import sqlglot

def validator(sql_query):
    try:
        tree = sqlglot.parse_one(sql_query)
        return True
    except Exception as e:
        return str(e)
# if __name__ == "__main__":
#     res = validator("""SELECT Person.Name
# FROM Person
# JOIN M_Cast ON Person.PID = M_Cast.PID
# JOIN Movie ON M_Cast.MID = Movie.MID
# WHERE Movie.title = 'Anand'""")
#     print(res)
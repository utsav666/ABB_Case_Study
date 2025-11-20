import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.sql_generation_agent import SQLGenerationAgent
from agent.table_identification_agent import TableIdentiferAgent
from agent.schema_info_agent import SchemaInfoAgent
#from router.route import Router
from router.route_improvisation import Router
from utils.sql_validator import validator
from utils.sql_executor import SqlExec

######Simple Hardcoded Orchetrator########
class Orchestrator:
    def __init__(self,user_question):
        self.user_question = user_question
        self.TA= TableIdentiferAgent(self.user_question)
        self.router = Router()
    def executor(self):
        print("user question .... "+self.user_question)
        print("\n.....Running the Table Identifier Agent ......")
        decision_maker = self.router.route_question(self.user_question)
        print(decision_maker,"decision_maker")
        if decision_maker["route"]== "sql_query":
            ta_agent_result = self.TA.run()
            print("\n....Table identifier result... "+str(ta_agent_result) )
            print("\nRunning SQL Generation Agent")
            sql_gen_agent = SQLGenerationAgent(self.user_question , ta_agent_result['relevant_tables'])
            sql_gen_res = sql_gen_agent.run()
            print("\n....SQL generation Result... "+sql_gen_res)
            sql_val = validator(sql_gen_res)
            if sql_val is True:
                print("\n.........executing query......")
                sql_exec = SqlExec().sql_executor(sql_gen_res)
                #print(sql_exec.shape)
                return sql_exec
            else:
                print("\nFailed to generate sql")
        elif decision_maker["route"] == "schema_info":
            schema_info_agent = SchemaInfoAgent(self.user_question,
                                                decision_maker["reasoning"])
            res = schema_info_agent.run()
            print(res)
            return res
        
#### Unite Test 3 #########    
# if __name__ == "__main__":
#     user_query = "what are the columns of movie table ?"
#     print("..Orchestration Started.....")
#     orch_run = Orchestrator(user_query)
#     orch_run.executor()
#     print("..Orchetsration Completed.....")
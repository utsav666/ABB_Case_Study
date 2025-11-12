import sys, os
import openai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt.prompts import sql_gen_prompts
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from envsetup.env_loader import load_env 
from utils.schema_summarizer import Summarizer
from utils.table_struc_creation import TabStrucCreation
#from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
load_env()
class SQLGenerationAgent:
    def __init__(self,user_question,table_list):
        self.user_question = user_question
        self.tab_struc_gen=TabStrucCreation(table_list).struc_creation()

    def run(self):
        prompt = sql_gen_prompts.format(
            table_structure=self.tab_struc_gen,
            user_question=self.user_question
        )
        response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a world-class SQL generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
        return response.choices[0].message.content.strip()
# if __name__ == "__main__":
#     user_question="list of all actor played in the movie 'Anand'"
#     table_list = ["Movie", "M_Cast", "Person"]
#     sql_gen =  SQLGenerationAgent(user_question,table_list)
#     res=sql_gen.run()
#     print(res)
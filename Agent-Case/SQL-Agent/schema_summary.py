import json
import os
import openai
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.output_parsers import PydanticOutputParser
#from langchain.output_parsers import JsonOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Load variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")

# Use the key with OpenAI
openai.api_key = api_key

# Define LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ---------- Pydantic Schema ----------
class TableIdentification(BaseModel):
    user_question: str
    relevant_tables: List[str]

# Define parser
#parser = PydanticOutputParser(pydantic_object=TableIdentification)
parser = JsonOutputParser()

# ---------- Summarize Schema ----------
def summarize_schema(schema_path):
    with open(schema_path) as f:
        data = json.load(f)
    
    summaries = []
    for tname, tinfo in data["tables"].items():
        cols = [c for c in tinfo["columns"].keys()]
        pk = tinfo.get("primary_key", [])
        fks = [
            f'{fk["from"][0]}→{fk["to_table"]}.{fk["to_columns"][0]}'
            for fk in tinfo.get("foreign_keys", [])
        ]
        summary = f"{tname}({', '.join(cols)})"
        if pk: summary += f" PK: {', '.join(pk)}"
        if fks: summary += f" FK: {', '.join(fks)}"
        summaries.append(summary)
    return summaries
res = "\n".join(summarize_schema("sample_data/schema_preserve.json"))
print(res)
#----------Table structure creation--------
def tab_struc_creation(tab_list):
    with open('sample_data/schema_preserve.json') as f:
        data = json.load(f)
    tab_struc = ''
    for val in tab_list:
        tab_struc = tab_struc+'\n'+str({val:data['tables'][val]})
    return tab_struc
    

#----Table Identification-------
def identify_relevant_tables(parser,res:str,user_question: str, model: str = "gpt-4"):
    """
    Given a schema summary (text) and a natural-language question,
    ask GPT to identify which tables are relevant for generating SQL.
    """
    prompt = ChatPromptTemplate.from_template(f"""
You are an expert data analyst who generates SQL queries from natural language questions.
Below is a summary of the database schema. Each line describes a table and its key relationships.
---
{res}
---
Now, analyze the user's question and list which tables are relevant for answering it.
Only return a JSON list of table names. Do not include explanations.
Please find the relevant table which is required as per user question 
Do no return the table which are irrelevant
User Question: "{user_question}"
    """)
    # Inject format instructions from parser (no need to instruct "return JSON")
    formatted_prompt = prompt.format_messages(
        schema_summary=res,
        user_question=user_question,
        format_instructions=parser.get_format_instructions()
    )

    # Run the model
    response = llm(formatted_prompt)

    # Parse structured output directly into Pydantic object
    parsed = parser.parse(response.content)
    return parsed    

def sql_generator(table_structure: str, user_question: str, model: str = "gpt-4"):
    sql_prompt = f"""You are a world-class SQL engineer.
Your task is to write the perfect SQL query based on the given **table structure (in JSON format)** and **user question**.
Guidelines:
1. Understand the user's question carefully before writing SQL.
2. Use ONLY the tables and columns that appear in the given table structure.
3. DO NOT invent new columns or tables.
4. Keep all column names exactly as they appear in the table structure.
5. Write the most optimized and syntactically correct SQL query possible.
6. Return ONLY the SQL query (no explanations, no markdown, no comments).
------------------    
Table Structure (JSON):
{table_structure}
------------------
User Question:
{user_question}
------------------
Now write the SQL query:
"""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a world-class SQL generator."},
            {"role": "user", "content": sql_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Example usage:
if __name__ == "__main__":
    #user_question = "list of all actor played in the movie 'anand'"
    user_question = '''List all directors who directed 10 movies or more, in descending order of the number of movies they directed. 
    Return the directors' names and the number of movies each of them directed.'''
    #user_question = '''Find all the actors that made more movies with Yash Chopra than any other director.'''
    tables = identify_relevant_tables(parser,res,user_question)
    print("🧩 Relevant tables:", tables)
    # tab_struc = tab_struc_creation(tables)
    # print(tab_struc)
    # result = sql_generator(tab_struc,user_question)
    # print(result,'sql generation')

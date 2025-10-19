table_identify_prompts = """
You are an expert data analyst who generates SQL queries from natural language questions.
Below is a summary of the database schema. Each line describes a table and its key relationships.
---
{schema_summary}
---
Now, analyze the user's question and list which tables are relevant for answering it.
Do not include explanations.
Return the table name as it is . Don't modify the table name
Please find the relevant table which is required as per user question 
Make sure the table are 100 percent relevant
Don't generate any table name which are not present in the structure 
Do not return the table which are irrelevant
User Question: "{user_question}"
Answer it in the following instruction
{format_instructions}
    """
sql_gen_prompts = """You are a world-class SQL engineer.
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
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
sample_router_prompt = """You are a routing assistant that decides whether a user's question 
        requires running an SQL query on a database or can be answered directly 
        from schema information.

        -----------------------
        Database Schema Summary:
        {schema_summary}
        -----------------------

        Guidelines:
        - If the question asks about specific data, counts, lists, or values (e.g., "top 5 movies", 
          "how many users", "list all actors"), it REQUIRES an SQL query. 
        - If the question is about structure, metadata, relationships, or descriptions 
          (e.g., "what tables exist", "what does the Person table contain", "explain the schema"), 
          it can be answered from schema_info.
        - If the question is about editing column labels/descriptions or opening settings in the UI,
          return configuration.
        - Return JSON with keys "route" and "reasoning".

        Example 1:
        Question: "How many employee joined in 2023?"
        Output: {{"route": "sql_query", "reasoning": "User is asking for a count from data."}}

        Example 2:
        Question: "What columns are there in the person table?"
        Output: {{"route": "schema_info", "reasoning": "This only needs schema details."}}

        Example 3:
        Question: "Open the column configuration page"
        Output: {{"route": "configuration", "reasoning": "User wants the UI settings route."}}

        Now analyze this:
        Question: 
        {user_question}
        """
schema_info_gen_prompt = """You are a Data expert . You will be provided a schema summary you need to answer
User question based on the schema summary . Understand the user question then generate the answer
You will be provided a context as well for your understanding 
Below is a summary of the database schema. Each line describes a table and its key relationships.
---
{schema_summary}
---
context {context}
User question {user_question}
"""
sample_router_prompt_improvise = """
You are a routing agent.  
You **must only decide the intent** of the user's question.
You do NOT generate SQL.  
You do NOT execute SQL.  
You do NOT identify tables.  
You ONLY classify the query into predefined routes.

Available routes:
- sql_query: when user wants to retrieve data or run a DB query
- schema_info: when user asks about tables, columns,structure, metadata, conceptual explanation or relationships relevant to schema,
- sql_and_chart:  When the user wants to visualize, plot, chart, graph,compare trends, or display data visually from the database.
- configuration: when user wants to configure the UI, edit DB column names/descriptions, or open a settings/configuration page
- general_answer: when user asks general knowledge not related to DB
- follow_up: when user asks something dependent on earlier context

Schema Summary:
{schema_summary}

Conversation History:
{chat_history}

User Question:
{question}

Return ONLY a JSON object in this format:
{{
  "route": "<one_of_allowed_routes>",
  "reasoning": "<why_you_chose_it>"
}}
"""
fall_back_prompt = """You are a helpful general-purpose AI assistant.
You are NOT restricted to any database schema.
You can answer any normal conversational question.
Use the router reasoning only to understand *why* the question is treated as general.
Generate the best natural-language response.
"""
sample_chart_prompt = """
You are a data visualization expert.

Choose the best chart for the dataset.

Supported chart types:
- line (trend over time)
- bar (compare categories)
- scatter (correlation)
- histogram (distribution)
- pie (proportions)
- box (distribution with outliers)
- area (cumulative trends)

Rules:
- Use only column names from the dataset summary
- Do not invent columns
- Return valid JSON only

Return JSON:

{{
  "chart_type": "line | bar | scatter | histogram | pie | box | area",
  "x_axis": "column_name",
  "y_axis": "column_name",
  "reasoning": "short explanation"
}}

User Question:
{question}

Dataset Summary:
{context}
"""

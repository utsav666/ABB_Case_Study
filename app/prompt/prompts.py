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

IMPORTANT:
- The SQL dialect is {dialect}.
- Use ONLY syntax compatible with this dialect.
- Do NOT mix SQL dialects.

Guidelines:
1. Understand the user's question carefully before writing SQL.
2. Use ONLY the tables and columns that appear in the given table structure.
3. DO NOT invent new columns or tables.
4. Keep all column names exactly as they appear in the table structure.
5. Write the most optimized and syntactically correct SQL query possible.
6. Return ONLY the SQL query.
7. If a table has "schema" and "qualified_name" fields, use the qualified_name (schema.table) in the FROM/JOIN clauses. Do NOT wrap the qualified name in double quotes as a single identifier.
8. When the user asks for data relative to a time period (e.g. "last 3 years"), do NOT use CURRENT_DATE. Instead, use a subquery to find the MAX date from the table and calculate the range relative to that. For example: WHERE date >= ((SELECT MAX(date) FROM schema.table) - INTERVAL '3 years').

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
- sql_query: when user wants simple data retrieval (single metric, no comparison needed)
- schema_info: when user asks about tables, columns, structure, metadata, conceptual explanation or relationships relevant to schema
- sql_and_chart: when user wants to visualize, plot, chart, graph, compare, show trends, or display data visually
  **IMPORTANT**: Use sql_and_chart when the query involves:
  - Multiple metrics (e.g., "sales AND quantity", "votes AND rating")
  - Rankings with comparisons (e.g., "top 5 by X and Y")
  - Trend analysis (e.g., "over time", "by year")
  - Explicit visualization words: chart, graph, plot, visualize, show, compare
  **Default to sql_and_chart for any query with 2+ numeric metrics**
- configuration: when user wants to configure the UI, edit DB column names/descriptions, or open a settings/configuration page
- general_answer: when user asks general knowledge not related to DB
- follow_up: when user asks something dependent on earlier context

Examples:

Question: "Top 5 products by sales and quantity"
→ {{"route": "sql_and_chart", "reasoning": "Query involves 2 metrics (sales AND quantity) for comparison - needs visualization"}}

Question: "Show movies by votes and rating"
→ {{"route": "sql_and_chart", "reasoning": "Multiple metrics (votes AND rating) - best shown as dual-axis chart"}}

Question: "List employees in engineering department"
→ {{"route": "sql_query", "reasoning": "Simple data retrieval, single dimension, no metrics to compare"}}

Question: "Compare revenue vs profit by region"
→ {{"route": "sql_and_chart", "reasoning": "Explicit comparison of 2 metrics - needs visualization"}}

Question: "What columns are in the sales table?"
→ {{"route": "schema_info", "reasoning": "Asking about table structure, not data"}}

---

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

**Remember**: If the question mentions 2+ metrics (e.g., "X and Y", "X vs Y") → use sql_and_chart
"""
fall_back_prompt = """You are a helpful general-purpose AI assistant.
You are NOT restricted to any database schema.
You can answer any normal conversational question.
Use the router reasoning only to understand *why* the question is treated as general.
Generate the best natural-language response.
"""
# sample_chart_prompt = """
# You are a data visualization expert.

# Choose the best chart for the dataset.

# Supported chart types:
# - line (trend over time)
# - bar (compare categories)
# - scatter (correlation)
# - histogram (distribution)
# - pie (proportions)
# - box (distribution with outliers)
# - area (cumulative trends)

# Rules:
# - Use only column names from the dataset summary
# - Do not invent columns
# - Return valid JSON only

# Return JSON:

# {{
#   "chart_type": "line | bar | scatter | histogram | pie | box | area",
#   "x_axis": "column_name",
#   "y_axis": "column_name",
#   "reasoning": "short explanation"
# }}

# User Question:
# {question}

# Dataset Summary:
# {context}
# """
sample_chart_prompt = """
You are an expert data visualization assistant specializing in choosing the BEST chart type for business intelligence dashboards.

Your task: analyze the user’s question and dataset, then select the most appropriate and INTELLIGENT chart type.

---

## 🎯 Step 1: Count Numeric Metrics in Dataset

**If dataset has 1 numeric column:**
- Use: bar, line, pie, histogram (simple charts)

**If dataset has 2 numeric columns with DIFFERENT SCALES:**
- Example: votes (500K) + rating (7.8) → scales differ by 60,000x
- **Use dual_axis_column or column_line** to show both metrics clearly
- NEVER use simple "bar" when scales differ drastically

**If dataset has 2+ numeric columns with SIMILAR SCALES:**
- Use: grouped_column or stacked_column

---

## 🧠 Step 2: Match Intent to Chart Type

### 🔥 PRIORITY: Detect Question Intent First

**If question asks about RELATIONSHIP/CORRELATION:**
- Keywords: "relationship", "correlation", "between X and Y", "how X relates to Y", "X vs Y", "X against Y"
- **ALWAYS use scatter** chart with both metrics on X and Y axes
- Example: "relationship between sales and quantity" → scatter (sales on X, quantity on Y)
- Example: "how price correlates with demand" → scatter (price on X, demand on Y)

### Single-Metric Charts:
- **Comparison** → bar (e.g., "top 5 movies by votes")
- **Trend over time** → line or area
- **Distribution** → histogram
- **Proportion** → pie

### Multi-Metric Charts (2+ numeric columns):
- **If asking about relationship/correlation** → **scatter** (prioritize this!)

- **Two metrics, different scales** → **dual_axis_column** or **column_line**
  - dual_axis_column: both as bars, separate Y-axes
  - column_line: primary as bars, secondary as line overlay
  - Example: "votes and rating" → votes (bars, left axis) + rating (line, right axis)

- **Two metrics, similar scales** → grouped_column or stacked_column
  - grouped_column: bars side-by-side
  - stacked_column: cumulative stacking

- **Multiple series** → grouped_column or line (if time-based)

---

## 🔍 Step 3: Detect Scale Differences

**Check if numeric columns differ by >10x:**
- If column A max / column B max > 10 → **USE dual_axis or column_line**
- Example: num_votes (max 500K) / rating (max 8.5) = 58,000x → MUST use dual-axis

**If scales are similar (<10x difference):**
- Use grouped_column or stacked_column

---

## ⚠️ Critical Rules

1. **NEVER use simple "bar" when dataset has 2+ numeric columns** — you’ll hide one metric
2. **ALWAYS check scale differences** — if >10x, use dual_axis_column or column_line
3. Use ONLY columns present in the dataset
4. Output MUST be valid JSON (no extra text)
5. Keep reasoning concise (1 sentence)

---

## ✅ Output Format

### For single-metric charts:
{{
  "chart_type": "bar | line | scatter | histogram | pie | box | area",
  "x_axis": "column_name",
  "y_axis": "column_name",
  "reasoning": "brief explanation"
}}

### For dual-axis charts:
{{
  "chart_type": "dual_axis_column | column_line",
  "x_axis": "column_name",
  "y_axis": "primary_metric",
  "y_axis_secondary": "secondary_metric",
  "reasoning": "e.g., votes and rating have vastly different scales (500K vs 8)"
}}

### For multi-series charts:
{{
  "chart_type": "grouped_column | stacked_column",
  "x_axis": "column_name",
  "y_axis": ["metric1", "metric2"],
  "reasoning": "brief explanation"
}}

---

## 📌 User Question:
{question}

## 📊 Dataset Summary:
{context}

**Remember:** If 2+ numeric columns exist, analyze their scales FIRST. If they differ by >10x, you MUST use dual_axis_column or column_line.
"""

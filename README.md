# Data Model Insight Accelerator

The **Data Model Insight Accelerator** is an intelligent query-generation and data-understanding framework that transforms natural-language questions into optimized database queries. It leverages schema-aware reasoning, multi-layered intent understanding, and LLM fallback to deliver accurate and meaningful insights from your database.

---

## Key Features

---

### 1. Schema Knowledge Base

- Stores database metadata (tables, columns, relationships, constraints).  
- Enhances contextual understanding and improves query accuracy.  
- Allows the system to reason using actual database structures instead of pure text predictions.  

---

### 2. Dual Reasoning Layers

#### a. Intent Reasoning Layer
Interprets what the user *means* and decides whether the request requires:

- SQL generation  
- Insight retrieval  
- Data explanation  
- LLM fallback for general or non-database questions  

#### b. Query Reasoning Layer

- Converts intent into structured query actions  
- Generates optimal SQL using schema information  
- Applies query optimization strategies  
- Ensures accuracy, correct joins, and valid filter logic  

---

### 3. LLM Fallback

If the user question is **too general**, **not database-specific**, or **non-technical**, the system delegates to an LLM to produce a helpful conversational response.

---

### 4. Execution + Post-Processing

- Executes the generated SQL on the database  
- Enhances results with explanations and human-readable insights  
- Returns a clean, structured final response to the user  

---

## How It Works

1. User asks a question (e.g., *“What are the top 5 customers by revenue?”*).  
2. **Intent Reasoning Layer** determines whether it is database-specific.  
3. The system queries the **Schema Knowledge Base** to understand tables and relationships.  
4. The **Query Reasoning Layer** generates and optimizes SQL.  
5. The SQL is executed on the target database.  
6. Results are enriched with insights and returned to the user.  
7. For general questions (e.g., *“Explain regression vs classification”*), the **LLM Fallback** generates the answer instead.

---

## Backend Request Workflow (Exact Agents + Models)

This section documents the runtime flow for `POST /chat` in the FastAPI backend,
including the exact orchestrator node path, agent module, and model used.

### Entry Point

- API endpoint: `app/main.py` -> `@app.post("/chat")`
- Orchestrator: `app/orchestrator/orch_graph.py` -> `GraphOrchestrator`
- Input payload fields used: `question` (or `message/query/prompt/text`), optional `connection_id`, and optional `display` (`"table"` or `"chart"`) to override the automatic display

### End-to-End Flow Chart

```mermaid
flowchart TD
  A[Client POST /chat] --> B[app/main.py: chat()]
  B --> C[GraphOrchestrator.executor()]
  C --> D[router_node]

  D -->|sql_query or follow_up| E[table_resolver_node]
  E --> F[generate_sql_node]
  F --> G[cardinality_resolver_node]
  G --> H[validate_sql_node]
  H -->|valid| I[execute_sql_node]
  H -->|invalid + retries left| F
  H -->|invalid + max retries| N[build_data_response_node]
  I -->|success| J[verify_result_node]
  I -->|error + retries left| F
  I -->|error + max retries| N
  J -->|0 rows or mixed grain, once| F
  J -->|display = chart| K[chart_approval_node]
  J -->|display = table or value| N
  K -->|chart_type none| N
  K --> Z[final_response]
  N --> Z

  D -->|schema_info| L[schema_info_node]
  D -->|general_answer| M[general_answer_node]
  D -->|ga_query| O[ga_agent_node]

  L --> Z
  M --> Z
  O --> Z
  Z --> P[app/main.py: build_chat_response() -> JSON]
```

### Agent and Model Map

| Stage | Orchestrator Node | Agent Module | Model | Notes |
|---|---|---|---|---|
| Route intent | `router_node` | `app/router/route_improvisation.py` (`Router`) | `gpt-4o` | Chooses route (SQL, follow-up, explanation, schema, general) and display intent (chart, table, value, auto) from the question and chat history |
| Identify tables | `table_resolver_node` | `app/agent/table_identification_agent.py` (`identify_tables`) | `gpt-4` | Skipped when only one table is available |
| Generate SQL | `generate_sql_node` | `app/agent/sql_generation_agent.py` (`generate_sql`) | `SQL_GENERATION_MODEL` env, default `gpt-4` | Returns SQL plus token usage and estimated cost |
| Resolve ambiguous filter values | `cardinality_resolver_node` | `app/agent/value_resolver_agent.py` (`resolve_filter_value`) | `gpt-4o` | Triggers HITL when confidence is low |
| Validate SQL | `validate_sql_node` | `app/utils/sql_validator.py` | No LLM | Exactly one read-only query; writes, DDL and data-modifying CTEs are rejected |
| Execute SQL | `execute_sql_node` | `app/utils/sql_executor.py` (`SqlExec`) | No LLM | Read-only transaction with `DB_STATEMENT_TIMEOUT_MS` (default 8000) on PostgreSQL |
| Check result + choose display | `verify_result_node` | `app/orchestrator/display_policy.py` | No LLM | Regenerates once on 0 rows or repeated labels; picks chart/table/value from intent + data shape |
| Schema Q&A | `schema_info_node` | `app/agent/schema_info_agent.py` (`explain_schema`) | `gpt-4` | Used when route is `schema_info` |
| General fallback | `general_answer_node` | `app/agent/fall_back.py` (`general_answer`) | `gpt-4o` | Non-database questions |
| Chart decision | `chart_approval_node` | `app/agent/chart_gen_agent.py` (`decide_chart`) | `gpt-4o` | Recommends chart config, or `none` to fall back to a table |
| GA report spec | `ga_agent_node` | `app/agent/ga_query_agent.py` (`generate_ga_report_spec`) | `gpt-4o` | Only for Google Analytics connections |

### Data Objects Passed Between Nodes

- `GraphState` keys used in flow:
  - `question`, `connection_id`, `route`, `tables`, `sql`, `sql_error`, `retry_count`, `data`, `final_response`
- SQL path state transitions:
  - question -> tables -> sql -> validated sql -> query data -> final response
- If `connection_id` is provided:
  - orchestrator loads stored schema graph and uses it for routing/table selection/SQL generation context

### Retry and HITL Behavior

- Retry limit: `MAX_RETRIES = 2` in `app/orchestrator/orch_graph.py`
- SQL retry happens after validation or execution errors, and once more when the result is empty or repeats the same labels with different numbers (mixed versions/scenarios)
- HITL points:
  - value resolution ambiguity in `cardinality_resolver_node`
  - chart approval path in `chart_approval_node`
- Resume endpoint for HITL: `POST /chat/resume`


### Semantic Profile

The schema graph describes structure; the semantic profile describes what the
data contains, so SQL uses real filter values instead of guesses. It lives in
`app/semantic_catalog/connection_profile.py` and is stored inside the
connection's graph (`settings.schema_graphs.graph.profile`).

- **Facts** (two batched queries per table): row counts, distinct counts,
  min/max, every value of categorical columns, and values that differ only by
  case (e.g. `Granted`/`granted`) with the column that separates them.
- **Inferred semantics** (one LLM call per connection, validated against the
  facts): each table's grain, metric-as-row columns, scenario columns with
  their base value, and version columns with the column that orders them.
- Built when tables are selected (column sync), and on demand with
  `POST /settings/db-connections/{id}/semantic-profile/refresh`. If missing at
  chat time, facts are collected and stored automatically (no LLM). View it with
  `GET /settings/db-connections/{id}/semantic-profile`.
- The SQL generator adds a `-- Assumptions:` comment (defaults it applied); the
  answer shows it to the user.
- Tunables: `PROFILER_VALUE_STORE_LIMIT`, `PROFILER_PROMPT_VALUE_LIMIT`,
  `PROFILER_QUERY_TIMEOUT_MS`, `SEMANTIC_PROFILE_MODEL`.
- Profiling currently supports PostgreSQL targets; other engines fall back to
  name-based value discovery.

### Golden-question evaluation

`scripts/eval_questions.py` runs questions through the full orchestrator
against a connection (read-only) and checks the answers, not the SQL text:

```bash
python -m scripts.eval_questions eval/cap_table_questions.json --connection-id <uuid>
```

The exit code is the number of failed questions.


# Basic Flow Diagram
![Image 1](DMIA_flow_diagram.png)

# How to run
keep your open ai key in the `.env` file inside `app/config`.

Then run this command from the project root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive FastAPI docs.

---

## AWS Lambda container deployment

This project now includes a Lambda-ready container entrypoint using `Mangum`.

### 1. Configure environment variables
Set these in the Lambda function configuration or in your IaC stack:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`
- `ALLOWED_ORIGINS`
- `FASTAPI_ROOT_PATH` (optional, only if your API is mounted behind a stage path)

> Do not bake `app/config/.env` into the production image.

### 2. Build the container image locally

```bash
docker build -t dmia-lambda .
```

### 3. Test the Lambda container locally

```bash
docker run --rm -p 9000:8080 --env-file app/config/.env dmia-lambda
```

In another terminal:

```bash
curl http://localhost:9000/2015-03-31/functions/function/invocations \
  -d '{
    "version": "2.0",
    "routeKey": "GET /health",
    "rawPath": "/health",
    "rawQueryString": "",
    "headers": {"host": "localhost"},
    "requestContext": {"http": {"method": "GET", "path": "/health", "sourceIp": "127.0.0.1", "userAgent": "curl"}},
    "isBase64Encoded": false
  }'
```

### 4. Push to Amazon ECR and deploy to Lambda

```bash
aws ecr create-repository --repository-name dmia-lambda
aws ecr get-login-password --region <aws-region> \
  | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<aws-region>.amazonaws.com

docker tag dmia-lambda:latest <account-id>.dkr.ecr.<aws-region>.amazonaws.com/dmia-lambda:latest
docker push <account-id>.dkr.ecr.<aws-region>.amazonaws.com/dmia-lambda:latest

aws lambda create-function \
  --function-name dmia-api \
  --package-type Image \
  --code ImageUri=<account-id>.dkr.ecr.<aws-region>.amazonaws.com/dmia-lambda:latest \
  --role <lambda-execution-role-arn> \
  --timeout 120 \
  --memory-size 1024
```

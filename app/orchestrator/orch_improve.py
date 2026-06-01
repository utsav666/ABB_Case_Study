"""
Agent Orchestrator — LLM decides the tools to call at each step.
Same structure as orch_exec.py but the LLM drives the flow dynamically.
"""

import json
import logging

import matplotlib
import openai
from pandas import DataFrame

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.agent.chart_gen_agent import decide_chart
from app.agent.fall_back import general_answer
from app.agent.schema_info_agent import explain_schema
from app.agent.sql_generation_agent import generate_sql
from app.agent.table_identification_agent import identify_tables
from app.envsetup.env_loader import get_openai_api_key, load_env
from app.prompt.agent_system_prompt import AGENT_SYSTEM_PROMPT
from app.router.route_improvisation import Router
from app.utils.char_context import ChartContext
from app.utils.sql_executor import SqlExec
from app.utils.sql_validator import validator
from app.utils.schema_summarizer import Summarizer

logger = logging.getLogger(__name__)
load_env()

# ---- Tools the LLM can call ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "identify_tables",
            "description": "Find relevant database tables for a question. Call this first before generating SQL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user question"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sql",
            "description": "Generate SQL from a question + table list. Pass error_context if retrying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "tables": {"type": "array", "items": {"type": "string"}},
                    "error_context": {
                        "type": "string",
                        "description": "Previous error if retrying",
                    },
                },
                "required": ["question", "tables"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Validate and run a SQL query. Returns data or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to run"}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_chart",
            "description": "Create a chart from the last query results. Call after execute_sql.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Original question"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_schema",
            "description": "Answer questions about database structure, tables, columns.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_answer",
            "description": "Answer general questions not related to the database.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]


class AgentOrchestrator:
    """LLM-driven agent. Same interface as Orchestrator in orch_exec.py."""

    MAX_STEPS = 8

    def __init__(self, user_question: str):
        self.user_question = user_question
        self.client = openai.OpenAI(api_key=get_openai_api_key())
        self.schema_summary = Summarizer.summarize_schema()
        self.router = Router()
        # State
        self._dataframe: DataFrame | None = None
        self._sql: str = ""
        self._chart_fig = None

    def executor(self):
        """Run agent loop — same method name as orch_exec.py."""

        # Router hint — gives the agent a head start (agent can override)
        route_decision = self.router.route_question(self.user_question)
        router_hint = (
            f"Router suggests route='{route_decision.route}' "
            f"(reasoning: {route_decision.reasoning}). "
            f"Use this as guidance but you can override if needed."
        )
        logger.info(
            "Router hint: %s → %s", route_decision.route, route_decision.reasoning
        )

        messages = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT.format(
                    schema_summary=self.schema_summary
                ),
            },
            {"role": "system", "content": router_hint},
            {"role": "user", "content": self.user_question},
        ]

        for step in range(self.MAX_STEPS):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message

            # No tool calls → LLM is done, return final answer
            if not msg.tool_calls:
                return self._build_result(msg.content or "")

            # Process tool calls
            messages.append(msg)
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                logger.info("Agent → %s(%s)", tool_name, list(tool_args.keys()))

                result = self._run_tool(tool_name, tool_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        return {"type": "error", "message": "Max steps reached."}

    # ---- Tool execution ----

    def _run_tool(self, name: str, args: dict) -> dict:
        """Run a tool by name. Returns dict with success/error."""

        if name == "identify_tables":
            return identify_tables(args["question"])

        elif name == "generate_sql":
            return generate_sql(
                args["question"],
                args["tables"],
                error_context=args.get("error_context", ""),
            )

        elif name == "execute_sql":
            sql = args["sql"]
            val = validator(sql)
            if val is not True:
                return {"success": False, "error": f"Syntax error: {val}"}
            result = SqlExec().sql_executor(sql)
            if isinstance(result, DataFrame):
                self._dataframe = result
                self._sql = sql
                return {
                    "success": True,
                    "rows": len(result),
                    "columns": list(result.columns),
                    "sample": result.head(5).to_dict(orient="records"),
                }
            return {"success": False, "error": str(result)}

        elif name == "render_chart":
            if self._dataframe is None:
                return {"success": False, "error": "No data. Call execute_sql first."}
            ctx = ChartContext.build_chart_context(self._dataframe)
            res = decide_chart(args["question"], ctx)
            if res["success"]:
                self._chart_fig = self._make_chart(self._dataframe, res["chart"])
                return {"success": True, "chart_type": res["chart"].chart_type}
            return res

        elif name == "explain_schema":
            return explain_schema(args["question"])

        elif name == "general_answer":
            return general_answer(args["question"])

        return {"success": False, "error": f"Unknown tool: {name}"}

    # ---- Helpers ----

    def _build_result(self, message: str) -> dict:
        """Build final response dict."""
        if self._chart_fig:
            return {
                "type": "chart",
                "figure": self._chart_fig,
                "data": self._dataframe,
                "sql": self._sql,
                "message": message,
            }
        elif self._dataframe is not None:
            return {
                "type": "data",
                "data": self._dataframe,
                "sql": self._sql,
                "message": message,
            }
        else:
            return {"type": "text", "message": message}

    def _make_chart(self, df: DataFrame, decision):
        """Render matplotlib chart."""
        if decision.x_axis not in df.columns or decision.y_axis not in df.columns:
            return None
        fig, ax = plt.subplots()
        ct = decision.chart_type
        x, y = df[decision.x_axis], df[decision.y_axis]
        if ct == "line":
            ax.plot(x, y)
        elif ct == "bar":
            ax.bar(x, y)
        elif ct == "scatter":
            ax.scatter(x, y)
        elif ct == "histogram":
            ax.hist(x, bins=10)
        elif ct == "pie":
            ax.pie(y, labels=x.astype(str).tolist(), autopct="%1.1f%%")
        elif ct == "box":
            ax.boxplot(y)
        elif ct == "area":
            ax.fill_between(x, y)
        ax.set_xlabel(decision.x_axis)
        ax.set_ylabel(decision.y_axis)
        return fig


# ====== Unit Test ======
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("=" * 50)
    print("TEST: SQL query")
    print("=" * 50)
    r = AgentOrchestrator("list the total unique categories ").executor()
    print(f"Type: {r['type']}, Message: {r.get('message', '')[:100]}")

    print("\n" + "=" * 50)
    print("TEST: Schema question")
    print("=" * 50)
    r = AgentOrchestrator("what tables are in the database?").executor()
    print(f"Type: {r['type']}, Message: {r.get('message', '')[:100]}")

    print("\n" + "=" * 50)
    print("TEST: Chart")
    print("=" * 50)
    r = AgentOrchestrator("show year wise movie count as bar chart").executor()
    print(f"Type: {r['type']}, Message: {r.get('message', '')[:100]}")

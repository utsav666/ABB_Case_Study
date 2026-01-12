import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Literal

from envsetup.env_loader import load_env
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt.prompts import sample_router_prompt_improvise
from pydantic import BaseModel, Field
from utils.schema_summarizer import Summarizer

# ------------------------
# Schema-aware Router Model
# ------------------------
load_env()


# ----------------------------
# Output Model
# ----------------------------
class RoutingDecision(BaseModel):
    route: Literal[
        "sql_query", "schema_info", "general_answer", "explanation", "follow_up"
    ]
    reasoning: str


# ----------------------------
# Router Agent (Decision Only)
# ----------------------------
class Router:
    def __init__(self):
        self.schema_summary = Summarizer.summarize_schema()

        # LLM
        # Ensure environment is loaded (safe no-op if already loaded)
        try:
            load_env()
        except Exception:
            # If loading fails here, let getenv below provide a clear error
            pass

        # Accept several common env names and sanitize surrounding quotes
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("OPENAI_APIKEY")
            or os.getenv("OPENAI")
        )
        # If dotenv produced an empty string or didn't parse the key, try a manual fallback
        if api_key:
            api_key = api_key.strip().strip("'\"")
        else:
            # Attempt to read the config .env manually and extract the key
            try:
                env_path = os.path.join(
                    os.path.dirname(__file__), "..", "config", ".env"
                )
                env_path = os.path.abspath(env_path)
                if os.path.exists(env_path):
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line or "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            if k.strip() == "OPENAI_API_KEY":
                                candidate = v.strip().strip().strip("'\"\r\n")
                                if candidate:
                                    api_key = candidate
                                    os.environ["OPENAI_API_KEY"] = api_key
                                    break
            except Exception:
                api_key = None

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. Add it to app/config/.env or export OPENAI_API_KEY."
                )

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

        # Output parser
        self.parser = JsonOutputParser(pydantic_object=RoutingDecision)

        # Prompt template
        self.prompt = ChatPromptTemplate.from_template(sample_router_prompt_improvise)

    # ----------------------------
    # PUBLIC METHOD
    # ----------------------------
    def route_question(self, question: str) -> RoutingDecision:
        # Prepare full prompt
        messages = self.prompt.format_messages(
            schema_summary=self.schema_summary, chat_history="", question=question
        )

        # LLM call
        response = self.llm.invoke(messages)

        # Parse into RoutingDecision
        return self.parser.parse(response.content)


# if __name__ == "__main__":
#     user_question = "Write SQL to find items with low stock from an Inventory table.?"
#     route =Router()
#     response = route.route_question(user_question)
#     print(response)

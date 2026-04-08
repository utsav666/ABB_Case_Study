import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Literal

from envsetup.env_loader import get_openai_api_key, load_env
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt.prompts import sample_router_prompt_improvise
from pydantic import BaseModel, SecretStr
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
        "sql_query",
        "sql_and_chart",
        "schema_info",
        "general_answer",
        "explanation",
        "follow_up",
    ]
    reasoning: str


# ----------------------------
# Router Agent (Decision Only)
# ----------------------------
class Router:
    def __init__(self):
        self.schema_summary = Summarizer.summarize_schema()

        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=SecretStr(get_openai_api_key()),
        )

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
        parsed = self.parser.parse(str(response.content))
        return RoutingDecision(**parsed)


# if __name__ == "__main__":
#     user_question = "can you show me year wise movie count ?"
#     route =Router()
#     response = route.route_question(user_question)
#     print(response)

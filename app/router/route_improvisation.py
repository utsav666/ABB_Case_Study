from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.envsetup.env_loader import get_openai_api_key, load_env
from app.prompt.prompts import sample_router_prompt_improvise
from app.utils.schema_summarizer import Summarizer

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
        "configuration",
        "general_answer",
        "explanation",
        "follow_up",
        "ga_query",
    ]
    reasoning: str


# ----------------------------
# Router Agent (Decision Only)
# ----------------------------
class Router:
    def __init__(self, schema_summary: str | None = None):
        # Allow callers to inject a connection-scoped schema summary; fall back
        # to the default (sample/config) summary when none is provided.
        self.schema_summary = (
            schema_summary
            if schema_summary is not None
            else Summarizer.summarize_schema()
        )

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
    def route_question(self, question: str) -> tuple[RoutingDecision, any]:
        """Route a question and return both decision and response for tracking.

        Returns:
            Tuple of (RoutingDecision, response) where response contains usage metadata
        """
        # Prepare full prompt
        messages = self.prompt.format_messages(
            schema_summary=self.schema_summary, chat_history="", question=question
        )

        # LLM call
        response = self.llm.invoke(messages)

        # Parse into RoutingDecision
        parsed = self.parser.parse(str(response.content))
        return RoutingDecision(**parsed), response


# if __name__ == "__main__":
#     user_question = "can you show me year wise movie count ?"
#     route =Router()
#     response = route.route_question(user_question)
#     print(response)

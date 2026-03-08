import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt.prompts import sample_router_prompt_improvise
from envsetup.env_loader import load_env
from utils.schema_summarizer import Summarizer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal


# ------------------------
# Schema-aware Router Model
# ------------------------
load_env()
# ----------------------------
# Output Model
# ----------------------------
class RoutingDecision(BaseModel):
    route: Literal["sql_query","sql_and_chart" ,"schema_info", "general_answer", "explanation", "follow_up"]
    reasoning: str

# ----------------------------
# Router Agent (Decision Only)
# ----------------------------
class Router:
    def __init__(self):

        self.schema_summary = Summarizer.summarize_schema()

        # LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
            schema_summary=self.schema_summary,
            chat_history="",
            question=question
        )

        # LLM call
        response = self.llm.invoke(messages)

        # Parse into RoutingDecision
        return self.parser.parse(response.content)

# if __name__ == "__main__":
#     user_question = "can you show me year wise movie count ?"
#     route =Router()
#     response = route.route_question(user_question)
#     print(response)
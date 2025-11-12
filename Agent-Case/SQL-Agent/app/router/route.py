import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt.prompts import sample_router_prompt
from envsetup.env_loader import load_env
from utils.schema_summarizer import Summarizer
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

# ------------------------
# Schema-aware Router Model
# ------------------------
load_env()
class RoutingDecision(BaseModel):
    route: Literal["sql_query", "schema_info"] = Field(
        description="Decide whether to run an SQL query or answer directly from schema info."
    )
    reasoning: str = Field(description="Short reasoning why this route was chosen.")

class Router:
    def __init__(self, schema_summary: str = None):
        self.schema_summary = Summarizer.summarize_schema() or "No schema summary provided."
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def route_question(self, user_question: str):
        """
        Route a user question to either SQL or schema reasoning.
        Returns a RoutingDecision object.
        """
        router_prompt = ChatPromptTemplate.from_template(sample_router_prompt)
        

        parser = JsonOutputParser(pydantic_object=RoutingDecision)
        formatted_prompt = router_prompt.format_messages(
            schema_summary=self.schema_summary,
            user_question=user_question
        )

        response = self.llm(formatted_prompt)
        decision = parser.parse(response.content)
        return decision


# ------------------------
# Example Usage
# ------------------------
# if __name__ == "__main__":
#     # schema_summary = """
#     # Tables: Movie(id, title, year, rating), Person(id, name, role), M_Cast(movie_id, person_id)
#     # Relationships: M_Cast links movies and persons.
#     # """
#     router = Router()

#     examples = [
#         "List all actors who acted in 'Inception'",
#         "What are the tables available in the database?",
#         "Explain how Movie and Person tables are related"
#     ]

#     for q in examples:
#         result = router.route_question(q)
#         print(result,type(result))
#         #print(f"\nQuestion: {q}\nRoute: {result.route}\nReasoning: {result.reasoning}")

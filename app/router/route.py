from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.envsetup.env_loader import load_env
from app.prompt.prompts import sample_router_prompt
from app.utils.schema_summarizer import Summarizer

# ------------------------
# Schema-aware Router Model
# ------------------------
load_env()


class RoutingDecision(BaseModel):
    route: Literal["sql_query", "schema_info", "configuration"] = Field(
        description=(
            "Decide whether to run an SQL query, answer from schema info, "
            "or direct the user to the UI configuration page."
        )
    )
    reasoning: str = Field(description="Short reasoning why this route was chosen.")


class Router:
    def __init__(self, schema_summary: str | None = None):
        self.schema_summary = (
            Summarizer.summarize_schema() or "No schema summary provided."
        )
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def route_question(self, user_question: str):
        """
        Route a user question to either SQL or schema reasoning.
        Returns a RoutingDecision object.
        """
        router_prompt = ChatPromptTemplate.from_template(sample_router_prompt)

        parser = JsonOutputParser(pydantic_object=RoutingDecision)
        formatted_prompt = router_prompt.format_messages(
            schema_summary=self.schema_summary, user_question=user_question
        )

        response = self.llm.invoke(formatted_prompt)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        decision = parser.parse(content)
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

from __future__ import annotations

from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, model_validator

from app.envsetup.env_loader import load_env
from app.prompt.prompts import sample_router_prompt_improvise
from app.utils.llm_factory import create_chat_llm

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
    ]
    display: Literal["chart", "table", "value", "auto"] = "auto"
    reasoning: str

    @model_validator(mode="before")
    @classmethod
    def _normalize_display(cls, data: Any) -> Any:
        """Unknown display values become "auto" rather than failing the route."""
        if isinstance(data, dict) and data.get("display") not in (
            "chart",
            "table",
            "value",
            "auto",
        ):
            data = {**data, "display": "auto"}
        return data

    @model_validator(mode="after")
    def _normalize_legacy_chart_route(self) -> RoutingDecision:
        """`sql_and_chart` is the old way of asking for a chart."""
        if self.route == "sql_and_chart":
            self.route = "sql_query"
            if self.display == "auto":
                self.display = "chart"
        return self


# ----------------------------
# Router Agent (Decision Only)
# ----------------------------
class Router:
    def __init__(
        self,
        schema_summary: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ):
        # Allow callers to inject a connection-scoped schema summary; fall back
        # to the default (sample/config) summary when none is provided.
        if not schema_summary:
            raise ValueError(
                "schema_summary is required: render it from the connection's "
                "schema graph (settings.schema_graphs). There is no sample-schema "
                "fallback."
            )
        self.schema_summary = schema_summary

        self.llm = create_chat_llm(
            purpose="router", temperature=0, provider=provider, model=model
        )

        # Output parser
        self.parser = JsonOutputParser(pydantic_object=RoutingDecision)

        # Prompt template
        self.prompt = ChatPromptTemplate.from_template(sample_router_prompt_improvise)

    # ----------------------------
    # PUBLIC METHOD
    # ----------------------------
    def route_question(
        self, question: str, chat_history: str = ""
    ) -> tuple[RoutingDecision, Any]:
        """Route a question and return both decision and response for tracking.

        ``chat_history`` lets the router recognise follow-ups ("same for Canada")
        and keep the previous answer's display format.

        Returns:
            Tuple of (RoutingDecision, response) where response contains usage metadata
        """
        messages = self.prompt.format_messages(
            schema_summary=self.schema_summary,
            chat_history=chat_history.strip() or "(none)",
            question=question,
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

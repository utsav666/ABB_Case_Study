from typing import Literal

from envsetup.env_loader import get_openai_api_key, get_openai_model, load_env
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt.prompts import sample_chart_prompt
from pydantic import BaseModel, SecretStr

load_env()


class ChartDecision(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "histogram", "pie", "box", "area"]
    x_axis: str
    y_axis: str
    reasoning: str


class ChartAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=get_openai_model(),
            temperature=0,
            api_key=SecretStr(get_openai_api_key()),
        )
        self.parser = JsonOutputParser(pydantic_object=ChartDecision)

        self.prompt = ChatPromptTemplate.from_template(sample_chart_prompt)

    def run(self, question, context):

        messages = self.prompt.format_messages(question=question, context=context)

        response = self.llm.invoke(messages)

        parsed = self.parser.parse(str(response.content))

        return ChartDecision(**parsed)

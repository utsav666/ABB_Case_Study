from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Literal
from prompt.prompts import sample_chart_prompt
from envsetup.env_loader import load_env
load_env()

class ChartDecision(BaseModel):
    chart_type: Literal["bar", "line", "scatter"]
    x_axis: str
    y_axis: str
    reasoning: str

class ChartAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=ChartDecision)

        self.prompt = ChatPromptTemplate.from_template(sample_chart_prompt)

    def run(self,question, context):

        messages = self.prompt.format_messages(
            question=question,
            context=context
            )

        response = self.llm.invoke(messages)

        parsed = self.parser.parse(response.content)

        return ChartDecision(**parsed)

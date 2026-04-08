import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List

from envsetup.env_loader import get_openai_api_key, load_env
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt.prompts import table_identify_prompts
from pydantic import BaseModel, Field, SecretStr
from utils.schema_summarizer import Summarizer


class TableIdentification(BaseModel):
    user_question: str = Field(..., description="question asked by the user")
    relevant_tables: List[str] = Field(
        ..., description="Identify the tables absolutely relevant to the user question"
    )


load_env()


class TableIdentiferAgent:
    def __init__(self, user_question):
        self.schema_summary = Summarizer.summarize_schema()
        self.user_question = user_question

        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=SecretStr(get_openai_api_key()),
        )

    def run(self):
        # Inject format instructions from parser (no need to instruct "return JSON")
        parser = JsonOutputParser(pydantic_object=TableIdentification)
        tab_identify_agent_tamplate_prompt = ChatPromptTemplate.from_template(
            table_identify_prompts
        )
        formatted_prompt = tab_identify_agent_tamplate_prompt.format_messages(
            schema_summary=self.schema_summary,
            user_question=self.user_question,
            format_instructions=parser.get_format_instructions(),
        )
        # Run the model
        response = self.llm.invoke(formatted_prompt)
        parsed = parser.parse(str(response.content))
        return parsed


####### unit test ######
# if __name__ == "__main__":
#     TA = TableIdentiferAgent("list of all actor played in the movie 'Anand'")
#     result=TA.run()
#     print(result)

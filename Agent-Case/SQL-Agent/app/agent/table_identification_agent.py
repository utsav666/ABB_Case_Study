import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List

from envsetup.env_loader import load_env
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt.prompts import table_identify_prompts
from pydantic import BaseModel, Field, SecretStr
from utils.schema_summarizer import Summarizer


class TableIdentification(BaseModel):
    user_question: str = Field(user_question="question asked by the user")
    relevant_tables: List[str] = Field(
        relevant_tables="""Identify the tables from the 
    which are absolutely relevant with user question"""
    )


load_env()


class TableIdentiferAgent:
    def __init__(self, user_question):
        self.schema_summary = Summarizer.summarize_schema()
        self.user_question = user_question

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

        self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=SecretStr(api_key))

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
        parsed = parser.parse(response.content)
        return parsed


####### unit test ######
# if __name__ == "__main__":
#     TA = TableIdentiferAgent("list of all actor played in the movie 'Anand'")
#     result=TA.run()
#     print(result)

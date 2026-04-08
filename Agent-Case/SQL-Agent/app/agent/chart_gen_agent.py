from typing import Literal

from envsetup.env_loader import load_env
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

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=SecretStr(api_key))
        self.parser = JsonOutputParser(pydantic_object=ChartDecision)

        self.prompt = ChatPromptTemplate.from_template(sample_chart_prompt)

    def run(self, question, context):

        messages = self.prompt.format_messages(question=question, context=context)

        response = self.llm.invoke(messages)

        parsed = self.parser.parse(response.content)

        return ChartDecision(**parsed)

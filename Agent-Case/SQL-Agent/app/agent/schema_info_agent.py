import os
import sys

import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envsetup.env_loader import load_env
from prompt.prompts import schema_info_gen_prompt
from utils.schema_summarizer import Summarizer

load_env()


class SchemaInfoAgent:
    def __init__(self, user_question, context):
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

        openai.api_key = api_key

        self.user_question = user_question
        self.context = context
        self.schema_summary = Summarizer.summarize_schema()

    def run(self):
        prompt = schema_info_gen_prompt.format(
            schema_summary=self.schema_summary,
            user_question=self.user_question,
            context=self.context,
        )
        print(prompt)
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Schema expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# if __name__ == "__main__":
#     schema_info_agent = SchemaInfoAgent("what are the columns of movie table ? ",
#                                         "This only need schema detail")
#     sample_Res = schema_info_agent.run()
#     print(sample_Res)

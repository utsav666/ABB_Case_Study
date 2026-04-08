import os
import sys

import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envsetup.env_loader import load_env
from prompt.prompts import sql_gen_prompts
from utils.table_struc_creation import TabStrucCreation

load_env()


class SQLGenerationAgent:
    def __init__(self, user_question, table_list):
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
        self.tab_struc_gen = TabStrucCreation(table_list).struc_creation()

    def run(self):
        prompt = sql_gen_prompts.format(
            table_structure=self.tab_struc_gen, user_question=self.user_question
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a world-class SQL generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# if __name__ == "__main__":
#     user_question="list of all actor played in the movie 'Anand'"
#     table_list = ["Movie", "M_Cast", "Person"]
#     sql_gen =  SQLGenerationAgent(user_question,table_list)
#     res=sql_gen.run()
#     print(res)

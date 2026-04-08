import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envsetup.env_loader import load_env
from langchain_openai import ChatOpenAI
from prompt.prompts import fall_back_prompt
from pydantic import SecretStr

load_env()


class GeneralAnswerAgent:
    """
    A simple agent that answers general questions (not schema / SQL specific).
    It uses conversation history + user question + router reasoning to generate a clean answer.
    """

    def __init__(self, model="gpt-4o", temperature=0.2):
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

        self.llm = ChatOpenAI(
            model=model, temperature=temperature, api_key=SecretStr(api_key)
        )

    def run(self, conversation_history, user_question, router_reasoning):
        """
        conversation_history: list of {"role": "...", "content": "..."}
        user_question: str
        router_reasoning: str
        """
        print(conversation_history, ".....conv history.....")
        messages = [{"role": "system", "content": fall_back_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add router reasoning context
        messages.append(
            {"role": "system", "content": f"Router reasoning: {router_reasoning}"}
        )

        # Add user question
        messages.append({"role": "user", "content": user_question})

        # Pass everything to the LLM
        response = self.llm.invoke(messages)

        return response.content


# if __name__ == "__main__":
#     gen_answer_agent = GeneralAnswerAgent()
#     sample_Res = gen_answer_agent.run('what is love?',
#                                       'could you explain ?',
#                                       'The user is asking for an explanation, which typically involves providing a conceptual understanding or clarification of a topic. The question is open-ended and does not specify a request for data retrieval, schema information, or general knowledge, nor does it reference any prior context that would make it a follow-up'
# )
#     print(sample_Res)

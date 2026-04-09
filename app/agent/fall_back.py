from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.envsetup.env_loader import get_openai_api_key, load_env
from app.prompt.prompts import fall_back_prompt

load_env()


class GeneralAnswerAgent:
    """
    A simple agent that answers general questions (not schema / SQL specific).
    It uses conversation history + user question + router reasoning to generate a clean answer.
    """

    def __init__(self, model="gpt-4o", temperature=0.2):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=SecretStr(get_openai_api_key()),
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

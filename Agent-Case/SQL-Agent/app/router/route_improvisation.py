import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt.prompts import sample_router_prompt_improvise
from envsetup.env_loader import load_env
from utils.schema_summarizer import Summarizer
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal


# ------------------------
# Schema-aware Router Model
# ------------------------
load_env()
# ----------------------------
# Output Model
# ----------------------------
class RoutingDecision(BaseModel):
    route: Literal["sql_query", "schema_info", "general_answer", "explanation", "follow_up"]
    reasoning: str

# ----------------------------
# Router Agent (Decision Only)
# ----------------------------
class Router:
    def __init__(self):

        self.schema_summary = Summarizer.summarize_schema()

        # Store full conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
             k=4,  # keep only last 4 messages
            return_messages=True
        )

        # LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Output parser
        self.parser = JsonOutputParser(pydantic_object=RoutingDecision)

        # Prompt template
        self.prompt = ChatPromptTemplate.from_template(sample_router_prompt_improvise)

    # ----------------------------
    # PUBLIC METHOD
    # ----------------------------
    def route_question(self, question: str) -> RoutingDecision:

        # Prepare full prompt
        messages = self.prompt.format_messages(
            schema_summary=self.schema_summary,
            chat_history=self.memory.load_memory_variables({})["chat_history"],
            question=question
        )

        # LLM call
        response = self.llm(messages)

        # Save user AND agent messages to memory
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response.content)

        # Parse into RoutingDecision
        return self.parser.parse(response.content)

if __name__ == "__main__":
    user_question = "Write SQL to find items with low stock from an Inventory table.?"
    route =Router()
    response = route.route_question(user_question)
    print(response)
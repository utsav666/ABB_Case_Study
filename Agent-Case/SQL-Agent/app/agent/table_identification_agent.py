import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.prompts import ChatPromptTemplate
from prompt.prompts import table_identify_prompts
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from envsetup.env_loader import load_env 
from utils.schema_summarizer import Summarizer
# class TableIdentification(BaseModel):
#     user_question: str
#     relevant_tables: List[str]
load_env()
class TableIdentiferAgent:
    def __init__(self,user_question):
        self.schema_summary = Summarizer('SQL-Agent/sample_data/schema_preserve.json')
        self.user_question = user_question
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    def run(self) :
        # Inject format instructions from parser (no need to instruct "return JSON")
        parser = JsonOutputParser()
        tab_identify_agent_tamplate_prompt = ChatPromptTemplate.from_template(table_identify_prompts)
        formatted_prompt = tab_identify_agent_tamplate_prompt.format_messages(
            schema_summary=self.schema_summary,
            user_question=self.user_question,
            format_instructions=parser.get_format_instructions()
    )
        
        # Run the model
        response = self.llm(formatted_prompt)
        return response
# unit test ####3
if __name__ == "__main__":
    TA = TableIdentiferAgent("list of all actor played in the movie 'Anand'")
    result=TA.run()
    print(result)

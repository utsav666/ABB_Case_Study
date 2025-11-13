import sys, os
import openai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt.prompts import schema_info_gen_prompt
from utils.schema_summarizer import Summarizer
from envsetup.env_loader import load_env
load_env()
class SchemaInfoAgent:
    def __init__(self,user_question,context):
        self.user_question = user_question
        self.context = context
        self.schema_summary = Summarizer.summarize_schema()
    def run(self):
        prompt = schema_info_gen_prompt.format(
            schema_summary = self.schema_summary,
            user_question = self.user_question,
            context = self.context
        )
        response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a world-class SQL generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
        return response.choices[0].message.content.strip()

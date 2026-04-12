import openai

from app.envsetup.env_loader import get_openai_api_key, load_env
from app.prompt.prompts import sql_gen_prompts
from app.utils.table_struc_creation import TabStrucCreation

load_env()


class SQLGenerationAgent:
    def __init__(self, user_question, table_list):
        openai.api_key = get_openai_api_key()

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

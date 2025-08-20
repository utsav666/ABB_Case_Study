# FastAPI app code goes here
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import openai
import os

app = FastAPI()
client = OpenAI(api_key = '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')
class Prompt(BaseModel):
    text: str

@app.post("/ask")
async def ask(prompt: Prompt):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.text}]
    )
    return {"response": response.choices[0].message.content}


# if __name__ == "__main__":
#     uvicorn.run('main:app',port=5000)
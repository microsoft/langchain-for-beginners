"""
Lesson 01 - Hello World with LangChain
This example demonstrates a basic LLM call using ChatOpenAI.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    print("🦜🔗 Hello LangChain!\n")

    # Create a chat model instance
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )
    response = model.invoke("Say 'Hello, Ramesh'")
    print("AI Response:" , response.content)

if __name__ == "__main__":
    main()

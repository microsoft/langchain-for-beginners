"""
Lesson 01 - Hello World with LangChain
This example demonstrates a basic LLM call using ChatOpenAI.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Create a ChatOpenAI instance with the model from environment
model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

# Invoke the model with a simple prompt
response = model.invoke("Hello, how are you?")

# Print the response content
print(response.content)

"""
Lesson 01 - Message Types in LangChain
This example demonstrates how to use different message types (SystemMessage, HumanMessage).
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Create a ChatOpenAI instance
model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

# Create messages with different types
messages = [
    SystemMessage(content="You are a helpful assistant that speaks like a pirate."),
    HumanMessage(content="Tell me about the weather today."),
]

# Invoke the model with the messages
response = model.invoke(messages)

# Print the response content
print(response.content)

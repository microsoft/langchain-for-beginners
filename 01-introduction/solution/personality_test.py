"""
Lesson 01 - Solution: Experiment with System Prompts
This solution demonstrates how different system prompts affect LLM responses.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Create a ChatOpenAI instance
model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

# Define different personalities via system prompts
personalities = [
    "You are a helpful assistant that speaks like a pirate.",
    "You are a formal business analyst who uses professional language.",
    "You are an enthusiastic teacher who loves to explain things simply.",
]

# The same question for all personalities
question = "What is Python programming?"

# Test each personality
for personality in personalities:
    print(f"\n--- Personality: {personality[:50]}... ---")
    
    messages = [
        SystemMessage(content=personality),
        HumanMessage(content=question),
    ]
    
    response = model.invoke(messages)
    print(f"Response: {response.content}\n")
    print("-" * 50)

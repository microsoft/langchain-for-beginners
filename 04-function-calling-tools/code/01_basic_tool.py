"""
Basic Tool Example
Run: python 04-function-calling-tools/code/01_basic_tool.py

This example demonstrates how to create and use a simple tool.
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Bind the tool to the model
    model_with_tools = model.bind_tools([get_word_length])

    print("🔧 Basic Tool Example\n")
    print("=" * 80)

    # Invoke the model with a question that requires the tool
    response = model_with_tools.invoke("How long is the word 'LangChain'?")

    print(f"\n🤖 AI Response: {response.content}")
    
    if response.tool_calls:
        print(f"\n🔧 Tool called: {response.tool_calls[0]['name']}")
        print(f"   Arguments: {response.tool_calls[0]['args']}")
        
        # Execute the tool
        result = get_word_length.invoke(response.tool_calls[0]['args'])
        print(f"   Result: {result}")

    print("\n" + "=" * 80)
    print("\n✅ Tools extend AI capabilities!")

if __name__ == "__main__":
    main()

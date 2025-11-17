"""
Basic Template Example
Run: python 03-prompts-messages-outputs/code/01_basic_template.py

This example demonstrates how to create and use basic prompt templates in LangChain.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def main():
    # Initialize the model
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create a basic template with a variable
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful translator."),
        ("human", "Translate '{text}' to {language}")
    ])

    # Create a chain by piping the template to the model
    chain = template | model

    # Use the chain with different inputs
    print("🌍 Translation Example\n")
    print("=" * 80)
    
    # Example 1: Translate to French
    result1 = chain.invoke({
        "text": "Hello, world!",
        "language": "French"
    })
    print(f"\n🇫🇷 French: {result1.content}")

    # Example 2: Translate to Spanish
    result2 = chain.invoke({
        "text": "Hello, world!",
        "language": "Spanish"
    })
    print(f"🇪🇸 Spanish: {result2.content}")

    # Example 3: Translate to Japanese
    result3 = chain.invoke({
        "text": "Hello, world!",
        "language": "Japanese"
    })
    print(f"🇯🇵 Japanese: {result3.content}")

    print("\n" + "=" * 80)
    print("\n✅ Templates make prompts reusable!")
    print("💡 Change the variables, keep the structure")

if __name__ == "__main__":
    main()

"""
Basic Prompt Template
Run: python 03-prompts-messages-outputs/code/03_basic_template.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does template | model create a chain that can be invoked?"
- "What happens if I forget to provide one of the template variables?"
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def main():
    print("📝 Basic Prompt Template Example\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Create a reusable translation template
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{text}"),
    ])

    print("Template created with variables: input_language, output_language, text\n")

    # Create a chain by piping template to model
    chain = template | model

    # Example 1: English to French
    print("1️⃣  Translating to French:")
    result1 = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?",
    })
    print("   →", result1.content, "\n")

    # Example 2: English to Spanish
    print("2️⃣  Translating to Spanish:")
    result2 = chain.invoke({
        "input_language": "English",
        "output_language": "Spanish",
        "text": "Hello, how are you?",
    })
    print("   →", result2.content, "\n")

    # Example 3: English to Japanese
    print("3️⃣  Translating to Japanese:")
    result3 = chain.invoke({
        "input_language": "English",
        "output_language": "Japanese",
        "text": "Hello, how are you?",
    })
    print("   →", result3.content, "\n")

    print("✅ Same template, different outputs!")
    print("💡 Templates make prompts reusable and maintainable.")


if __name__ == "__main__":
    main()

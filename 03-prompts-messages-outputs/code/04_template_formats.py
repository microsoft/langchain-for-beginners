"""
Multiple Template Formats
Run: python 03-prompts-messages-outputs/code/04_template_formats.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "When should I use ChatPromptTemplate vs PromptTemplate?"
- "How does string_template.format() differ from using pipe and invoke?"
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def main():
    print("🎨 Template Formats Example\n")
    print("=" * 80)

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Format 1: ChatPromptTemplate (structured messages)
    print("\n1️⃣  ChatPromptTemplate (Recommended for chat models):\n")

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} who speaks in {style} style."),
        ("human", "{question}"),
    ])

    chain1 = chat_template | model

    result1 = chain1.invoke({
        "role": "pirate captain",
        "style": "dramatic and adventurous",
        "question": "What is Python?",
    })

    print("Pirate response:")
    print(result1.content)

    # Format 2: PromptTemplate (simple string-based)
    print("\n" + "=" * 80)
    print("\n2️⃣  PromptTemplate (Simple string format):\n")

    string_template = PromptTemplate.from_template(
        "Write a {adjective} {item} about {topic}."
    )

    # Format the template to see the final prompt
    formatted_prompt = string_template.format(
        adjective="funny",
        item="limerick",
        topic="Python developers",
    )

    print("Generated prompt:", formatted_prompt)

    result2 = model.invoke(formatted_prompt)
    print("\nResponse:")
    print(result2.content)

    # Format 3: Multiple variables
    print("\n" + "=" * 80)
    print("\n3️⃣  Complex Template with Many Variables:\n")

    complex_template = ChatPromptTemplate.from_messages([
        ("system", "You are a {job_title} at {company} writing to a {recipient_role}."),
        ("human", "Write a {message_type} about {topic}. Tone: {tone}"),
    ])

    chain3 = complex_template | model

    result3 = chain3.invoke({
        "job_title": "Senior Developer",
        "company": "TechCorp",
        "recipient_role": "Product Manager",
        "message_type": "brief update",
        "topic": "API migration progress",
        "tone": "professional but friendly",
    })

    print(result3.content)

    print("\n" + "=" * 80)
    print("\n✅ Key Takeaways:")
    print("   - ChatPromptTemplate: Best for multi-message conversations")
    print("   - PromptTemplate: Good for simple single-string prompts")
    print("   - Both support multiple variables with {variable} syntax")
    print("   - Use | model to create chains")


if __name__ == "__main__":
    main()

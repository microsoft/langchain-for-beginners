"""
Multi-Turn Conversation
Run: python 02-chat-models/code/01_multi_turn.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "Why do we need to append AIMessage to the messages list after each response?"
- "How would I implement a loop to keep the conversation going with user input?"
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def main():
    print("💬 Multi-Turn Conversation Example\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Start with system message and first question
    messages = [
        SystemMessage(content="You are a helpful coding tutor who gives clear, concise explanations."),
        HumanMessage(content="What is Python?"),
    ]

    print("👤 User: What is Python?")

    # First exchange
    response1 = model.invoke(messages)
    print(f"\n🤖 AI: {response1.content}")
    messages.append(AIMessage(content=str(response1.content)))

    # Second exchange - AI remembers the context
    print("\n👤 User: Can you show me a simple example?")
    messages.append(HumanMessage(content="Can you show me a simple example?"))

    response2 = model.invoke(messages)
    print(f"\n🤖 AI: {response2.content}")

    # Third exchange - AI still remembers everything
    print("\n👤 User: What are the benefits compared to other languages?")
    messages.append(AIMessage(content=str(response2.content)))
    messages.append(HumanMessage(content="What are the benefits compared to other languages?"))

    response3 = model.invoke(messages)
    print(f"\n🤖 AI: {response3.content}")

    print("\n\n✅ Notice how the AI maintains context throughout the conversation!")
    print(f"📊 Total messages in history: {len(messages)}")


if __name__ == "__main__":
    main()

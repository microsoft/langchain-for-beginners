"""
Message Construction Patterns
Run: python 03-prompts-messages-outputs/code/02_message_construction.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How do I build a multi-turn conversation with message arrays?"
- "Can I serialize and deserialize message arrays for storage?"
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def main():
    print("🔨 Message Construction Patterns\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # ==========================================
    # PATTERN 1: Basic Message Types
    # ==========================================
    print("=" * 80)
    print("\n📝 PATTERN 1: Basic Message Types\n")

    system_msg = SystemMessage(content="You are a helpful programming assistant.")
    human_msg = HumanMessage(content="What is a variable?")

    print("Message types:")
    print(f"   • SystemMessage: {system_msg.content}")
    print(f"   • HumanMessage: {human_msg.content}\n")

    response1 = model.invoke([system_msg, human_msg])
    print(f"🤖 AI Response: {response1.content}\n")

    # ==========================================
    # PATTERN 2: Multi-Turn Conversations
    # ==========================================
    print("=" * 80)
    print("\n💬 PATTERN 2: Multi-Turn Conversations\n")

    conversation_messages = [
        SystemMessage(content="You are a math tutor for beginners."),
        HumanMessage(content="What is 5 + 3?"),
        AIMessage(content="5 + 3 equals 8!"),
        HumanMessage(content="Now what is 8 * 2?"),
    ]

    print("Conversation history:")
    for i, msg in enumerate(conversation_messages):
        role = msg.type
        print(f"   {i + 1}. [{role}]: {msg.content}")

    response2 = model.invoke(conversation_messages)
    print(f"\n🤖 AI Response: {response2.content}\n")

    print("💡 Key insights:")
    print("   • SystemMessage sets the AI's role (first message)")
    print("   • HumanMessage represents user input")
    print("   • AIMessage represents previous AI responses")
    print("   • Order matters - messages build conversation context")

    # ==========================================
    # PATTERN 3: Dynamic Message Construction
    # ==========================================
    print("\n" + "=" * 80)
    print("\n🎯 PATTERN 3: Dynamic Message Construction\n")

    def create_conversation(
        role: str,
        examples: list[dict[str, str]],
        new_question: str,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=f"You are a {role}.")]

        # Add examples (few-shot pattern using messages)
        for example in examples:
            messages.append(HumanMessage(content=example["question"]))
            messages.append(AIMessage(content=example["answer"]))

        # Add the new question
        messages.append(HumanMessage(content=new_question))

        return messages

    emoji_messages = create_conversation(
        "emoji translator",
        [
            {"question": "happy", "answer": "😊"},
            {"question": "sad", "answer": "😢"},
            {"question": "excited", "answer": "🎉"},
        ],
        "surprised",
    )

    print("Dynamically constructed conversation:")
    for i, msg in enumerate(emoji_messages):
        print(f"   {i + 1}. [{msg.type}]: {msg.content}")

    response3 = model.invoke(emoji_messages)
    print(f"\n🤖 AI Response: {response3.content}\n")

    print("💡 This pattern is useful for:")
    print("   • Building few-shot prompts programmatically")
    print("   • Creating conversation builders")
    print("   • Managing state in agents")
    print("   • Storing/loading conversation history from databases")

    # ==========================================
    # PATTERN 4: Message Metadata
    # ==========================================
    print("\n" + "=" * 80)
    print("\n🏷️  PATTERN 4: Messages with Metadata\n")

    from datetime import datetime

    message_with_metadata = HumanMessage(
        content="What's the weather like?",
        # Additional metadata can be stored
        additional_kwargs={
            "timestamp": datetime.now().isoformat(),
            "userId": "user-123",
        },
    )

    print("Message with metadata:")
    print(f"   Content: {message_with_metadata.content}")
    print(f"   Metadata: {message_with_metadata.additional_kwargs}\n")

    print("💡 Use metadata for:")
    print("   • Tracking conversation timestamps")
    print("   • Storing user IDs for multi-user systems")
    print("   • Adding context without affecting AI processing")
    print("   • Debugging and logging")

    # ==========================================
    # COMPARISON WITH AGENTS
    # ==========================================
    print("\n" + "=" * 80)
    print("\n🚀 How Agents Use Messages\n")

    print("When you use create_agent() (Chapter 7), it:")
    print("   1. Takes message arrays as input")
    print("   2. Processes messages through middleware")
    print("   3. Adds ToolMessage for tool calls/results")
    print("   4. Returns updated message array with agent's response\n")

    print("Example agent flow:")
    print("   [HumanMessage] → Agent → [HumanMessage, ToolMessage, AIMessage]")
    print("                              ↑")
    print("                        Agent adds tool")
    print("                        call and response\n")

    print("✅ Messages are the foundation of agent systems!")
    print("   Learn this pattern → use it in Chapter 7 (Agents & MCP)\n")

    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Chapter 8 Assignment Solution: Bonus Challenge
Conversational Agentic RAG

Run: python 08-agentic-rag-systems/solution/conversational_rag.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Knowledge base about Python
knowledge_base = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991.",
        metadata={"title": "Python Overview", "section": "Introduction"},
    ),
    Document(
        page_content="Python's main benefits include easy-to-read syntax, extensive standard library, cross-platform compatibility, and strong community support. It's used for web development, data science, AI, and automation.",
        metadata={"title": "Python Benefits", "section": "Advantages"},
    ),
    Document(
        page_content="Python uses duck typing and dynamic typing. Variables don't need type declarations, though type hints are supported since Python 3.5+ for better code documentation and IDE support.",
        metadata={"title": "Python Type System", "section": "Type System"},
    ),
    Document(
        page_content="Python decorators are a powerful feature that allows you to modify or enhance functions and classes. Common decorators include @property, @staticmethod, @classmethod, and custom decorators using functools.wraps.",
        metadata={"title": "Python Decorators", "section": "Advanced Features"},
    ),
    Document(
        page_content="Python's list comprehensions provide a concise way to create lists based on existing sequences. They're more readable and often faster than traditional for loops for simple transformations.",
        metadata={"title": "List Comprehensions", "section": "Core Features"},
    ),
]


def main():
    print("💬 Conversational Agentic RAG System\n")
    print("=" * 80 + "\n")

    # 1. Setup
    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    print(f"Creating vector store with {len(knowledge_base)} documents...\n")

    # 2. Create vector store
    vector_store = InMemoryVectorStore.from_documents(knowledge_base, embeddings)

    # 3. Create retrieval tool for the agent
    @tool
    def search_python_knowledge_base(query: str) -> str:
        """Search the Python knowledge base for information about Python features, benefits, type system, decorators, and list comprehensions. Use this when you need specific information about Python from the documentation."""
        print(f'   🔍 Agent searching for: "{query}"')
        results = vector_store.similarity_search(query, k=2)

        if not results:
            return "No relevant Python documentation found."

        return "\n\n".join(
            f"[{doc.metadata['title']}]: {doc.page_content}"
            for doc in results
        )

    # 4. Create agent with retrieval tool
    agent = create_react_agent(model, tools=[search_python_knowledge_base])

    # 5. Initialize conversation history
    conversation_history: list[HumanMessage | AIMessage] = []

    # Check if running in CI mode for automated testing
    is_ci = os.getenv("CI") == "true"

    print("💡 Instructions:")
    print("   - Ask questions about Python")
    print("   - Ask follow-up questions to test conversation memory")
    print("   - Type 'reset' to start a new conversation")
    print("   - Type 'exit' or 'quit' to end\n")
    print("=" * 80 + "\n")

    # 6. Conversation loop
    question_count = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\n👋 Goodbye! Thanks for chatting!\n")
            break

        # Handle special commands
        if user_input.lower() in ("exit", "quit"):
            print("\n👋 Goodbye! Thanks for chatting!\n")
            break

        if user_input.lower() == "reset":
            conversation_history.clear()
            print("\n🔄 Conversation reset. Starting fresh!\n")
            continue

        if not user_input:
            continue

        # Add user message to history
        user_message = HumanMessage(content=user_input)
        conversation_history.append(user_message)

        try:
            # Invoke agent with full conversation history
            response = agent.invoke({
                "messages": list(conversation_history),
            })

            # Get agent's response
            agent_message = response["messages"][-1]

            # Add agent's response to history
            conversation_history.append(AIMessage(content=agent_message.content))

            print(f"\nAgent: {agent_message.content}\n")
            print("=" * 80 + "\n")

            # In CI mode, exit after answering one question
            question_count += 1
            if is_ci and question_count >= 1:
                print("✅ CI Mode: Answered one question successfully. Exiting.\n")
                break

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

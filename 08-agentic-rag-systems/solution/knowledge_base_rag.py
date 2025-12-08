"""
Chapter 8 Assignment Solution: Challenge 1
Personal Knowledge Base Q&A (Agentic RAG)

Run: python 08-agentic-rag-systems/solution/knowledge_base_rag.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI

load_dotenv()


def get_embeddings_endpoint():
    """Get the Azure OpenAI endpoint, removing /openai/v1 suffix if present."""
    endpoint = os.getenv("AI_ENDPOINT", "")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "")
    return endpoint


# Sample knowledge base - you can replace with your own documents
knowledge_base = [
    Document(
        page_content="Python is a typed superset of Python... wait, that's TypeScript! Python is a versatile, interpreted programming language known for its readability and extensive standard library. It's popular for data science, web development, and automation.",
        metadata={"title": "Python Basics", "source": "my-notes"},
    ),
    Document(
        page_content="React hooks like useState and useEffect allow functional components to have state and side effects. useState returns a state variable and a setter function, while useEffect runs side effects after render.",
        metadata={"title": "React Hooks", "source": "my-notes"},
    ),
    Document(
        page_content="Docker containers package applications with their dependencies, ensuring consistent behavior across environments. Containers are lightweight, portable, and share the host OS kernel, making them more efficient than virtual machines.",
        metadata={"title": "Docker Containers", "source": "my-notes"},
    ),
    Document(
        page_content="REST APIs follow principles like statelessness, client-server architecture, and uniform interface. HTTP methods (GET, POST, PUT, DELETE) map to CRUD operations. Status codes indicate request outcomes.",
        metadata={"title": "REST API Design", "source": "my-notes"},
    ),
    Document(
        page_content="Git branching strategies like Git Flow and trunk-based development help teams manage code changes. Feature branches isolate work, pull requests enable code review, and merge commits preserve history.",
        metadata={"title": "Git Workflows", "source": "my-notes"},
    ),
    Document(
        page_content="Python's asyncio module enables asynchronous programming with async/await syntax. The event loop manages coroutines, allowing efficient handling of I/O-bound operations without blocking.",
        metadata={"title": "Python Async", "source": "my-notes"},
    ),
    Document(
        page_content="Database indexing improves query performance by creating data structures that allow fast lookups. B-tree indexes work well for range queries, while hash indexes excel at equality comparisons. Over-indexing can slow writes.",
        metadata={"title": "Database Indexing", "source": "my-notes"},
    ),
]


def main():
    print("📚 Personal Knowledge Base Q&A (Agentic RAG)\n")
    print("=" * 80 + "\n")

    # 1. Setup
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
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
    def search_my_notes(query: str) -> str:
        """Search my personal knowledge base for information about Python, React, Docker, REST APIs, Git, and databases. Use this when you need specific technical information from my notes."""
        print(f'   🔍 Agent searching for: "{query}"')
        results = vector_store.similarity_search(query, k=3)

        if not results:
            return "No relevant information found in the knowledge base."

        return "\n\n".join(
            f"[{doc.metadata['title']}]: {doc.page_content}"
            for doc in results
        )

    # 4. Create agent with retrieval tool
    agent = create_agent(
        model,
        tools=[search_my_notes],
        system_prompt="You are a helpful personal assistant with access to my knowledge base containing notes about Python, React, Docker, REST APIs, Git, and databases. Use the search tool when you need specific technical information from my notes. For general knowledge questions, answer directly.",
    )

    # 5. Test with mix of questions
    questions = [
        # General knowledge - agent should answer directly
        "What is 2 + 2?",
        "What is the capital of France?",
        # Knowledge base questions - agent should search
        "What is Python?",
        "How does async work in Python?",
        "What are the benefits of Docker containers?",
        # Not in knowledge base - agent may search but won't find
        "What is Kubernetes?",
    ]

    print("💡 Testing agent with different types of questions:\n")
    print("Watch how the agent intelligently decides when to search vs answer directly!\n")

    for question in questions:
        print("=" * 80)
        print(f"\n❓ Question: {question}\n")

        response = agent.invoke({
            "messages": [HumanMessage(content=question)],
        })

        final_message = response["messages"][-1]
        print("🤖 Answer:", final_message.content)
        print()

    print("=" * 80)
    print("\n✅ Key Observations:")
    print("   - Agent answers general knowledge questions directly (no search needed)")
    print("   - Agent searches knowledge base for technical questions")
    print("   - Agent decides WHEN to search based on question context")
    print("   - More efficient than always searching!")
    print("\n💡 This is the power of Agentic RAG - intelligent, autonomous retrieval!")


if __name__ == "__main__":
    main()

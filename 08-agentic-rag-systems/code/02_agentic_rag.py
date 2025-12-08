"""
Agentic RAG System

Run: python 08-agentic-rag-systems/code/02_agentic_rag.py

This example demonstrates the modern agentic RAG pattern where an AI agent
intelligently decides when to search your documents vs. answering directly
from its general knowledge. Unlike traditional RAG that always searches,
agentic RAG gives your AI autonomy to determine whether retrieval is necessary.

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does the agent decide when to use the retrieval tool vs answering directly?"
- "How would I add metadata filtering to the retrieval tool?"
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


def main():
    print("🤖 Agentic RAG System Example\n")

    # 1. Setup embeddings and model
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

    # 2. Create knowledge base about LangChain and RAG
    docs = [
        Document(
            page_content="LangChain was created in 2022 and quickly became popular for building LLM applications. The Python version was first, followed by LangChain.js for JavaScript/TypeScript.",
            metadata={"source": "langchain-history", "topic": "introduction"},
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) combines document retrieval with LLM generation. It allows models to access external knowledge without retraining, making responses more accurate and up-to-date.",
            metadata={"source": "rag-explanation", "topic": "concepts"},
        ),
        Document(
            page_content="Vector stores like Pinecone, Weaviate, and Chroma enable semantic search over documents. They store embeddings and perform fast similarity searches to find relevant content.",
            metadata={"source": "vector-stores", "topic": "infrastructure"},
        ),
        Document(
            page_content="LangChain supports multiple document loaders for PDFs, web pages, databases, and APIs. Text splitters help break large documents into chunks that fit within LLM context windows while preserving semantic meaning.",
            metadata={"source": "document-processing", "topic": "development"},
        ),
    ]

    print(f"📚 Creating vector store with {len(docs)} documents...\n")

    # 3. Create vector store
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    # 4. Create retrieval tool from vector store
    # The agent will decide when to use this tool based on the question
    @tool
    def search_langchain_docs(query: str) -> str:
        """Search LangChain documentation for specific information about LangChain, RAG systems, vector stores, and document processing. Use this when you need factual information from the LangChain knowledge base."""
        print(f'   🔍 Agent is searching for: "{query}"')
        results = vector_store.similarity_search(query, k=2)
        return "\n\n".join(
            f"[{doc.metadata['source']}]: {doc.page_content}"
            for doc in results
        )

    # 5. Create agent with retrieval tool
    # The agent will autonomously decide when to search vs answer directly
    agent = create_agent(
        model,
        tools=[search_langchain_docs],
        system_prompt="You are a helpful assistant with access to LangChain documentation. Use the search tool when you need specific information about LangChain, RAG, or vector stores. For general knowledge questions, answer directly without searching.",
    )

    # 6. Ask different types of questions to see agent decision-making
    questions = [
        # General knowledge - agent should answer directly without searching
        "What is the capital of France?",
        # Document-specific questions - agent should use the retrieval tool
        "When was LangChain created?",
        "What is RAG and why is it useful?",
    ]

    print("💡 Watch how the agent decides when to search vs answer directly:\n")

    for question in questions:
        print("=" * 80)
        print(f"\n❓ Question: {question}\n")

        response = agent.invoke({
            "messages": [HumanMessage(content=question)],
        })

        # The agent's final response
        final_message = response["messages"][-1]
        print("🤖 Answer:", final_message.content)
        print()

    print("=" * 80)
    print("\n💡 Key Observations:")
    print("   - Agent answers general knowledge questions directly (no search)")
    print("   - Agent uses retrieval tool for document-specific questions")
    print("   - Agent autonomously decides WHEN to search based on context")
    print("   - More efficient than traditional RAG that always searches")
    print("\n🎯 Agentic RAG Benefits:")
    print("   ✓ Reduced API calls (only searches when needed)")
    print("   ✓ Faster responses for general knowledge questions")
    print("   ✓ Better user experience with intelligent decision-making")
    print("   ✓ Scalable to multiple retrieval sources and tools")


if __name__ == "__main__":
    main()

"""
Traditional RAG System

Run: python 08-agentic-rag-systems/code/01a_traditional_rag.py

This example demonstrates the traditional "always-search" RAG pattern where
the system searches documents for EVERY query, even simple ones that don't
need retrieval. Compare this to the agentic approach in 02_agentic_rag.py.

Traditional RAG Pattern:
1. User asks a question (ANY question)
2. System ALWAYS searches the vector store
3. System passes retrieved documents + question to LLM
4. LLM generates answer based on retrieved context

Problem: Searches even for "What is 2+2?" - wasting time and money!

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "Why does traditional RAG search for every query?"
- "What are the cost implications of always searching?"
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
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
    print("📖 Traditional RAG System Example\n")
    print("=" * 80 + "\n")

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

    # Knowledge base about LangChain and RAG
    docs = [
        Document(
            page_content="LangChain was created in 2022 and quickly became popular for building LLM applications. The Python version was first, followed by the JavaScript/TypeScript port.",
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

    # Create vector store
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    # Simple 2-step RAG function (no chains!)
    def traditional_rag(question: str, k: int = 2) -> tuple[str, list[Document]]:
        """
        Traditional RAG: Always retrieves documents, then generates answer.
        
        Step 1: Retrieve relevant documents (ALWAYS runs)
        Step 2: Generate answer using retrieved context
        """
        # Step 1: ALWAYS retrieve documents
        retrieved_docs = vector_store.similarity_search(question, k=k)
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 2: Generate answer with context
        messages = [
            SystemMessage(content="""You are a helpful assistant. Answer the question based on the provided context.
If the question can be answered without the context, still try to reference it if relevant.
If the context is not helpful, answer based on your general knowledge."""),
            HumanMessage(content=f"""Context:
{context}

Question: {question}

Answer:""")
        ]
        
        response = model.invoke(messages)
        return response.content, retrieved_docs

    print("💡 Watch how traditional RAG searches for EVERY query:\n")

    # Mix of questions - general knowledge AND document-specific
    questions = [
        "What is the capital of France?",  # General knowledge - doesn't need search!
        "When was LangChain created?",  # Document-specific - needs search
        "What is RAG and why is it useful?",  # Document-specific - needs search
    ]

    for question in questions:
        print("=" * 80)
        print(f"\n❓ Question: {question}\n")

        print("   🔍 Traditional RAG: ALWAYS searching documents...")
        answer, retrieved_docs = traditional_rag(question)

        print(f"🤖 Answer: {answer}")
        print(f"\n📄 Searched {len(retrieved_docs)} documents (even if not needed)")
        for i, doc in enumerate(retrieved_docs):
            print(f"   {i + 1}. {doc.metadata['source']}")
        print()

    print("=" * 80)
    print("\n💡 Key Observations:")
    print("   - Traditional RAG searches on EVERY query")
    print("   - Even 'What is the capital of France?' triggers a search")
    print("   - Wastes API calls, time, and money on unnecessary searches")
    print("   - Simple, predictable, but inefficient")
    print("\n🎯 Compare to Agentic RAG (Example 2):")
    print("   ✓ Agent decides when to search")
    print("   ✓ Answers general knowledge questions directly")
    print("   ✓ Only searches when needed for document-specific info")
    print("   ✓ More efficient and cost-effective")


if __name__ == "__main__":
    main()

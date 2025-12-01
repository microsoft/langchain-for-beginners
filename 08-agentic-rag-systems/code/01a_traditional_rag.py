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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def main():
    print("📖 Traditional RAG System Example\n")
    print("=" * 80 + "\n")

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

    # Create vector store and retriever
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Create RAG prompt
    prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

{context}

Question: {input}

Answer: Provide a clear answer. If the question can be answered without the context, still try to reference it if relevant.
""")

    # Create traditional RAG chain - ALWAYS searches!
    combine_docs_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

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
        response = rag_chain.invoke({"input": question})

        print(f"🤖 Answer: {response['answer']}")
        print(f"\n📄 Searched {len(response['context'])} documents (even if not needed)")
        for i, doc in enumerate(response["context"]):
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

"""
Sample: Document Organizer

Shows how to organize and categorize documents using metadata,
then retrieve documents by filtering.

Run: python 07-documents-embeddings-semantic-search/samples/doc_organizer.py
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def main():
    print("📂 Document Organizer\n")
    print("=" * 80 + "\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create documents with rich metadata
    docs = [
        Document(
            page_content="Python is great for data science and machine learning.",
            metadata={
                "category": "programming",
                "language": "python",
                "level": "beginner",
                "date": "2024-01-15",
            },
        ),
        Document(
            page_content="JavaScript powers interactive web applications.",
            metadata={
                "category": "programming",
                "language": "javascript",
                "level": "beginner",
                "date": "2024-01-20",
            },
        ),
        Document(
            page_content="Neural networks can learn complex patterns in data.",
            metadata={
                "category": "ai",
                "topic": "deep_learning",
                "level": "advanced",
                "date": "2024-02-01",
            },
        ),
        Document(
            page_content="Vector databases store and query embedding vectors efficiently.",
            metadata={
                "category": "database",
                "topic": "vectors",
                "level": "intermediate",
                "date": "2024-02-10",
            },
        ),
        Document(
            page_content="LangChain simplifies building LLM applications.",
            metadata={
                "category": "ai",
                "language": "python",
                "level": "intermediate",
                "date": "2024-02-15",
            },
        ),
    ]

    print("📚 Document Library:")
    print("─" * 80)
    for doc in docs:
        content_preview = doc.page_content[:50]
        print(f"• {content_preview}...")
        print(f"  Metadata: {doc.metadata}")
    print()

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("=" * 80 + "\n")

    # Filter by category
    print("🔍 Filter: Category = 'programming'\n")

    # Note: For filtering, we need to search and then filter results manually
    # since InMemoryVectorStore doesn't support native filtering
    query = "coding languages"
    all_results = vector_store.similarity_search(query, k=10)
    filtered_results = [
        doc for doc in all_results if doc.metadata.get("category") == "programming"
    ]

    for doc in filtered_results:
        print(f"  • {doc.page_content}")
        print(f"    {doc.metadata}\n")

    print("=" * 80 + "\n")

    # Filter by level
    print("🔍 Filter: Level = 'advanced' OR 'intermediate'\n")

    query = "machine learning systems"
    all_results = vector_store.similarity_search(query, k=10)
    filtered_results = [
        doc
        for doc in all_results
        if doc.metadata.get("level") in ["advanced", "intermediate"]
    ]

    for doc in filtered_results:
        print(f"  • {doc.page_content}")
        print(f"    {doc.metadata}\n")

    print("=" * 80)
    print("\n💡 Best Practices for Document Metadata:")
    print("   1. Use consistent metadata keys across documents")
    print("   2. Include category, date, source, and relevance tags")
    print("   3. Metadata enables post-filtering for precise results")
    print("   4. Use production vector stores for native filter support")


if __name__ == "__main__":
    main()

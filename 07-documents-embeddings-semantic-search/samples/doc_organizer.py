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
from langchain_openai import AzureOpenAIEmbeddings

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
    print("📂 Document Organizer\n")
    print("=" * 80 + "\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    # Create documents with rich metadata
    docs = [
        Document(
            page_content="Python is great for data science and machine learning.",
            metadata={
                "category": "programming",
                "language": "python",
                "difficulty": "beginner",
                "date": "2024-01-15",
            },
        ),
        Document(
            page_content="Advanced Python decorators and metaclasses explained.",
            metadata={
                "category": "programming",
                "language": "python",
                "difficulty": "advanced",
                "date": "2024-02-20",
            },
        ),
        Document(
            page_content="JavaScript async/await patterns for modern web apps.",
            metadata={
                "category": "programming",
                "language": "javascript",
                "difficulty": "intermediate",
                "date": "2024-01-10",
            },
        ),
        Document(
            page_content="Introduction to neural networks and deep learning.",
            metadata={
                "category": "AI",
                "topic": "deep-learning",
                "difficulty": "beginner",
                "date": "2024-03-01",
            },
        ),
        Document(
            page_content="Transformer architectures and attention mechanisms.",
            metadata={
                "category": "AI",
                "topic": "transformers",
                "difficulty": "advanced",
                "date": "2024-02-15",
            },
        ),
    ]

    print(f"📚 Organizing {len(docs)} documents...\n")

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Documents indexed!\n")
    print("=" * 80 + "\n")

    # Search with different queries
    queries = [
        "machine learning basics",
        "advanced programming patterns",
        "web development",
    ]

    for query in queries:
        print(f'🔍 Query: "{query}"\n')

        results = vector_store.similarity_search(query, k=2)

        for i, doc in enumerate(results):
            print(f"   {i + 1}. {doc.page_content[:50]}...")
            print(f"      Category: {doc.metadata.get('category')}")
            print(f"      Difficulty: {doc.metadata.get('difficulty')}")
            print(f"      Date: {doc.metadata.get('date')}\n")

        print("─" * 80 + "\n")

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Metadata enriches documents with structured information")
    print("   - You can filter results based on metadata after search")
    print("   - Combine semantic search with metadata filtering for precise results")


if __name__ == "__main__":
    main()

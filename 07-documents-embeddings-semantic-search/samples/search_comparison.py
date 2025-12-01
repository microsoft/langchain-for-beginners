"""
Sample: Search Comparison - Keyword vs Semantic

This sample demonstrates the difference between keyword-based search
and semantic search using embeddings.

Run: python 07-documents-embeddings-semantic-search/samples/search_comparison.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def keyword_search(docs: list[Document], query: str) -> list[Document]:
    """Simple keyword-based search."""
    query_words = query.lower().split()
    results = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        if any(word in content_lower for word in query_words):
            results.append(doc)
    return results


def main():
    print("🔍 Search Comparison: Keyword vs Semantic\n")
    print("=" * 80 + "\n")

    # Create sample documents
    docs = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"id": 1},
        ),
        Document(
            page_content="A speedy auburn canine leaps above the idle hound.",
            metadata={"id": 2},
        ),
        Document(
            page_content="Python is a popular programming language for AI.",
            metadata={"id": 3},
        ),
        Document(
            page_content="Machine learning models can classify images.",
            metadata={"id": 4},
        ),
        Document(
            page_content="Dogs are loyal and playful companions for families.",
            metadata={"id": 5},
        ),
    ]

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    # Test queries
    queries = [
        "fast fox jumping",
        "coding languages",
        "pet animals",
    ]

    for query in queries:
        print(f'Query: "{query}"\n')
        print("─" * 80)

        # Keyword search
        keyword_results = keyword_search(docs, query)
        print(f"\n📝 Keyword Search Results: {len(keyword_results)} found")
        if keyword_results:
            for doc in keyword_results:
                print(f"   • {doc.page_content}")
        else:
            print("   (No exact keyword matches)")

        # Semantic search
        semantic_results = vector_store.similarity_search(query, k=2)
        print(f"\n🧠 Semantic Search Results: {len(semantic_results)} found")
        for doc in semantic_results:
            print(f"   • {doc.page_content}")

        print("\n" + "=" * 80 + "\n")

    print("💡 Key Observation:")
    print("   Semantic search finds documents with SIMILAR MEANING")
    print("   even when exact keywords don't match!")
    print("   - 'fast fox jumping' finds 'speedy auburn canine leaps'")
    print("   - 'coding languages' finds 'programming language'")
    print("   - 'pet animals' finds 'dogs' and 'loyal companions'")


if __name__ == "__main__":
    main()

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
        Document(page_content="Python is great for machine learning and AI applications."),
        Document(page_content="JavaScript powers interactive web experiences."),
        Document(page_content="Data science involves statistical analysis and modeling."),
        Document(page_content="Neural networks learn patterns from training examples."),
        Document(page_content="Cats are independent pets that enjoy napping."),
        Document(page_content="Dogs love outdoor activities and playing fetch."),
    ]

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    # Test queries
    queries = [
        "artificial intelligence programming",  # Semantic: should find Python/ML
        "deep learning models",  # Semantic: should find neural networks
        "pets for apartments",  # Semantic: should find cats
        "outdoor exercise",  # Semantic: should find dogs
    ]

    for query in queries:
        print(f'Query: "{query}"\n')

        # Keyword search
        keyword_results = keyword_search(docs, query)
        print(f"📝 Keyword Search ({len(keyword_results)} results):")
        if keyword_results:
            for doc in keyword_results[:2]:
                print(f"   - {doc.page_content[:60]}...")
        else:
            print("   No exact keyword matches found!")

        # Semantic search
        semantic_results = vector_store.similarity_search(query, k=2)
        print(f"\n🧠 Semantic Search (top 2):")
        for doc in semantic_results:
            print(f"   - {doc.page_content[:60]}...")

        print("\n" + "─" * 80 + "\n")

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Keyword search only finds exact word matches")
    print("   - Semantic search understands meaning and context")
    print("   - Semantic search finds relevant content without exact keywords")


if __name__ == "__main__":
    main()

"""
Similarity Search with Scores

Run: python 07-documents-embeddings-semantic-search/code/07_similarity_scores.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What similarity score threshold should I use to filter out irrelevant results?"
- "How does similarity_search_with_score differ from the regular similarity_search method?"
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


def main():
    print("📊 Similarity Search with Scores\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    # Create a diverse set of documents
    docs = [
        Document(
            page_content="Python is excellent for data science and machine learning applications.",
            metadata={"category": "programming", "language": "python"},
        ),
        Document(
            page_content="JavaScript powers interactive web applications and modern frontends.",
            metadata={"category": "programming", "language": "javascript"},
        ),
        Document(
            page_content="Cats are independent pets that sleep up to 16 hours a day.",
            metadata={"category": "animals", "type": "cat"},
        ),
        Document(
            page_content="Dogs are social animals that require daily walks and playtime.",
            metadata={"category": "animals", "type": "dog"},
        ),
        Document(
            page_content="Machine learning models learn patterns from training data.",
            metadata={"category": "AI", "topic": "ML"},
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"category": "AI", "topic": "DL"},
        ),
    ]

    print(f"📚 Creating vector store with {len(docs)} documents...\n")

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Vector store created!\n")
    print("=" * 80 + "\n")

    # Search queries with different relevance levels
    queries = [
        "AI and machine learning programming",
        "pets that need less attention",
        "web development frameworks",
        "cooking recipes",  # Intentionally unrelated
    ]

    for query in queries:
        print(f'🔍 Query: "{query}"\n')

        # Get results with scores
        results_with_scores = vector_store.similarity_search_with_score(query, k=3)

        for doc, score in results_with_scores:
            relevance = "🟢 High" if score > 0.8 else "🟡 Medium" if score > 0.6 else "🔴 Low"
            print(f"   Score: {score:.4f} {relevance}")
            print(f"   Content: {doc.page_content[:60]}...")
            print(f"   Category: {doc.metadata.get('category')}\n")

        print("─" * 80 + "\n")

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Scores help filter out irrelevant results")
    print("   - Higher scores indicate stronger semantic similarity")
    print("   - Set thresholds based on your use case (e.g., >0.7 for relevance)")


if __name__ == "__main__":
    main()

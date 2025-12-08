"""
Vector Store and Semantic Search

Run: python 07-documents-embeddings-semantic-search/code/06_vector_store.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What's the difference between InMemoryVectorStore and persistent stores like Pinecone?"
- "Can I save and load a vector store to avoid recomputing embeddings?"
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
    print("🗄️  Vector Store and Semantic Search\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    # Create documents about different topics
    docs = [
        Document(
            page_content="Python is a popular programming language for data science and machine learning.",
            metadata={"category": "programming", "language": "python"},
        ),
        Document(
            page_content="JavaScript is widely used for web development and building interactive websites.",
            metadata={"category": "programming", "language": "javascript"},
        ),
        Document(
            page_content="Machine learning algorithms can identify patterns in large datasets.",
            metadata={"category": "AI", "topic": "machine-learning"},
        ),
        Document(
            page_content="Neural networks are inspired by the human brain and used in deep learning.",
            metadata={"category": "AI", "topic": "deep-learning"},
        ),
        Document(
            page_content="Cats are independent pets that enjoy napping and hunting mice.",
            metadata={"category": "animals", "type": "mammals"},
        ),
        Document(
            page_content="Dogs are loyal companions that love playing fetch and going for walks.",
            metadata={"category": "animals", "type": "mammals"},
        ),
    ]

    print(f"📚 Creating vector store with {len(docs)} documents...\n")

    # Create vector store
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Vector store created!\n")
    print("=" * 80 + "\n")

    # Perform semantic searches
    searches = [
        {"query": "programming languages for AI", "k": 2},
        {"query": "pets that need exercise", "k": 2},
        {"query": "building websites", "k": 2},
        {"query": "understanding data patterns", "k": 2},
    ]

    for search in searches:
        query = search["query"]
        k = search["k"]
        print(f'🔍 Search: "{query}" (top {k} results)\n')

        results = vector_store.similarity_search(query, k=k)

        for i, doc in enumerate(results):
            print(f"   {i + 1}. {doc.page_content}")
            print(f"      Category: {doc.metadata.get('category')}\n")

        print("─" * 80 + "\n")

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Vector stores enable fast similarity search over documents")
    print("   - Semantic search finds relevant content even without exact keyword matches")
    print("   - Metadata helps categorize and filter results")


if __name__ == "__main__":
    main()

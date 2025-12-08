"""
Sample: Chunk Optimizer

Compares different chunking strategies to find optimal settings
for your documents.

Run: python 07-documents-embeddings-semantic-search/samples/chunk_optimizer.py
"""

import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_embeddings_endpoint():
    """Get the Azure OpenAI endpoint, removing /openai/v1 suffix if present."""
    endpoint = os.getenv("AI_ENDPOINT", "")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "")
    return endpoint


# Sample text for testing
SAMPLE_TEXT = """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers 
to learn from data without being explicitly programmed. There are three main 
types of machine learning: supervised learning, unsupervised learning, and 
reinforcement learning.

Supervised Learning

In supervised learning, the algorithm learns from labeled training data. 
Common applications include classification (categorizing emails as spam or not) 
and regression (predicting house prices based on features).

Unsupervised Learning

Unsupervised learning works with unlabeled data to find hidden patterns. 
Clustering algorithms group similar data points together, while dimensionality 
reduction techniques simplify complex datasets.

Reinforcement Learning

Reinforcement learning trains agents through trial and error with rewards and 
penalties. This approach powers game-playing AI and autonomous systems.
"""


def test_chunking_strategy(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings,
    query: str,
) -> dict:
    """Test a specific chunking strategy and return metrics."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = splitter.create_documents([text])

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    results = vector_store.similarity_search_with_score(query, k=1)

    return {
        "num_chunks": len(docs),
        "avg_chunk_size": sum(len(d.page_content) for d in docs) / len(docs),
        "top_score": results[0][1] if results else 0,
        "top_content": results[0][0].page_content[:50] + "..." if results else "",
    }


def main():
    print("⚙️ Chunk Optimizer\n")
    print("=" * 80 + "\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    # Different chunking strategies to test
    strategies = [
        {"chunk_size": 100, "chunk_overlap": 0, "name": "Small, No Overlap"},
        {"chunk_size": 100, "chunk_overlap": 20, "name": "Small, With Overlap"},
        {"chunk_size": 200, "chunk_overlap": 0, "name": "Medium, No Overlap"},
        {"chunk_size": 200, "chunk_overlap": 40, "name": "Medium, With Overlap"},
        {"chunk_size": 500, "chunk_overlap": 0, "name": "Large, No Overlap"},
        {"chunk_size": 500, "chunk_overlap": 100, "name": "Large, With Overlap"},
    ]

    query = "What is supervised learning?"

    print(f'Query: "{query}"\n')
    print("Testing different chunking strategies...\n")

    for strategy in strategies:
        result = test_chunking_strategy(
            SAMPLE_TEXT,
            strategy["chunk_size"],
            strategy["chunk_overlap"],
            embeddings,
            query,
        )

        print(f"📊 {strategy['name']}")
        print(f"   Chunks: {result['num_chunks']}, Avg size: {result['avg_chunk_size']:.0f}")
        print(f"   Top score: {result['top_score']:.4f}")
        print(f"   Best match: {result['top_content']}\n")

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Smaller chunks = more precise matches but may lose context")
    print("   - Overlap helps preserve context across chunk boundaries")
    print("   - Optimal chunk size depends on your content and queries")


if __name__ == "__main__":
    main()

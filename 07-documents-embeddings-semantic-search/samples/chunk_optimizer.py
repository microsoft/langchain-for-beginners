"""
Sample: Chunk Optimizer

Compares different chunking strategies to find optimal settings
for your documents.

Run: python 07-documents-embeddings-semantic-search/samples/chunk_optimizer.py
"""

import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


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

Unsupervised learning works with unlabeled data. The algorithm tries to find 
hidden patterns or structures. Clustering is a common technique, grouping 
similar data points together.

Deep Learning

Deep learning uses neural networks with many layers. These networks can 
automatically learn hierarchical features from data. Applications include 
image recognition, natural language processing, and speech recognition.
"""


def test_chunking_strategy(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    query: str,
    embeddings: OpenAIEmbeddings,
) -> tuple[int, list]:
    """Test a chunking strategy and return results."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = splitter.create_documents([text])

    if len(chunks) == 0:
        return 0, []

    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    results = vector_store.similarity_search_with_score(query, k=2)

    return len(chunks), results


def main():
    print("⚙️ Chunk Optimizer\n")
    print("=" * 80 + "\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Different chunking strategies to test
    strategies = [
        {"chunk_size": 100, "chunk_overlap": 0, "name": "Small, No Overlap"},
        {"chunk_size": 100, "chunk_overlap": 20, "name": "Small, 20% Overlap"},
        {"chunk_size": 200, "chunk_overlap": 50, "name": "Medium, 25% Overlap"},
        {"chunk_size": 400, "chunk_overlap": 100, "name": "Large, 25% Overlap"},
    ]

    query = "What are neural networks used for?"

    print(f'Testing query: "{query}"\n')
    print("=" * 80 + "\n")

    for strategy in strategies:
        print(f"📊 Strategy: {strategy['name']}")
        print(f"   Chunk Size: {strategy['chunk_size']}, Overlap: {strategy['chunk_overlap']}")
        print("─" * 80)

        num_chunks, results = test_chunking_strategy(
            SAMPLE_TEXT,
            strategy["chunk_size"],
            strategy["chunk_overlap"],
            query,
            embeddings,
        )

        print(f"   Total Chunks: {num_chunks}")

        if results:
            best_score = results[0][1]
            best_content = results[0][0].page_content[:100] + "..."
            print(f"   Best Match Score: {best_score:.4f}")
            print(f'   Best Match: "{best_content}"')

        print("\n")

    print("=" * 80)
    print("\n💡 Optimization Tips:")
    print("   1. Smaller chunks = more precise matches but less context")
    print("   2. Larger chunks = more context but might dilute relevance")
    print("   3. Overlap helps preserve context at chunk boundaries")
    print("   4. Best strategy depends on your content and query types")


if __name__ == "__main__":
    main()

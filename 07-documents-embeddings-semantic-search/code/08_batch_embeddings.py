"""
Batch Embeddings for Efficiency

Run: python 07-documents-embeddings-semantic-search/code/08_batch_embeddings.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What's the maximum batch size I can use with embed_documents?"
- "How do I handle rate limiting when embedding large document collections?"
"""

import os
import time

from dotenv import load_dotenv
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
    print("⚡ Batch Embeddings Example\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties",
        "Supervised learning uses labeled training data",
        "Unsupervised learning finds patterns in unlabeled data",
        "Transfer learning applies knowledge from one task to another",
    ]

    print(f"📝 Processing {len(texts)} texts...\n")

    # Method 1: Batch embedding (recommended)
    print("Method 1: Batch embedding with embed_documents()")
    start_time = time.time()
    batch_embeddings = embeddings.embed_documents(texts)
    batch_time = time.time() - start_time

    print(f"   ✅ Created {len(batch_embeddings)} embeddings in {batch_time:.2f}s")
    print(f"   Dimensions per embedding: {len(batch_embeddings[0])}\n")

    # Method 2: Individual embedding (for comparison)
    print("Method 2: Individual embedding with embed_query()")
    start_time = time.time()
    individual_embeddings = []
    for text in texts:
        embedding = embeddings.embed_query(text)
        individual_embeddings.append(embedding)
    individual_time = time.time() - start_time

    print(f"   ✅ Created {len(individual_embeddings)} embeddings in {individual_time:.2f}s\n")

    # Compare performance
    print("=" * 80 + "\n")
    print("📊 Performance Comparison:\n")
    print(f"   Batch method:      {batch_time:.2f}s")
    print(f"   Individual method: {individual_time:.2f}s")

    if individual_time > batch_time:
        speedup = individual_time / batch_time
        print(f"\n   🚀 Batch processing is {speedup:.1f}x faster!")
    else:
        print(f"\n   Note: For small batches, performance may be similar")

    print("\n" + "=" * 80)
    print("\n💡 Key Insights:")
    print("   - Use embed_documents() for multiple texts (batching)")
    print("   - Use embed_query() for single queries")
    print("   - Batching reduces API calls and improves efficiency")
    print("   - Consider rate limits when processing large datasets")


if __name__ == "__main__":
    main()

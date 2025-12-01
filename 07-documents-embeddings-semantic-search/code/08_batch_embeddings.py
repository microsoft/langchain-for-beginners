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
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def main():
    print("⚡ Batch Embeddings Example\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
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

    # Method 1: Individual embeddings (slower)
    print("1️⃣  Creating embeddings one-by-one (SLOW):")
    start_time = time.time()

    individual_embeddings = []
    for text in texts:
        embedding = embeddings.embed_query(text)
        individual_embeddings.append(embedding)

    individual_time = time.time() - start_time
    print(f"   Time: {individual_time:.2f}s")
    print(f"   Created {len(individual_embeddings)} embeddings\n")

    # Method 2: Batch embeddings (faster!)
    print("2️⃣  Creating embeddings in batch (FAST):")
    start_time = time.time()

    batch_embeddings = embeddings.embed_documents(texts)

    batch_time = time.time() - start_time
    print(f"   Time: {batch_time:.2f}s")
    print(f"   Created {len(batch_embeddings)} embeddings\n")

    print("=" * 80)
    print("\n📊 Embedding Details:")
    print(f"   Dimensions per embedding: {len(batch_embeddings[0])}")
    print(f"   Total vectors created: {len(batch_embeddings)}")
    sample = ", ".join(f"{n:.4f}" for n in batch_embeddings[0][:5])
    print(f"   First vector sample: [{sample}...]")

    print("\n💡 Key Takeaways:")
    print("   - Batch processing is typically faster")
    print("   - Reduces API calls (lower costs)")
    print("   - Always use embed_documents() for multiple texts")
    print("   - Both methods produce identical embeddings")

    # Verify they're the same
    print("\n✅ Verification:")
    match = all(
        abs(a - b) < 0.0001
        for a, b in zip(individual_embeddings[0], batch_embeddings[0])
    )
    print(f"   Individual vs Batch results match: {'YES' if match else 'NO'}")


if __name__ == "__main__":
    main()

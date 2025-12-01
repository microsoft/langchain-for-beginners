"""
Basic Embeddings

Run: python 07-documents-embeddings-semantic-search/code/05_basic_embeddings.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What is the cosine_similarity function doing mathematically?"
- "Can I use different embedding models and how do they compare?"
"""

import math
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (mag_a * mag_b)


def main():
    print("🔢 Basic Embeddings Example\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create embeddings for different texts
    texts = [
        "LangChain makes building AI apps easier",
        "LangChain simplifies AI application development",
        "I love eating pizza for dinner",
        "The weather is sunny today",
    ]

    print("Creating embeddings for texts...\n")

    all_embeddings = embeddings.embed_documents(texts)

    print(f"✅ Created {len(all_embeddings)} embeddings")
    print(f"   Each embedding has {len(all_embeddings[0])} dimensions\n")

    # Show first embedding details
    print("First embedding (first 10 values):")
    print(all_embeddings[0][:10])
    print("\n" + "=" * 80 + "\n")

    # Compare similarities
    print("📊 Similarity Comparisons:\n")

    pairs = [
        (0, 1, "LangChain vs LangChain (similar meaning)"),
        (0, 2, "LangChain vs Pizza (different topics)"),
        (0, 3, "LangChain vs Weather (different topics)"),
        (2, 3, "Pizza vs Weather (both different from LangChain)"),
    ]

    for i, j, description in pairs:
        similarity = cosine_similarity(all_embeddings[i], all_embeddings[j])
        print(f"{description}:")
        print(f"   Score: {similarity:.4f}")
        print(f'   Texts: "{texts[i]}" vs "{texts[j]}"\n')

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Similar meanings → High similarity scores (>0.8)")
    print("   - Different topics → Low similarity scores (<0.5)")
    print("   - Embeddings capture semantic meaning, not just keywords!")


if __name__ == "__main__":
    main()

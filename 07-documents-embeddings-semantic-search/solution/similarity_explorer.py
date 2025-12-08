"""
Chapter 7 Assignment Solution: Challenge 1
Similarity Explorer

Run: python 07-documents-embeddings-semantic-search/solution/similarity_explorer.py
"""

import math
import os

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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (mag_a * mag_b)


SENTENCES = [
    "I love programming in JavaScript",
    "JavaScript is my favorite language",
    "Python is great for data science",
    "Machine learning uses Python often",
    "I enjoy coding web applications",
    "Dogs are loyal pets",
    "Cats are independent animals",
    "Pets bring joy to families",
    "The weather is sunny today",
    "It's raining outside",
]


def main():
    print("🔬 Similarity Explorer\n")
    print("=" * 80 + "\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    print("📝 Creating embeddings for 10 sentences...\n")

    all_embeddings = embeddings.embed_documents(SENTENCES)

    print(f"✅ Created {len(all_embeddings)} embeddings")
    print(f"   Dimensions: {len(all_embeddings[0])}\n")

    print("=" * 80 + "\n")

    # Calculate all pairs
    similarities: list[dict] = []

    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            score = cosine_similarity(all_embeddings[i], all_embeddings[j])
            similarities.append(
                {
                    "pair": f'"{SENTENCES[i]}" <-> "{SENTENCES[j]}"',
                    "score": score,
                    "i": i,
                    "j": j,
                }
            )

    # Sort by score
    similarities.sort(key=lambda x: x["score"], reverse=True)

    # Most similar pair
    print("🏆 MOST SIMILAR PAIR:\n")
    print("─" * 80)
    most_similar = similarities[0]
    print(f"Score: {most_similar['score']:.4f}")
    print(f'\n"{SENTENCES[most_similar["i"]]}"')
    print(f'"{SENTENCES[most_similar["j"]]}"\n')

    # Least similar pair
    print("─" * 80 + "\n")
    print("❄️  LEAST SIMILAR PAIR:\n")
    print("─" * 80)
    least_similar = similarities[-1]
    print(f"Score: {least_similar['score']:.4f}")
    print(f'\n"{SENTENCES[least_similar["i"]]}"')
    print(f'"{SENTENCES[least_similar["j"]]}"\n')

    # High similarity pairs (> 0.8)
    print("─" * 80 + "\n")
    print("⭐ HIGH SIMILARITY PAIRS (Score > 0.8):\n")
    print("─" * 80)

    high_similarity = [s for s in similarities if s["score"] > 0.8]

    if not high_similarity:
        print("No pairs with similarity > 0.8\n")
    else:
        for sim in high_similarity:
            print(f"\nScore: {sim['score']:.4f}")
            print(f'"{SENTENCES[sim["i"]]}"')
            print(f'"{SENTENCES[sim["j"]]}"')
        print()

    # Topic clustering
    print("─" * 80 + "\n")
    print("📊 TOPIC CLUSTERING:\n")
    print("─" * 80 + "\n")

    # Programming cluster (0, 1, 2, 3, 4)
    print("💻 Programming/Tech Topic:")
    for i in [0, 1, 2, 3, 4]:
        print(f"   {i + 1}. {SENTENCES[i]}")

    print("\n🐾 Pets Topic:")
    for i in [5, 6, 7]:
        print(f"   {i + 1}. {SENTENCES[i]}")

    print("\n☀️  Weather Topic:")
    for i in [8, 9]:
        print(f"   {i + 1}. {SENTENCES[i]}")

    print("\n" + "=" * 80)
    print("\n✅ Analysis complete!")
    print("\n💡 Key Observations:")
    print("   - Sentences about the same topic cluster together")
    print("   - JavaScript/programming sentences are most similar to each other")
    print("   - Unrelated topics (e.g., programming vs weather) have low similarity")
    print("   - Embeddings capture semantic meaning, not just keyword matching!")


if __name__ == "__main__":
    main()

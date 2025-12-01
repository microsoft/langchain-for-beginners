"""
Embedding Relationships - Vector Math Demo

This example demonstrates how embeddings capture semantic relationships
that can be manipulated through vector arithmetic.

Run: python 07-documents-embeddings-semantic-search/code/09_embedding_relationships.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does vector arithmetic like 'King - Man + Woman = Queen' actually work?"
- "What real-world applications benefit from embedding relationships?"
"""

import math
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    return dot_product / (magnitude_a * magnitude_b)


def subtract_vectors(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """Subtract two vectors."""
    return [a - b for a, b in zip(vec_a, vec_b)]


def add_vectors(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """Add two vectors."""
    return [a + b for a, b in zip(vec_a, vec_b)]


def main():
    print("🔬 Embedding Relationships: Vector Math Demo\n")
    print("This demonstrates how embeddings capture semantic relationships")
    print("that can be manipulated mathematically.\n")
    print("=" * 70 + "\n")

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # ============================================================================
    # Example 1: Animal Life Stages
    # Demonstrating: Puppy - Dog + Cat ≈ Kitten
    # ============================================================================

    print("🐶 Example 1: Animal Life Stages")
    print("─" * 70)
    print("\nTesting: Embedding('Puppy') - Embedding('Dog') + Embedding('Cat')")
    print("Expected result: Should be similar to Embedding('Kitten')\n")

    # Generate embeddings for animals and their young
    animal_texts = ["Puppy", "Dog", "Cat", "Kitten"]
    animal_embeds = embeddings.embed_documents(animal_texts)
    puppy_embed, dog_embed, cat_embed, kitten_embed = animal_embeds

    # Perform vector arithmetic: Puppy - Dog + Cat
    puppy_minus_dog = subtract_vectors(puppy_embed, dog_embed)
    result1 = add_vectors(puppy_minus_dog, cat_embed)

    # Calculate similarity with Kitten
    similarity_to_kitten = cosine_similarity(result1, kitten_embed)

    print(f"✅ Similarity to 'Kitten': {similarity_to_kitten * 100:.2f}%")
    print("\nWhat this means:")
    print("  • Puppy is to Dog as Kitten is to Cat")
    print("  • The vectors encode 'species' and 'life stage' as separate dimensions")
    print("  • Subtracting 'Dog' removes the adult dog, adding 'Cat' finds the young cat")

    # Show comparison with unrelated animal
    bird_embed = embeddings.embed_query("Bird")
    similarity_to_bird = cosine_similarity(result1, bird_embed)

    print(f"\n📊 Comparison: Similarity to 'Bird': {similarity_to_bird * 100:.2f}%")
    print("   (Lower than Kitten - Bird is a different species, not a young cat)\n")

    print("=" * 70 + "\n")

    # ============================================================================
    # Example 2: Cultural Food Relationships
    # Demonstrating: pizza - Italy + Japan ≈ sushi
    # ============================================================================

    print("🍕 Example 2: Cultural Food Relationships")
    print("─" * 70)
    print("\nTesting: Embedding('pizza') - Embedding('Italy') + Embedding('Japan')")
    print("Expected result: Should be similar to Embedding('sushi')\n")

    # Generate embeddings for food and countries
    food_texts = ["pizza", "Italy", "Japan", "sushi"]
    food_embeds = embeddings.embed_documents(food_texts)
    pizza_embed, italy_embed, japan_embed, sushi_embed = food_embeds

    # Perform vector arithmetic: pizza - Italy + Japan
    pizza_minus_italy = subtract_vectors(pizza_embed, italy_embed)
    result2 = add_vectors(pizza_minus_italy, japan_embed)

    # Calculate similarity with sushi
    similarity_to_sushi = cosine_similarity(result2, sushi_embed)

    print(f"✅ Similarity to 'sushi': {similarity_to_sushi * 100:.2f}%")
    print("\nWhat this means:")
    print("  • Pizza is to Italy as sushi is to Japan")
    print("  • The embeddings understand cultural food associations")
    print("  • Subtracting 'Italy' removes the country, adding 'Japan' finds Japan's iconic food")

    # Show comparison with unrelated food
    burger_embed = embeddings.embed_query("hamburger")
    similarity_to_burger = cosine_similarity(result2, burger_embed)

    print(f"\n📊 Comparison: Similarity to 'hamburger': {similarity_to_burger * 100:.2f}%")
    print("   (Lower than sushi, as expected - hamburger is more associated with USA)\n")

    print("=" * 70 + "\n")

    # ============================================================================
    # Example 3: Synonym Clustering
    # Demonstrating: Similar words have similar embeddings
    # ============================================================================

    print("😊 Example 3: Synonym Clustering")
    print("─" * 70)
    print("\nTesting similarity between synonyms:\n")

    # Generate embeddings for synonyms
    emotion_texts = ["happy", "joyful", "cheerful", "sad"]
    emotion_embeds = embeddings.embed_documents(emotion_texts)
    happy_embed, joyful_embed, cheerful_embed, sad_embed = emotion_embeds

    # Calculate similarities
    happy_joyful = cosine_similarity(happy_embed, joyful_embed)
    happy_cheerful = cosine_similarity(happy_embed, cheerful_embed)
    happy_sad = cosine_similarity(happy_embed, sad_embed)

    print(f"Similarity: 'happy' ↔ 'joyful'   = {happy_joyful * 100:.2f}% ✅ (synonyms)")
    print(f"Similarity: 'happy' ↔ 'cheerful' = {happy_cheerful * 100:.2f}% ✅ (synonyms)")
    print(f"Similarity: 'happy' ↔ 'sad'      = {happy_sad * 100:.2f}% ❌ (opposites)")

    print("\nWhat this means:")
    print("  • Words with similar meanings have similar embeddings")
    print("  • They cluster close together in vector space")
    print("  • Opposite meanings have lower similarity scores")

    print("\n" + "=" * 70 + "\n")

    # ============================================================================
    # Summary
    # ============================================================================

    print("🎓 Key Takeaways:")
    print("─" * 70)
    print("\n1. Embeddings capture semantic relationships:")
    print("   • Puppy:Dog :: Kitten:Cat (animal life stages)")
    print("   • pizza:Italy :: sushi:Japan (cultural foods)")
    print("")
    print("2. Vector arithmetic works on meanings:")
    print("   • Adding/subtracting embeddings preserves relationships")
    print("   • The math 'understands' concepts like species, life stage, country, food")
    print("")
    print("3. Synonyms cluster together:")
    print("   • Similar meanings = nearby in vector space")
    print("   • Different meanings = farther apart")
    print("")
    print("4. This enables powerful applications:")
    print("   • Semantic search (find similar meanings, not just keywords)")
    print("   • Analogy completion (A:B :: C:?)")
    print("   • Document clustering (group by topic)")
    print("   • Recommendation systems (find similar items)")
    print("\n✅ These relationships emerge from training on massive text corpora!")


if __name__ == "__main__":
    main()

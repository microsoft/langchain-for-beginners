"""
Sample: Embedding Visualizer (Conceptual)

Demonstrates how to visualize embedding relationships
using dimensionality reduction techniques.

Note: In a full implementation, you would use matplotlib, plotly,
or other visualization libraries. This sample shows the concepts.

Run: python 07-documents-embeddings-semantic-search/samples/embedding_visualizer.py
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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0


def simple_pca_2d(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """
    Simplified PCA-like projection to 2D for visualization.
    Note: This is a simplified demonstration, not true PCA.
    In practice, use sklearn.decomposition.PCA or UMAP.
    """
    # Use first two principal components (simplified)
    return [(v[0], v[1]) for v in vectors]


def print_similarity_matrix(texts: list[str], embeddings_list: list[list[float]]):
    """Print a similarity matrix for the given texts."""
    n = len(texts)

    # Print header
    print("\n" + " " * 20, end="")
    for i in range(n):
        print(f"{i + 1:>8}", end="")
    print()

    # Print matrix
    for i in range(n):
        label = texts[i][:18].ljust(18)
        print(f"{i + 1}. {label}", end=" ")
        for j in range(n):
            sim = cosine_similarity(embeddings_list[i], embeddings_list[j])
            print(f"{sim:>8.3f}", end="")
        print()


def main():
    print("📊 Embedding Visualizer\n")
    print("=" * 80 + "\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    # Texts in semantic clusters
    texts = [
        # Cluster 1: Animals
        "Dogs are loyal pets",
        "Cats are independent",
        "Hamsters are small",
        # Cluster 2: Programming
        "Python is popular",
        "JavaScript is versatile",
        "Rust is fast",
        # Cluster 3: Food
        "Pizza is delicious",
        "Sushi is healthy",
        "Tacos are tasty",
    ]

    print("Creating embeddings for 9 texts across 3 categories...\n")

    all_embeddings = embeddings.embed_documents(texts)

    print("✅ Embeddings created!\n")

    # Show similarity matrix
    print("📊 Similarity Matrix:")
    print("   (Higher values = more similar)\n")

    print_similarity_matrix(texts, all_embeddings)

    # Show cluster analysis
    print("\n" + "=" * 80 + "\n")
    print("📈 Cluster Analysis:\n")

    clusters = [
        ("Animals", [0, 1, 2]),
        ("Programming", [3, 4, 5]),
        ("Food", [6, 7, 8]),
    ]

    for cluster_name, indices in clusters:
        # Calculate average within-cluster similarity
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = cosine_similarity(all_embeddings[indices[i]], all_embeddings[indices[j]])
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        print(f"   {cluster_name}: avg within-cluster similarity = {avg_sim:.3f}")

    # Cross-cluster comparison
    print("\n   Cross-cluster similarities (should be lower):")
    cross_pairs = [
        ("Animals vs Programming", 0, 3),
        ("Animals vs Food", 0, 6),
        ("Programming vs Food", 3, 6),
    ]

    for name, i, j in cross_pairs:
        sim = cosine_similarity(all_embeddings[i], all_embeddings[j])
        print(f"   {name}: {sim:.3f}")

    print("\n" + "=" * 80)
    print("\n💡 Key Insights:")
    print("   - Items in the same category have higher similarity scores")
    print("   - Cross-category similarities are lower")
    print("   - Embeddings naturally cluster by semantic meaning")


if __name__ == "__main__":
    main()

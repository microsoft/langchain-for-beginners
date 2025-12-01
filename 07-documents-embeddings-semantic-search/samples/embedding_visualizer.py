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
from langchain_openai import OpenAIEmbeddings

load_dotenv()


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
    if not vectors:
        return []

    # Simple projection using first two "principal-like" dimensions
    # Just for demonstration - real PCA would compute eigenvectors
    n_dims = min(len(vectors[0]), 2)
    return [(v[0], v[1] if n_dims > 1 else 0) for v in vectors]


def ascii_scatter_plot(
    points: list[tuple[float, float]], labels: list[str], width: int = 60, height: int = 20
) -> str:
    """Create a simple ASCII scatter plot."""
    if not points:
        return "No data to plot"

    # Normalize to grid
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Place points
    for i, (x, y) in enumerate(points):
        col = int((x - x_min) / x_range * (width - 1))
        row = int((1 - (y - y_min) / y_range) * (height - 1))  # Invert y
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        marker = str(i + 1) if i < 9 else "*"
        grid[row][col] = marker

    # Build plot
    plot_lines = ["┌" + "─" * width + "┐"]
    for row in grid:
        plot_lines.append("│" + "".join(row) + "│")
    plot_lines.append("└" + "─" * width + "┘")

    return "\n".join(plot_lines)


def main():
    print("📊 Embedding Visualizer\n")
    print("=" * 80 + "\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Texts in semantic clusters
    texts = [
        # Cluster 1: Animals
        "The cat sleeps on the couch",  # 1
        "Dogs love to play fetch",  # 2
        "Cats and dogs are popular pets",  # 3
        # Cluster 2: Programming
        "Python is a programming language",  # 4
        "JavaScript runs in browsers",  # 5
        "Coding is fun and creative",  # 6
        # Cluster 3: Food
        "Pizza is my favorite food",  # 7
        "I love eating sushi for dinner",  # 8
    ]

    print("📝 Texts to visualize:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")
    print()

    print("⏳ Generating embeddings...\n")
    vectors = embeddings.embed_documents(texts)
    print(f"✅ Generated {len(vectors)} embeddings of {len(vectors[0])} dimensions\n")

    # Project to 2D
    points_2d = simple_pca_2d(vectors)

    print("=" * 80)
    print("\n📈 2D Projection (ASCII Plot):\n")
    plot = ascii_scatter_plot(points_2d, texts)
    print(plot)
    print()

    print("Legend:")
    for i, text in enumerate(texts, 1):
        marker = str(i) if i <= 9 else "*"
        print(f"   {marker} = {text[:40]}...")
    print()

    # Show similarity matrix
    print("=" * 80)
    print("\n📊 Similarity Matrix:\n")

    # Print header
    print("     ", end="")
    for i in range(len(texts)):
        print(f"  {i + 1}  ", end="")
    print()

    for i, vec1 in enumerate(vectors):
        print(f"  {i + 1}  ", end="")
        for j, vec2 in enumerate(vectors):
            sim = cosine_similarity(vec1, vec2)
            # Color code: high similarity in bright
            if sim > 0.9:
                print(f" .██ ", end="")
            elif sim > 0.7:
                print(f" .▓▓ ", end="")
            elif sim > 0.5:
                print(f" .▒▒ ", end="")
            else:
                print(f" .░░ ", end="")
        print()

    print()
    print("   Legend: ██ > 0.9  ▓▓ > 0.7  ▒▒ > 0.5  ░░ < 0.5")

    print("\n" + "=" * 80)
    print("\n💡 Visualization Insights:")
    print("   • Semantically similar texts cluster together")
    print("   • Animals (1-3), Programming (4-6), Food (7-8) form groups")
    print("   • High similarity scores within clusters")
    print("   • For real visualization, use matplotlib or plotly with UMAP/t-SNE")


if __name__ == "__main__":
    main()

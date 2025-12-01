"""
Sample: Smart Chunker

Demonstrates intelligent text splitting that respects document structure
like headers, paragraphs, and sentences.

Run: python 07-documents-embeddings-semantic-search/samples/smart_chunker.py
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def main():
    print("🧠 Smart Chunker - Structure-Aware Text Splitting\n")
    print("=" * 80 + "\n")

    # Sample markdown document
    markdown_doc = """
# Machine Learning Guide

## Introduction

Machine learning is a powerful technology that enables computers to learn from data.
It has revolutionized many industries including healthcare, finance, and technology.

## Types of Learning

### Supervised Learning

Supervised learning uses labeled data to train models. Examples include:
- Classification: Categorizing emails as spam or not spam
- Regression: Predicting house prices based on features

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Common techniques:
- Clustering: Grouping similar customers together
- Dimensionality reduction: Simplifying complex datasets

## Best Practices

Always split your data into training and testing sets. Use cross-validation
to ensure your model generalizes well to new data.
"""

    print("📄 Original Markdown Document:")
    print("─" * 80)
    print(markdown_doc[:300] + "...")
    print("\n" + "=" * 80 + "\n")

    # Method 1: Markdown-aware splitting
    print("📋 Method 1: Markdown Header Splitting\n")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = md_splitter.split_text(markdown_doc)

    for i, chunk in enumerate(md_chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Metadata: {chunk.metadata}")
        content_preview = chunk.page_content[:80].replace("\n", " ")
        print(f"  Content: {content_preview}...")
        print()

    print("=" * 80 + "\n")

    # Method 2: Character splitting with structure awareness
    print("📋 Method 2: Recursive Character Splitting\n")

    # The recursive splitter tries to split on these separators in order
    # This preserves structure by preferring paragraph breaks over mid-sentence
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=[
            "\n\n",  # Paragraphs first
            "\n",  # Then newlines
            ". ",  # Then sentences
            " ",  # Then words
            "",  # Finally characters
        ],
    )

    char_chunks = char_splitter.create_documents([markdown_doc])

    for i, chunk in enumerate(char_chunks[:4]):  # Show first 4
        content_preview = chunk.page_content[:80].replace("\n", " ")
        print(f"Chunk {i + 1}: {content_preview}...")
    print(f"... and {len(char_chunks) - 4} more chunks")

    print("\n" + "=" * 80)
    print("\n💡 Key Insights:")
    print("   • Markdown splitting preserves document hierarchy in metadata")
    print("   • Recursive splitting respects natural text boundaries")
    print("   • Choose based on your document format and query needs")
    print("   • Metadata can be used for filtering during retrieval")


if __name__ == "__main__":
    main()

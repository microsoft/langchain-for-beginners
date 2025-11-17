"""
Basic Embeddings Example
Run: python 07-documents-embeddings-semantic-search/code/01_basic_embeddings.py

This example demonstrates how to create and compare embeddings.
"""

import os

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

def main():
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    print("🔢 Embeddings & Similarity Example\n")
    print("=" * 80)

    # Create embeddings for different texts
    texts = [
        "The cat sat on the mat",
        "The feline rested on the rug",
        "Python is a programming language",
    ]

    print("\n📝 Creating embeddings for:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")

    # Generate embeddings
    text_embeddings = embeddings.embed_documents(texts)

    # Calculate similarities
    print("\n🔍 Similarity Scores:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = cosine_similarity(
                [text_embeddings[i]], 
                [text_embeddings[j]]
            )[0][0]
            print(f"   Text {i+1} ↔ Text {j+1}: {similarity:.4f}")

    print("\n" + "=" * 80)
    print("\n✅ Embeddings capture semantic meaning!")
    print("💡 Similar meanings have higher similarity scores")

if __name__ == "__main__":
    main()

"""
Similarity Search with Scores

Run: python 07-documents-embeddings-semantic-search/code/07_similarity_scores.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What similarity score threshold should I use to filter out irrelevant results?"
- "How does similarity_search_with_score differ from the regular similarity_search method?"
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def main():
    print("📊 Similarity Search with Scores\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create a diverse set of documents
    docs = [
        Document(
            page_content="Python is excellent for data science and machine learning applications.",
            metadata={"category": "programming", "language": "python"},
        ),
        Document(
            page_content="JavaScript powers interactive web applications and modern frontends.",
            metadata={"category": "programming", "language": "javascript"},
        ),
        Document(
            page_content="Machine learning algorithms identify patterns in large datasets.",
            metadata={"category": "AI", "topic": "ml"},
        ),
        Document(
            page_content="Cats are independent pets that enjoy lounging in sunny spots.",
            metadata={"category": "animals", "type": "pets"},
        ),
        Document(
            page_content="Dogs are loyal companions that love outdoor activities and play.",
            metadata={"category": "animals", "type": "pets"},
        ),
        Document(
            page_content="TypeScript adds static typing to JavaScript for safer code.",
            metadata={"category": "programming", "language": "typescript"},
        ),
    ]

    print(f"📚 Creating vector store with {len(docs)} documents...\n")

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Vector store created!\n")
    print("=" * 80)

    # Search with scores
    queries = [
        "programming languages for web development",
        "pets that are good for apartments",
        "understanding data with AI",
    ]

    for query in queries:
        print(f'\n🔍 Query: "{query}"\n')

        results_with_scores = vector_store.similarity_search_with_score(query, k=3)

        for index, (doc, score) in enumerate(results_with_scores):
            print(f"{index + 1}. Score: {score:.4f}")
            print(f"   Text: {doc.page_content}")
            print(f"   Category: {doc.metadata.get('category')}")

            # Interpret score
            if score > 0.85:
                interpretation = "🎯 Excellent match"
            elif score > 0.75:
                interpretation = "✅ Good match"
            elif score > 0.65:
                interpretation = "⚠️  Moderate match"
            else:
                interpretation = "❌ Weak match"

            print(f"   {interpretation}\n")

        print("─" * 80)

    print("\n💡 Understanding Similarity Scores:")
    print("   - Closer to 1.0 = More similar")
    print("   - Closer to 0.0 = Less similar")
    print("   - Typically use threshold (e.g., > 0.7) to filter results")


if __name__ == "__main__":
    main()

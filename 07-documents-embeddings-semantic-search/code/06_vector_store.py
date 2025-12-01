"""
Vector Store and Semantic Search

Run: python 07-documents-embeddings-semantic-search/code/06_vector_store.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What's the difference between InMemoryVectorStore and persistent stores like Pinecone?"
- "Can I save and load a vector store to avoid recomputing embeddings?"
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def main():
    print("🗄️  Vector Store and Semantic Search\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create documents about different topics
    docs = [
        Document(
            page_content="Python is a popular programming language for data science and machine learning.",
            metadata={"category": "programming", "language": "python"},
        ),
        Document(
            page_content="JavaScript is widely used for web development and building interactive websites.",
            metadata={"category": "programming", "language": "javascript"},
        ),
        Document(
            page_content="Machine learning algorithms can identify patterns in large datasets.",
            metadata={"category": "AI", "topic": "machine-learning"},
        ),
        Document(
            page_content="Neural networks are inspired by the human brain and used in deep learning.",
            metadata={"category": "AI", "topic": "deep-learning"},
        ),
        Document(
            page_content="Cats are independent pets that enjoy napping and hunting mice.",
            metadata={"category": "animals", "type": "mammals"},
        ),
        Document(
            page_content="Dogs are loyal companions that love playing fetch and going for walks.",
            metadata={"category": "animals", "type": "mammals"},
        ),
    ]

    print(f"📚 Creating vector store with {len(docs)} documents...\n")

    # Create vector store
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Vector store created!\n")
    print("=" * 80 + "\n")

    # Perform semantic searches
    searches = [
        {"query": "programming languages for AI", "k": 2},
        {"query": "pets that need exercise", "k": 2},
        {"query": "building websites", "k": 2},
        {"query": "understanding data patterns", "k": 2},
    ]

    for search in searches:
        query = search["query"]
        k = search["k"]
        print(f'🔍 Search: "{query}" (top {k} results)\n')

        results = vector_store.similarity_search(query, k=k)

        for i, doc in enumerate(results):
            print(f"   {i + 1}. {doc.page_content}")
            print(f"      Category: {doc.metadata.get('category')}\n")

        print("─" * 80 + "\n")

    # Search with similarity scores
    print("=" * 80)
    print("\n📊 Search with Similarity Scores:\n")

    query = "animals that make good house pets"
    results_with_scores = vector_store.similarity_search_with_score(query, k=4)

    print(f'Query: "{query}"\n')

    for doc, score in results_with_scores:
        print(f"Score: {score:.4f} - {doc.page_content}")

    print("\n" + "=" * 80)
    print("\n💡 Notice:")
    print("   - Results are ranked by semantic similarity")
    print("   - Exact keywords aren't required!")
    print("   - AI understands context and meaning")


if __name__ == "__main__":
    main()

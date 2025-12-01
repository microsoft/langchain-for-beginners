"""
Chapter 7 Assignment Solution: Challenge 2
Semantic Book Search

Run: python 07-documents-embeddings-semantic-search/solution/book_search.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BOOKS = [
    {
        "title": "The AI Revolution",
        "summary": "How artificial intelligence is transforming society and business",
    },
    {
        "title": "JavaScript Mastery",
        "summary": "Complete guide to modern web development with JavaScript",
    },
    {
        "title": "Data Science Handbook",
        "summary": "Statistical analysis and machine learning for beginners",
    },
    {
        "title": "The Startup Playbook",
        "summary": "Building and scaling technology companies from scratch",
    },
    {
        "title": "Mystery at Midnight",
        "summary": "A detective solves crimes in Victorian London",
    },
    {
        "title": "Space Odyssey",
        "summary": "Humans explore distant galaxies and alien civilizations",
    },
    {
        "title": "Cooking Basics",
        "summary": "Essential techniques for home chefs and food enthusiasts",
    },
    {
        "title": "Python for Data",
        "summary": "Using Python for data analysis and visualization",
    },
]


def main():
    print("📚 Semantic Book Search\n")
    print("=" * 80 + "\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    print("📖 Loading books into vector store...\n")

    documents = [
        Document(
            page_content=book["summary"],
            metadata={"title": book["title"]},
        )
        for book in BOOKS
    ]

    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)

    print(f"✅ Loaded {len(BOOKS)} books\n")
    print("=" * 80 + "\n")

    queries = [
        "books about programming",
        "stories set in space",
        "learning about AI and technology",
        "cooking and recipes",
    ]

    for query in queries:
        print(f'🔍 Query: "{query}"\n')
        print("─" * 80)

        results = vector_store.similarity_search_with_score(query, k=3)

        for index, (doc, score) in enumerate(results):
            print(f"\n{index + 1}. {doc.metadata.get('title')}")
            print(f"   Relevance: {score * 100:.1f}%")
            print(f"   Summary: {doc.page_content}")

        print("\n" + "─" * 80 + "\n")

    print("=" * 80)
    print("\n✅ Book search complete!")
    print("\n💡 Notice how semantic search finds:")
    print("   - 'programming' matches JavaScript AND Python books")
    print("   - 'space' finds the Space Odyssey story")
    print("   - 'AI' finds both AI Revolution AND Data Science")
    print("   - Searches by meaning, not just exact keywords!")


if __name__ == "__main__":
    main()

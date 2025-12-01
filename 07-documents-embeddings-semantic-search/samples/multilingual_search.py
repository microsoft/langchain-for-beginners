"""
Sample: Multilingual Semantic Search

Demonstrates that embeddings can find semantically similar content
across different languages.

Run: python 07-documents-embeddings-semantic-search/samples/multilingual_search.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def main():
    print("🌍 Multilingual Semantic Search\n")
    print("=" * 80 + "\n")

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create documents in different languages
    docs = [
        Document(
            page_content="The cat is sleeping on the sofa.",
            metadata={"language": "English"},
        ),
        Document(
            page_content="Le chat dort sur le canapé.",
            metadata={"language": "French"},
        ),
        Document(
            page_content="El gato está durmiendo en el sofá.",
            metadata={"language": "Spanish"},
        ),
        Document(
            page_content="Die Katze schläft auf dem Sofa.",
            metadata={"language": "German"},
        ),
        Document(
            page_content="The dog is running in the park.",
            metadata={"language": "English"},
        ),
        Document(
            page_content="I love eating pizza for dinner.",
            metadata={"language": "English"},
        ),
    ]

    print(f"📚 Creating vector store with {len(docs)} multilingual documents...\n")

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Vector store created!\n")
    print("=" * 80 + "\n")

    # Search in English - should find all cat-related sentences
    query = "feline resting on furniture"
    print(f'🔍 Query (English): "{query}"\n')
    print("─" * 80 + "\n")

    results = vector_store.similarity_search_with_score(query, k=4)

    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Language: {doc.metadata.get('language')}")
        print(f"Content: {doc.page_content}\n")

    print("=" * 80)
    print("\n💡 Amazing Result:")
    print("   The English query 'feline resting on furniture' finds:")
    print("   - The cat is sleeping on the sofa (English)")
    print("   - Le chat dort sur le canapé (French)")
    print("   - El gato está durmiendo en el sofá (Spanish)")
    print("   - Die Katze schläft auf dem Sofa (German)")
    print("\n   Embeddings understand MEANING across languages!")


if __name__ == "__main__":
    main()

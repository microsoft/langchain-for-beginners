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


def main():
    print("🌍 Multilingual Semantic Search\n")
    print("=" * 80 + "\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    # Create documents in different languages
    docs = [
        Document(
            page_content="The cat is sleeping on the sofa.",
            metadata={"language": "English"},
        ),
        Document(
            page_content="El gato está durmiendo en el sofá.",
            metadata={"language": "Spanish"},
        ),
        Document(
            page_content="Le chat dort sur le canapé.",
            metadata={"language": "French"},
        ),
        Document(
            page_content="Die Katze schläft auf dem Sofa.",
            metadata={"language": "German"},
        ),
        Document(
            page_content="I love programming in Python.",
            metadata={"language": "English"},
        ),
        Document(
            page_content="Me encanta programar en Python.",
            metadata={"language": "Spanish"},
        ),
    ]

    print(f"📚 Indexing {len(docs)} documents in multiple languages...\n")

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    print("✅ Documents indexed!\n")
    print("=" * 80 + "\n")

    # Search in different languages
    queries = [
        ("A sleeping cat", "English query about cats"),
        ("Un chat qui dort", "French query about cats"),
        ("Python programming", "English query about programming"),
        ("Programación en Python", "Spanish query about programming"),
    ]

    for query, description in queries:
        print(f'🔍 Query ({description}): "{query}"\n')

        results = vector_store.similarity_search(query, k=3)

        for i, doc in enumerate(results):
            print(f"   {i + 1}. [{doc.metadata['language']}] {doc.page_content}")

        print("\n" + "─" * 80 + "\n")

    print("=" * 80)
    print("\n💡 Key Insights:")
    print("   - Embeddings capture semantic meaning across languages")
    print("   - A query in one language can find relevant content in others")
    print("   - This enables truly multilingual search applications")


if __name__ == "__main__":
    main()

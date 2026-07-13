"""
Host a LangChain agentic RAG agent in Microsoft Foundry.

Run locally:
    python main.py

Then send a non-streaming Responses request to http://localhost:8088/responses.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_azure_ai.agents.hosting import ResponsesHostServer

load_dotenv()


COURSE_NOTES = [
    Document(
        page_content=(
            "LangChain was created in 2022 and became popular for building "
            "LLM applications with models, prompts, tools, retrieval, and agents."
        ),
        metadata={"source": "langchain-history", "topic": "introduction"},
    ),
    Document(
        page_content=(
            "RAG stands for Retrieval Augmented Generation. It retrieves relevant "
            "documents and gives that context to a language model before generating an answer."
        ),
        metadata={"source": "rag-explanation", "topic": "concepts"},
    ),
    Document(
        page_content=(
            "Agentic RAG lets an agent decide when to search documents and when to "
            "answer directly. This avoids unnecessary retrieval for simple questions."
        ),
        metadata={"source": "agentic-rag", "topic": "agents"},
    ),
    Document(
        page_content=(
            "Vector stores hold embeddings for document chunks and support semantic "
            "similarity search over a knowledge base."
        ),
        metadata={"source": "vector-stores", "topic": "retrieval"},
    ),
]


def build_chat_model() -> ChatOpenAI:
    """Create a chat model for local course runs or Foundry hosted deployment."""

    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )


def build_embeddings_model() -> OpenAIEmbeddings:
    """Create an embedding model using the Foundry endpoint and identity."""

    return OpenAIEmbeddings(
        base_url=os.getenv("AI_ENDPOINT"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("AI_API_KEY"),
    )


def build_agent():
    vector_store = InMemoryVectorStore.from_documents(
        COURSE_NOTES,
        build_embeddings_model(),
    )

    @tool
    def search_course_notes(query: str) -> str:
        """Search course notes about LangChain, RAG, agentic RAG, and vector stores. Use this when you need course-specific information."""
        results = vector_store.similarity_search(query, k=3)
        return "\n\n".join(
            f"[{document.metadata['source']}]: {document.page_content}"
            for document in results
        )

    return create_agent(
        build_chat_model(),
        tools=[search_course_notes],
        system_prompt=(
            "You are a helpful agentic RAG assistant for the LangChain for Beginners course. "
            "Use search_course_notes for course-specific questions about LangChain, RAG, "
            "agentic RAG, or vector stores. Answer general questions directly. "
            "When you use course notes, cite the source names in square brackets."
        ),
    )


def main() -> None:
    port = int(os.getenv("PORT", "8088"))
    ResponsesHostServer(build_agent()).run(port=port)


if __name__ == "__main__":
    main()

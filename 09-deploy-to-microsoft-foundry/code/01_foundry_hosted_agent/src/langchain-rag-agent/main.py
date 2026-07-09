"""
Host a LangChain agentic RAG agent in Microsoft Foundry.

Run locally:
    python main.py

Then send a non-streaming Responses request to http://localhost:8088/responses.
"""

import os
import re

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_azure_ai.agents.hosting import ResponsesHostServer

AZURE_AI_SCOPE = "https://ai.azure.com/.default"

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
    foundry_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")

    if foundry_endpoint:
        credential = DefaultAzureCredential()
        project = AIProjectClient(
            endpoint=foundry_endpoint.rstrip("/"),
            credential=credential,
        )
        openai_client = project.get_openai_client()
        token_provider = get_bearer_token_provider(credential, AZURE_AI_SCOPE)
        deployment = (
            os.getenv("FOUNDRY_MODEL_NAME")
            or os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
            or os.getenv("AI_MODEL")
        )

        if not deployment:
            raise RuntimeError("Set FOUNDRY_MODEL_NAME to your Foundry model deployment name.")

        return ChatOpenAI(
            model=deployment,
            base_url=str(openai_client.base_url),
            api_key=token_provider,
        )

    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )


def score_document(query: str, document: Document) -> int:
    """Simple keyword scoring used only to keep the hosting sample compact."""
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    haystack = " ".join(
        [document.page_content, *[str(value) for value in document.metadata.values()]]
    ).lower()
    return sum(1 for term in terms if term in haystack)


@tool
def search_course_notes(query: str) -> str:
    """Search course notes about LangChain, RAG, agentic RAG, and vector stores. Use this when you need course-specific information."""
    ranked_documents = sorted(
        COURSE_NOTES,
        key=lambda document: score_document(query, document),
        reverse=True,
    )
    top_documents = ranked_documents[:3]

    if not any(score_document(query, document) for document in top_documents):
        return "No matching course notes found."

    return "\n\n".join(
        f"[{document.metadata['source']}]: {document.page_content}"
        for document in top_documents
    )


def build_agent():
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
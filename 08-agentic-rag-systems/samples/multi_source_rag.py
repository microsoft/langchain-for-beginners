"""
Sample: Agentic Multi-Source RAG System

This demonstrates how an agent intelligently decides which document sources
(text, markdown, web) to search based on the question context.

Run: python 08-agentic-rag-systems/samples/multi_source_rag.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI

load_dotenv()


def get_embeddings_endpoint():
    """Get the Azure OpenAI endpoint, removing /openai/v1 suffix if present."""
    endpoint = os.getenv("AI_ENDPOINT", "")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "")
    return endpoint


documents = [
    # Text sources
    Document(
        page_content="LangChain simplifies building AI applications with modular components",
        metadata={"source_type": "text", "source": "article.txt", "date": "2024-01-15"},
    ),
    Document(
        page_content="Vector databases store embeddings for semantic search capabilities",
        metadata={"source_type": "text", "source": "notes.txt", "date": "2024-01-20"},
    ),
    # Markdown sources
    Document(
        page_content="# Getting Started\n\nInstall LangChain using pip install langchain",
        metadata={"source_type": "markdown", "source": "README.md", "date": "2024-02-01"},
    ),
    Document(
        page_content="## Best Practices\n\nAlways validate user input before processing",
        metadata={"source_type": "markdown", "source": "GUIDE.md", "date": "2024-02-05"},
    ),
    # Web sources
    Document(
        page_content="LangChain provides Python and JavaScript libraries for building LLM applications",
        metadata={"source_type": "web", "source": "https://python.langchain.com", "date": "2024-02-10"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for accurate AI responses",
        metadata={"source_type": "web", "source": "https://docs.langchain.com/rag", "date": "2024-02-15"},
    ),
]


def main():
    print("🗂️  Agentic Multi-Source RAG System\n")
    print("=" * 80 + "\n")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    print("📚 Loading multi-source knowledge base...")
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    print("✅ Knowledge base ready!\n")

    # Create source-specific retrieval tools
    @tool
    def search_all_sources(query: str) -> str:
        """Search across ALL document sources (text files, markdown docs, web pages). Use this when you need comprehensive information from any available source."""
        results = vector_store.similarity_search(query, k=3)
        return "\n\n".join(
            f"[{i + 1}] [{doc.metadata['source_type'].upper()}] {doc.metadata['source']} ({doc.metadata['date']})\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(results)
        )

    @tool
    def search_text_files(query: str) -> str:
        """Search ONLY text files (.txt). Use this when you specifically need information from plain text sources like articles or notes."""
        all_results = vector_store.similarity_search(query, k=10)
        filtered = [doc for doc in all_results if doc.metadata["source_type"] == "text"]
        return "\n\n".join(
            f"[{i + 1}] {doc.metadata['source']} ({doc.metadata['date']})\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(filtered[:3])
        )

    @tool
    def search_markdown_docs(query: str) -> str:
        """Search ONLY markdown documentation (.md). Use this when you need documentation, guides, or README files."""
        all_results = vector_store.similarity_search(query, k=10)
        filtered = [doc for doc in all_results if doc.metadata["source_type"] == "markdown"]
        return "\n\n".join(
            f"[{i + 1}] {doc.metadata['source']} ({doc.metadata['date']})\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(filtered[:3])
        )

    @tool
    def search_web_pages(query: str) -> str:
        """Search ONLY web pages. Use this when you need information from online sources or official documentation websites."""
        all_results = vector_store.similarity_search(query, k=10)
        filtered = [doc for doc in all_results if doc.metadata["source_type"] == "web"]
        return "\n\n".join(
            f"[{i + 1}] {doc.metadata['source']} ({doc.metadata['date']})\n"
            f"Content: {doc.page_content}"
            for i, doc in enumerate(filtered[:3])
        )

    # Create agent with all source-specific tools
    agent = create_agent(
        model,
        tools=[search_all_sources, search_text_files, search_markdown_docs, search_web_pages],
        system_prompt="You are a helpful assistant with access to multiple document sources: text files, markdown documentation, and web pages. Choose the appropriate search tool based on the type of information needed. For general knowledge questions, answer directly without searching.",
    )

    print("=" * 80 + "\n")

    # Check CI mode
    if os.getenv("CI") == "true":
        print("Running in CI mode\n")

        test_questions = [
            "What is LangChain?",  # Agent might search all sources
            "How do I get started with installation?",  # Agent might search markdown
            "Tell me about vector databases",  # Agent might search text or all
        ]

        for question in test_questions:
            print(f"❓ Question: {question}\n")

            response = agent.invoke({
                "messages": [HumanMessage(content=question)],
            })

            last_message = response["messages"][-1]
            print(f"🤖 Answer: {last_message.content}\n")

            # Show which tool was used
            tool_use = next(
                (msg for msg in response["messages"]
                 if isinstance(msg, AIMessage) and msg.tool_calls and len(msg.tool_calls) > 0),
                None
            )
            if tool_use and tool_use.tool_calls:
                print(f"✅ Agent chose: {tool_use.tool_calls[0]['name']}\n")

            print("─" * 80 + "\n")

        print("✅ Agentic multi-source RAG working correctly!")
        return

    # Interactive mode
    print("🤖 The agent can intelligently choose between:")
    print("   • search_all_sources - Search all document types")
    print("   • search_text_files - Search only .txt files")
    print("   • search_markdown_docs - Search only .md files")
    print("   • search_web_pages - Search only web sources")
    print("\nThe agent will decide which source(s) to search based on your question!\n")

    while True:
        try:
            question = input("\n❓ Question (or 'quit'): ").strip()
        except EOFError:
            break

        if question.lower() == "quit":
            break

        print("\n🤔 Agent is analyzing your question and choosing source(s)...\n")

        response = agent.invoke({
            "messages": [HumanMessage(content=question)],
        })

        last_message = response["messages"][-1]
        print("─" * 80)
        print(f"\n🤖 Answer: {last_message.content}\n")

        # Show which tool(s) the agent chose
        tool_messages = [
            msg for msg in response["messages"]
            if isinstance(msg, AIMessage) and msg.tool_calls and len(msg.tool_calls) > 0
        ]

        if tool_messages:
            print("📊 Agent Decision:")
            for msg in tool_messages:
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"   ✓ Used: {call['name']}")

        print("\n" + "─" * 80)

    print("\n✅ Complete!")
    print("\n💡 Key Insights:")
    print("   ✓ Agent intelligently chooses which source types to search")
    print("   ✓ No manual source selection needed - agent decides based on context")
    print("   ✓ More efficient than always searching all sources")
    print("   ✓ Demonstrates multi-tool agent decision-making")


if __name__ == "__main__":
    main()

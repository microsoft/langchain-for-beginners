"""
Sample: Agentic RAG with Citation Generator

This demonstrates how an agent decides when to search documents
and automatically generates citations for retrieved information.

Run: python 08-agentic-rag-systems/samples/citation_rag.py
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


# Knowledge base with rich metadata
knowledge_base = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
        metadata={
            "title": "Introduction to Machine Learning",
            "section": "Chapter 1",
            "page": 12,
            "author": "AI Research Team",
        },
    ),
    Document(
        page_content="Supervised learning involves training a model on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs. Common applications include classification and regression problems.",
        metadata={
            "title": "Supervised Learning Fundamentals",
            "section": "Chapter 2",
            "page": 34,
            "author": "AI Research Team",
        },
    ),
    Document(
        page_content="Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process and transform data. Deep learning uses neural networks with many layers.",
        metadata={
            "title": "Neural Networks Explained",
            "section": "Chapter 3",
            "page": 56,
            "author": "Deep Learning Group",
        },
    ),
    Document(
        page_content="Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Techniques include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis.",
        metadata={
            "title": "Natural Language Processing",
            "section": "Chapter 5",
            "page": 89,
            "author": "NLP Research Lab",
        },
    ),
    Document(
        page_content="Transfer learning involves taking a pre-trained model and fine-tuning it for a specific task. This approach saves time and resources while often achieving better performance than training from scratch.",
        metadata={
            "title": "Transfer Learning Techniques",
            "section": "Chapter 7",
            "page": 134,
            "author": "AI Research Team",
        },
    ),
]


def main():
    print("📝 Agentic RAG with Citation Generator\n")
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

    print("📚 Loading knowledge base with rich metadata...\n")

    vector_store = InMemoryVectorStore.from_documents(knowledge_base, embeddings)

    # Create retrieval tool that includes citations
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for information about machine learning, NLP, neural networks, and AI topics. Use this when you need specific technical information from the knowledge base. The tool returns results with citation numbers [1], [2], [3] that you should reference in your answer."""
        results = vector_store.similarity_search_with_score(query, k=3)

        # Format results with citation numbers
        formatted_results = []
        for index, (doc, score) in enumerate(results):
            relevance_percent = round((1 - score) * 100)
            formatted_results.append(
                f"[{index + 1}] {doc.page_content}\n"
                f"Source: {doc.metadata['title']} - {doc.metadata['section']} (Page {doc.metadata['page']})\n"
                f"Author: {doc.metadata['author']}\n"
                f"Relevance: {relevance_percent}%"
            )

        return "\n\n".join(formatted_results) if formatted_results else "No relevant documents found."

    # Create agent with citation-aware retrieval tool
    agent = create_agent(
        model,
        tools=[search_knowledge_base],
        system_prompt="You are a helpful assistant that provides accurate answers with citations. When you search the knowledge base, include citation numbers (e.g., [1], [2]) in your response to reference the sources. For general knowledge questions, answer directly without searching.",
    )

    print("✅ Agentic citation system ready!\n")
    print("=" * 80 + "\n")

    # Questions to test - mix of general knowledge and knowledge base questions
    questions = [
        "What is machine learning?",  # Should use retrieval
        "What is 2 + 2?",  # Should answer directly
        "Explain neural networks and deep learning",  # Should use retrieval
        "What color is the sky?",  # Should answer directly
        "What is NLP and what can it do?",  # Should use retrieval
    ]

    for question in questions:
        print(f"❓ Question: {question}\n")
        print("─" * 80 + "\n")

        response = agent.invoke({
            "messages": [HumanMessage(content=question)],
        })

        last_message = response["messages"][-1]
        print(f"🤖 Answer:\n{last_message.content}\n")

        # Check if agent used the retrieval tool
        tool_use = next(
            (msg for msg in response["messages"] 
             if isinstance(msg, AIMessage) and msg.tool_calls and len(msg.tool_calls) > 0),
            None
        )

        if tool_use:
            print("─" * 80)
            print("\n✅ Agent Decision: Used knowledge base search")
            print("   📚 Citations included in answer\n")
        else:
            print("─" * 80)
            print("\n✅ Agent Decision: Answered from general knowledge")
            print("   💡 No retrieval needed\n")

        print("=" * 80 + "\n")

    print("✅ Agentic Citation RAG Complete!\n")
    print("💡 Key Features:")
    print("   ✓ Agent decides when to search vs answer directly")
    print("   ✓ Automatic citation generation with [1], [2], [3] format")
    print("   ✓ Detailed source information (title, section, page, author)")
    print("   ✓ Relevance scores for each source")
    print("   ✓ Intelligent decision-making (no unnecessary searches)")
    print("\n📊 Comparison to Traditional Approach:")
    print("   • Traditional RAG: Always searches, even for '2 + 2'")
    print("   • Agentic: Only searches when knowledge base information is needed")
    print("   • Result: More efficient, lower cost, better user experience")


if __name__ == "__main__":
    main()

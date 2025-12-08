"""
When to Use RAG: Decision Framework Demo

This example demonstrates the decision framework for choosing between:
1. Prompt Engineering (small, static data)
2. Agentic RAG (large, dynamic knowledge base with intelligent retrieval)

Run: python 08-agentic-rag-systems/code/01_when_to_use_rag.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does agent decision-making improve efficiency in agentic RAG?"
- "What factors should I consider when choosing between RAG and prompt engineering?"
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
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


def main():
    print("🎯 When to Use RAG: Decision Framework Demo\n")
    print("=" * 80 + "\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # ============================================================================
    # Scenario 1: Small FAQ (Use Prompt Engineering)
    # ============================================================================

    print("📋 SCENARIO 1: Small FAQ Bot")
    print("─" * 80)
    print("\nProblem: Answer 5 common questions about a product")
    print("Data size: 5 questions/answers (fits easily in prompt)")
    print("Update frequency: Rarely changes")
    print("\n✅ BEST APPROACH: Prompt Engineering\n")

    # Small knowledge base that fits in a prompt
    faq_context = """
Product FAQ:
Q: What is the return policy?
A: 30-day money-back guarantee, no questions asked.

Q: How long is shipping?
A: 2-3 business days for standard, 1 day for express.

Q: Is there a warranty?
A: Yes, 1-year manufacturer warranty on all products.

Q: Do you ship internationally?
A: Yes, we ship to over 100 countries worldwide.

Q: What payment methods do you accept?
A: We accept all major credit cards, PayPal, and Apple Pay.
"""

    faq_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful customer service assistant. Answer questions based on this FAQ:\n\n{context}",
        ),
        ("human", "{question}"),
    ])

    faq_chain = faq_prompt | model

    faq_question = "What's your return policy?"
    print(f'Question: "{faq_question}"\n')

    faq_response = faq_chain.invoke({
        "context": faq_context,
        "question": faq_question,
    })

    print("Answer:", faq_response.content)

    print("\n💡 Why Prompt Engineering works here:")
    print("   • Small dataset (5 Q&As) fits easily in prompt")
    print("   • No search needed - all context is relevant")
    print("   • Simple to maintain - just update the string")
    print("   • Fast and cost-effective")

    print("\n" + "=" * 80 + "\n")

    # ============================================================================
    # Scenario 2: Large Knowledge Base (Use RAG)
    # ============================================================================

    print("📚 SCENARIO 2: Company Documentation Bot")
    print("─" * 80)
    print("\nProblem: Answer questions from 1,000+ documentation pages")
    print("Data size: Too large to fit in prompt (exceeds context window)")
    print("Update frequency: Documentation changes frequently")
    print("\n✅ BEST APPROACH: Agentic RAG (Agent + Retrieval Tool)\n")

    # Simulate a large knowledge base (in reality, this would be 1000s of docs)
    docs = [
        Document(
            page_content="The API authentication uses OAuth 2.0 with bearer tokens. Tokens expire after 24 hours.",
            metadata={"source": "api-auth.md", "category": "API"},
        ),
        Document(
            page_content="Database migrations are handled automatically by the ORM. Use 'python manage.py migrate' to apply pending migrations.",
            metadata={"source": "database.md", "category": "Database"},
        ),
        Document(
            page_content="Deployment to production requires approval from two team leads. Use the GitHub Actions workflow.",
            metadata={"source": "deployment.md", "category": "DevOps"},
        ),
        Document(
            page_content="Error logging is handled by Sentry. All errors are automatically tracked and reported to the #alerts channel.",
            metadata={"source": "monitoring.md", "category": "DevOps"},
        ),
        Document(
            page_content="The frontend uses React 18 with TypeScript. All components should be functional with hooks.",
            metadata={"source": "frontend.md", "category": "Frontend"},
        ),
        Document(
            page_content="CSS styling uses Tailwind CSS. Avoid inline styles and use utility classes instead.",
            metadata={"source": "styling.md", "category": "Frontend"},
        ),
        Document(
            page_content="API rate limiting is 100 requests per minute per user. Exceeding this returns a 429 status code.",
            metadata={"source": "api-limits.md", "category": "API"},
        ),
        Document(
            page_content="User passwords are hashed using bcrypt with 12 rounds. Never store passwords in plain text.",
            metadata={"source": "security.md", "category": "Security"},
        ),
    ]

    print("Creating vector store from documents...")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    # Create retrieval tool from vector store
    @tool
    def search_docs(query: str) -> str:
        """Search company documentation for technical information about APIs, authentication, rate limits, deployment, etc."""
        results = vector_store.similarity_search(query, k=2)
        return "\n\n".join(
            f"[{doc.metadata['source']}]: {doc.page_content}"
            for doc in results
        )

    # Create agent with retrieval tool
    agent = create_agent(
        model,
        tools=[search_docs],
        system_prompt="You are a helpful technical documentation assistant. Search the docs when you need specific technical information.",
    )

    rag_question = "How does API authentication work?"
    print(f'\nQuestion: "{rag_question}"\n')

    rag_response = agent.invoke({
        "messages": [HumanMessage(content=rag_question)],
    })

    last_message = rag_response["messages"][-1]
    print("Answer:", last_message.content)

    print("\n💡 Why Agentic RAG works here:")
    print("   • Large dataset (1000s of docs) - can't fit in prompt")
    print("   • Agent decides when to search vs answer directly")
    print("   • Search capability - finds relevant 2 docs out of thousands")
    print("   • Easy to update - just add/remove documents from vector store")
    print("   • Source attribution - know which docs were used")
    print("   • Scalable - works with millions of documents")
    print("   • Intelligent - only searches when necessary")

    print("\n" + "=" * 80 + "\n")

    # ============================================================================
    # Scenario 3: When to Use Fine-Tuning (Not Demonstrated)
    # ============================================================================

    print("🎨 SCENARIO 3: Company-Specific Code Style")
    print("─" * 80)
    print("\nProblem: Generate code following company-specific patterns")
    print("Goal: Change model behavior, not add facts")
    print("Examples:")
    print("  • Always use async/await (never .then())")
    print("  • Specific error handling patterns")
    print("  • Company-specific naming conventions")
    print("  • Custom logging format")
    print("\n✅ BEST APPROACH: Fine-Tuning\n")

    print("💡 Why Fine-Tuning works here:")
    print("   • Teaching BEHAVIOR (coding style), not FACTS (documentation)")
    print("   • Need consistent patterns across all generated code")
    print("   • Examples can be collected from existing codebase")
    print("   • Style doesn't change frequently (worth the training cost)")

    print("\n❌ Why RAG wouldn't work:")
    print("   • RAG adds information, doesn't change how the model writes")
    print("   • Can't search for 'coding style' - it's a pattern, not content")
    print("   • Would need to retrieve style examples for every request (inefficient)")

    print("\n" + "=" * 80 + "\n")

    # ============================================================================
    # Decision Framework Summary
    # ============================================================================

    print("🎓 DECISION FRAMEWORK SUMMARY")
    print("─" * 80 + "\n")

    print("Step 1: Does your information fit in a prompt (< 8,000 tokens)?")
    print("  ✅ YES → Use PROMPT ENGINEERING (Scenario 1)")
    print("  ❌ NO  → Continue to Step 2\n")

    print("Step 2: Do you need to ADD INFORMATION or CHANGE BEHAVIOR?")
    print("  📚 Add information → Use RAG (Scenario 2)")
    print("  🎨 Change behavior → Use FINE-TUNING (Scenario 3)\n")

    print("Step 3: Does your information update frequently?")
    print("  ✅ YES → Definitely use RAG (easy to update)")
    print("  ❌ NO  → Either works, but RAG is cheaper\n")

    print("Step 4: Do you need to cite sources?")
    print("  ✅ YES → Use RAG (tracks source documents)")
    print("  ❌ NO  → Either approach works\n")

    print("=" * 80 + "\n")

    print("📋 Quick Reference:")
    print("─" * 80)
    print("\nPrompt Engineering:")
    print("  • Best for: Small, static data (< 8K tokens)")
    print("  • Example: FAQ bot with 5-10 questions")
    print("  • Pros: Simple, fast, cheap")
    print("  • Cons: Doesn't scale, hard to update large datasets\n")

    print("RAG (Retrieval Augmented Generation):")
    print("  • Best for: Large, searchable knowledge bases")
    print("  • Example: 1000+ documentation pages")
    print("  • Pros: Scalable, easy updates, source attribution")
    print("  • Cons: Requires vector store, retrieval overhead\n")

    print("Fine-Tuning:")
    print("  • Best for: Changing model behavior/style")
    print("  • Example: Company-specific code generation")
    print("  • Pros: Changes how model writes/reasons")
    print("  • Cons: Expensive, slow, hard to update\n")

    print("=" * 80)
    print("\n✅ In this course, we focus on RAG because it's the most versatile")
    print("   approach for building production AI applications with custom data!")


if __name__ == "__main__":
    main()

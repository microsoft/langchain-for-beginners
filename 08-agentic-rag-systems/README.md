# Building Agentic RAG Systems

In this chapter, you'll learn to build **Agentic RAG** systems where AI agents intelligently decide when and how to search your documents to answer questions. Unlike traditional RAG that always searches regardless of need, agentic RAG gives your AI the autonomy to determine whether retrieval is necessary—answering directly when it has the knowledge, or searching your documents when additional context is needed.

You'll combine everything you've learned to build intelligent question-answering systems that provide accurate, sourced answers from custom knowledge bases.

## Prerequisites

- Completed [Function Calling & Tools](../04-function-calling-tools/README.md)
- Completed [Getting Started with Agents](../05-agents/README.md)
- Completed [Documents, Embeddings & Semantic Search](../07-documents-embeddings-semantic-search/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Understand the difference between Agentic RAG and Traditional RAG
- ✅ Build agents that decide when to search vs answer directly
- ✅ Create retrieval tools from vector stores
- ✅ Implement intelligent document search with agent decision-making
- ✅ Handle context and citations in agentic systems
- ✅ Apply the decision framework (RAG vs Prompt Engineering)

---

## 📖 The Smart Student Analogy

**Imagine three types of students taking an exam:**

**Closed-Book Exam (Standard LLM)**:
- ❌ Student relies only on memorized knowledge
- ❌ Can't look up specific facts
- ❌ May give wrong answers confidently

**Open-Book Exam with No Strategy (Traditional RAG)**:
- ✅ Student can reference textbook during exam
- ❌ Looks up the textbook for EVERY question, even "What is 2+2?"
- ❌ Wastes time searching when they already know the answer

**Smart Open-Book Exam (Agentic RAG)**:
- ✅ Student can reference textbook during exam
- ✅ **Decides** when to look things up vs answering from knowledge
- ✅ "What is 2+2?" → Answers directly (no search needed)
- ✅ "What was our company's Q3 revenue?" → Searches documents

**This is the power of Agentic RAG!** The agent makes intelligent decisions about when retrieval is necessary.

---

## 🤖 Agentic RAG vs Traditional RAG

### The Key Difference

**Traditional RAG**:
```
User Question → ALWAYS Search → Retrieve Docs → Generate Answer
```
Every question triggers a search, even if the agent already knows the answer.

**Agentic RAG**:
```
User Question → Agent Decides → [Search if needed] → Generate Answer
```
The agent uses reasoning to determine whether retrieval is necessary.

### Example: The Difference in Action

**"What is 2 + 2?"**
- Traditional RAG: Searches vector store, retrieves irrelevant docs, answers "4" (wasted search)
- Agentic RAG: Answers immediately "4" (no search needed)

**"What was our company's revenue in Q3 2024?"**
- Traditional RAG: Searches vector store, retrieves financial docs, answers "$1.2M"
- Agentic RAG: Searches documents, answers "$1.2M based on Q3 financial report"

### Benefits of Agentic RAG

| Benefit | Traditional RAG | Agentic RAG |
|---------|-----------------|-------------|
| **Efficiency** | Searches every time | Only searches when needed |
| **Speed** | Slow for simple questions | Fast for simple, thorough for complex |
| **Cost** | Embedding + search cost on every query | Lower cost - searches only when necessary |
| **Intelligence** | Rigid, predictable | Adaptive, makes decisions |

---

## 🏗️ Building Agentic RAG

### Example 1: Traditional RAG (Always-Search Pattern)

First, let's see traditional RAG for comparison.

**Code**: [`code/01_traditional_rag.py`](./code/01_traditional_rag.py)  
**Run**: `python 08-agentic-rag-systems/code/01_traditional_rag.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Create sample documents
docs = [
    Document(page_content="Our Q3 2024 revenue was $1.2 million", metadata={"source": "financials.txt"}),
    Document(page_content="The API uses OAuth 2.0 authentication", metadata={"source": "api-docs.txt"}),
    Document(page_content="Company headquarters is in Seattle, WA", metadata={"source": "about.txt"}),
]

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Create the RAG chain (ALWAYS searches)
model = ChatOpenAI(model=os.getenv("AI_MODEL"))

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context:

Context: {context}

Question: {input}
""")

combine_docs_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# This searches EVERY time, even for general knowledge
queries = [
    "What is 2 + 2?",           # No need to search!
    "What was Q3 revenue?",     # Needs document search
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    result = rag_chain.invoke({"input": query})
    print(f"📄 Retrieved {len(result['context'])} documents")  # Always retrieves!
    print(f"Answer: {result['answer']}")
```

**The Problem**: Traditional RAG searches for "What is 2 + 2?" when it doesn't need to!

---

### Example 2: Agentic RAG (Smart Decision-Making)

Now let's build the intelligent version where the agent decides when to search.

**Code**: [`code/02_agentic_rag.py`](./code/02_agentic_rag.py)  
**Run**: `python 08-agentic-rag-systems/code/02_agentic_rag.py`

**Example code:**

```python
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Create sample documents
docs = [
    Document(page_content="Our Q3 2024 revenue was $1.2 million", metadata={"source": "financials.txt"}),
    Document(page_content="The API uses OAuth 2.0 authentication", metadata={"source": "api-docs.txt"}),
    Document(page_content="Company headquarters is in Seattle, WA", metadata={"source": "about.txt"}),
]

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

# Create retrieval tool - agent decides when to use it!
@tool
def search_company_docs(query: str) -> str:
    """Search company documentation for specific information about 
    revenue, API details, company information, etc.
    Use this when you need to look up specific company data.
    """
    results = vector_store.similarity_search(query, k=2)
    return "\n\n".join([
        f"[{doc.metadata['source']}]: {doc.page_content}" 
        for doc in results
    ])

# Create agent with the retrieval tool
agent = create_agent(
    model=os.getenv("AI_MODEL"),
    tools=[search_company_docs],
    system_prompt="You are a helpful assistant. Only search documents when you need specific company information.",
)

# Test: Agent decides when to search
queries = [
    "What is 2 + 2?",           # Agent answers directly (no search)
    "What was Q3 revenue?",     # Agent searches documents
    "What is the capital of France?",  # Agent answers directly
    "Where is the company headquarters?",  # Agent searches documents
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    # Check if agent used the search tool
    tool_used = any(
        hasattr(msg, 'tool_calls') and msg.tool_calls 
        for msg in response["messages"]
    )
    
    if tool_used:
        print("📚 Agent searched documents")
    else:
        print("🧠 Agent answered from knowledge")
    
    print(f"Answer: {response['messages'][-1].content}")
```

### Expected Output

```
🔍 Query: What is 2 + 2?
🧠 Agent answered from knowledge
Answer: 4

🔍 Query: What was Q3 revenue?
📚 Agent searched documents
Answer: According to our financial documents, Q3 2024 revenue was $1.2 million.

🔍 Query: What is the capital of France?
🧠 Agent answered from knowledge
Answer: The capital of France is Paris.

🔍 Query: Where is the company headquarters?
📚 Agent searched documents
Answer: Company headquarters is in Seattle, WA.
```

---

### Example 3: Complete Agentic RAG System

**Code**: [`code/03_complete_rag.py`](./code/03_complete_rag.py)  
**Run**: `python 08-agentic-rag-systems/code/03_complete_rag.py`

**Example code:**

```python
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

def build_rag_system(docs_path: str):
    """Build a complete agentic RAG system."""
    
    # 1. Load documents
    loader = TextLoader(docs_path)
    raw_docs = loader.load()
    
    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    docs = splitter.split_documents(raw_docs)
    print(f"Loaded {len(docs)} document chunks")
    
    # 3. Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # 4. Create retrieval tool
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for relevant information.
        Use this when you need specific details from the documentation.
        """
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            return "No relevant documents found."
        
        return "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in results
        ])
    
    # 5. Create agent
    agent = create_agent(
        model=os.getenv("AI_MODEL"),
        tools=[search_knowledge_base],
        system_prompt="""You are a knowledgeable assistant with access to a document search tool.
        
Guidelines:
- Answer general knowledge questions directly without searching
- Use the search tool for specific information from documents
- Always cite sources when using retrieved information
- If you can't find relevant information, say so clearly
""",
    )
    
    return agent

def main():
    # Build the system
    agent = build_rag_system("./data/knowledge_base.txt")
    
    # Interactive Q&A
    print("\n🤖 Agentic RAG System Ready!")
    print("Ask questions about your documents or general knowledge.\n")
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        response = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        print(f"\nAssistant: {response['messages'][-1].content}\n")

if __name__ == "__main__":
    main()
```

---

## 📋 When to Use RAG vs Prompt Engineering

| Criteria | Prompt Engineering | RAG |
|----------|-------------------|-----|
| **Data Size** | Small (fits in prompt) | Large (1000s of docs) |
| **Update Frequency** | Rarely changes | Frequently updates |
| **Need Citations** | No | Yes |
| **Example** | FAQ bot with 20 questions | Customer support with 10,000 manuals |

**Decision Tree**:
1. Fits easily in prompt? → **Prompt Engineering**
2. Large knowledge base that doesn't fit? → **RAG**
3. Updates frequently? → **RAG**
4. Need source citations? → **RAG**

---

## 🎓 Key Takeaways

- **Traditional RAG** always searches, even when unnecessary
- **Agentic RAG** lets the agent decide when to search
- Create **retrieval tools** from vector stores using `@tool`
- Use `create_agent()` to build intelligent decision-making
- **Cite sources** when using retrieved information
- Choose **RAG vs Prompt Engineering** based on data size and update frequency

---

## 📦 Dependencies

```bash
pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv
```

---

## 🗺️ Navigation

[← Previous: Documents & Embeddings](../07-documents-embeddings-semantic-search/README.md) | [Back to Main](../README.md)

---

## 🎉 Congratulations!

You've completed the LangChain for Beginners course! You've learned:

1. ✅ Setting up your Python environment for LangChain
2. ✅ Understanding LangChain's architecture and components
3. ✅ Working with chat models and multi-turn conversations
4. ✅ Creating prompts, templates, and structured outputs
5. ✅ Building tools with function calling
6. ✅ Creating autonomous agents with the ReAct pattern
7. ✅ Connecting to MCP servers for external tools
8. ✅ Processing documents with embeddings and vector stores
9. ✅ Building intelligent Agentic RAG systems

**What's next?**
- Explore more [LangChain integrations](https://python.langchain.com/docs/integrations/)
- Learn about [LangGraph](https://langchain-ai.github.io/langgraph/) for complex workflows
- Build production applications with [LangSmith](https://docs.smith.langchain.com/)

---

## 💬 Questions?

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
